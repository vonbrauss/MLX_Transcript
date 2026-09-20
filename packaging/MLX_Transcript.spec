# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the Apple Silicon macOS application."""

from pathlib import Path
import os
import shutil

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, get_package_paths

project = Path(SPEC).resolve().parents[1]

datas = []
datas.append((str(project / "app" / "assets"), "app/assets"))
binaries = []
hiddenimports = []

# Avoid importing every MLX submodule while PyInstaller analyzes the build.
# MLX initializes Metal during import, which is unsafe in headless builders,
# so every module name below is discovered from disk rather than by importing.
# PyInstaller follows mlx.core's linked dependencies and places libmlx and
# libjaccl in Frameworks. Adding the same dylibs again as package data makes
# macOS load two copies and nanobind aborts on duplicate DeviceType entries.
datas += [
    entry for entry in collect_data_files("mlx")
    if not entry[0].endswith(".dylib")
]
mlx_package_path = Path(get_package_paths("mlx")[1])
# libmlx is discovered from mlx.core's linkage. Its jaccl dependency uses an
# @rpath name that PyInstaller cannot resolve from the wheel's mlx/lib folder,
# so place that one dependency beside libmlx in Frameworks.
binaries.append((str(mlx_package_path / "lib" / "libjaccl.dylib"), "."))
datas += collect_data_files("mlx_whisper")
binaries += collect_dynamic_libs("sherpa_onnx")

# The native mlx.core extension imports these the moment it loads. Static
# analysis cannot see an import that happens inside a compiled module, so a
# bundle without them raises ImportError on "import mlx_whisper" and looks
# exactly like MLX not being installed. They are named explicitly, and the
# packaging test asserts they stay named.
REQUIRED_MLX_HIDDENIMPORTS = [
    "mlx",
    "mlx.core",
    "mlx._reprlib_fix",
    "mlx.__array_api_info",
    "mlx.nn",
    "mlx.optimizers",
    "mlx.utils",
]

# Folders inside the mlx wheel that ship headers, Metal shaders, and CMake
# files rather than importable Python. Walking them produced hidden imports
# such as "mlx.include.metal_cpp.SingleHeader.MakeSingleHeader", which cannot
# resolve and hid the entries that mattered among the warnings.
MLX_NON_MODULE_FOLDERS = {"include", "lib", "share", "core", "__pycache__"}


def discover_package_modules(root, package, skip_folders=frozenset()):
    """Return importable dotted module names found under a package folder.

    A folder only contributes modules when it carries an __init__.py, which is
    what keeps namespace-only directories and bundled data trees out.
    """
    root = Path(root)
    found = []
    for source in sorted(root.glob("*.py")):
        if source.stem != "__init__":
            found.append(f"{package}.{source.stem}")
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        if folder.name in skip_folders or not (folder / "__init__.py").is_file():
            continue
        found.append(f"{package}.{folder.name}")
        found.extend(
            discover_package_modules(folder, f"{package}.{folder.name}", skip_folders)
        )
    return found


mlx_path = mlx_package_path
mlx_modules = discover_package_modules(mlx_path, "mlx", MLX_NON_MODULE_FOLDERS)

hiddenimports += ["mlx_whisper", "sherpa_onnx"]
hiddenimports += REQUIRED_MLX_HIDDENIMPORTS
hiddenimports += mlx_modules

# Word timestamps are always requested, so the numba and scipy path in
# mlx_whisper.timing has to be in the bundle, not merely importable on the
# build machine. tiktoken carries a compiled extension the tokenizer needs.
hiddenimports += [
    "mlx_whisper.audio",
    "mlx_whisper.decoding",
    "mlx_whisper.load_models",
    "mlx_whisper.timing",
    "mlx_whisper.tokenizer",
    "mlx_whisper.transcribe",
    "mlx_whisper.whisper",
    "mlx_whisper.writers",
    "tiktoken",
    "tiktoken_ext",
    "tiktoken_ext.openai_public",
    "numba",
    "scipy.signal",
]

hiddenimports = sorted(set(hiddenimports))

# Local development builds may bundle Homebrew FFmpeg. Public releases must
# satisfy the license obligations of the exact FFmpeg build being distributed.
if os.environ.get("MLX_TRANSCRIPT_BUNDLE_FFMPEG", "1") == "1":
    for executable in ("ffmpeg", "ffprobe"):
        located = shutil.which(executable)
        if not located:
            raise SystemExit(f"{executable} was not found; install FFmpeg or set MLX_TRANSCRIPT_BUNDLE_FFMPEG=0")
        binaries.append((located, "bin"))

a = Analysis(
    [str(project / "main.py")],
    pathex=[str(project)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[
        "pytest",
        "torch",
        "mlx_whisper.torch_whisper",
        "tensorflow",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MLX Transcript",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch="arm64",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="MLX Transcript",
)
app = BUNDLE(
    coll,
    name="MLX Transcript.app",
    bundle_identifier="com.vonbrauss.mlxtranscript",
    info_plist={
        "CFBundleDisplayName": "MLX Transcript",
        "CFBundleName": "MLX Transcript",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "14.0",
        "LSApplicationCategoryType": "public.app-category.productivity",
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "Copyright © 2026 MLX Transcript contributors",
    },
)
