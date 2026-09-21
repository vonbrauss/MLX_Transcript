# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the Apple Silicon macOS application."""

from pathlib import Path
import importlib.util
import os
import re
import shutil
import subprocess

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, get_package_paths

project = Path(SPEC).resolve().parents[1]


def read_version_constant(name):
    """Read one constant out of app/version.py without importing anything."""
    source = (project / "app" / "version.py").read_text(encoding="utf-8")
    found = re.search(rf'^{name} = "([^"]+)"', source, re.MULTILINE)
    if found is None:
        raise SystemExit(f"app/version.py does not define {name}")
    return found.group(1)


VERSION = read_version_constant("VERSION")
BUILD_NUMBER = read_version_constant("BUILD_NUMBER")


def load_ffmpeg_requirements():
    """Load the shared FFmpeg contract by path.

    The recipe must not depend on the project being importable, so the module
    is loaded from its file rather than through sys.path.
    """
    location = project / "packaging" / "ffmpeg_requirements.py"
    spec = importlib.util.spec_from_file_location("mlx_ffmpeg_requirements", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ffmpeg_requirements = load_ffmpeg_requirements()

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

# --------------------------------------------------------------------- FFmpeg
#
# MLX Transcript only ever demuxes, probes, and decodes audio. It encodes
# nothing, so it never needs the GPL encoders a stock Homebrew FFmpeg enables.
# Bundling that build anyway would put a source-distribution obligation on
# every release, so the configuration is inspected here and a GPL build is
# refused unless the person building says otherwise in as many words.

GPL_CONFIGURE_FLAGS = ("--enable-gpl", "--enable-nonfree")


def ffmpeg_configuration(executable):
    """Return the configure line the located FFmpeg reports, or an empty string."""
    try:
        result = subprocess.run(
            [executable, "-version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in (result.stdout or "").splitlines():
        if line.strip().startswith("configuration:"):
            return line.split(":", 1)[1].strip()
    return ""


def gpl_flags_in(configuration):
    return [flag for flag in GPL_CONFIGURE_FLAGS if flag in configuration]


def record_media_tool_provenance(entries, validation):
    """Write what was bundled so the notices can describe the real binary."""
    report = project / "build" / "ffmpeg-configuration.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n\n".join(f"{name}: {path}\nconfiguration: {configuration}"
                    for name, path, configuration in entries)
        + "\n\nValidation\n"
        + "\n".join(f"  {line}" for line in validation)
        + "\n",
        encoding="utf-8",
    )


if os.environ.get("MLX_TRANSCRIPT_BUNDLE_FFMPEG", "1") == "1":
    allow_gpl = os.environ.get("MLX_TRANSCRIPT_ALLOW_GPL_FFMPEG") == "1"
    provenance = []
    media_tools = {}
    for executable in ("ffmpeg", "ffprobe"):
        located = os.environ.get(f"MLX_TRANSCRIPT_{executable.upper()}") or shutil.which(
            executable
        )
        if not located:
            raise SystemExit(
                f"{executable} was not found; install FFmpeg, point "
                f"MLX_TRANSCRIPT_{executable.upper()} at one, or set "
                "MLX_TRANSCRIPT_BUNDLE_FFMPEG=0"
            )
        configuration = ffmpeg_configuration(located)
        offending = gpl_flags_in(configuration)
        if offending and not allow_gpl:
            raise SystemExit(
                f"\nRefusing to bundle {located}.\n"
                f"It was built with {' '.join(offending)}, which makes the "
                "binary GPL and puts a source-distribution obligation on every "
                "release.\n\n"
                "MLX Transcript only decodes and probes, so it does not need "
                "those components. Build or install an FFmpeg configured with "
                "--disable-gpl --disable-nonfree and point "
                "MLX_TRANSCRIPT_FFMPEG / MLX_TRANSCRIPT_FFPROBE at it.\n\n"
                "Set MLX_TRANSCRIPT_ALLOW_GPL_FFMPEG=1 to override for a local "
                "build that will not be published.\n"
            )
        binaries.append((located, "bin"))
        provenance.append((executable, located, configuration))
        media_tools[executable] = located

    # The first standalone build shipped an FFmpeg with no s16le muxer and
    # failed on the user's first clip. Nothing is packaged now until the
    # located binaries have actually decoded a file to both raw PCM formats
    # the application asks for at runtime.
    try:
        validation = ffmpeg_requirements.verify_media_tools(
            media_tools["ffmpeg"], media_tools["ffprobe"], allow_gpl=allow_gpl
        )
    except ffmpeg_requirements.MediaToolError as error:
        raise SystemExit(
            "\nRefusing to package: the located FFmpeg cannot do what MLX "
            f"Transcript needs.\n\n{error}\n\n"
            "Rebuild it with the recipe in docs/BUILDING.md, or run:\n"
            "  scripts/finish_release_prep.sh ffmpeg\n"
        )
    for line in validation:
        print(f"  ffmpeg check: {line}")
    record_media_tool_provenance(provenance, validation)

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
icon_path = project / "packaging" / "MLX Transcript.icns"
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
    icon=str(icon_path) if icon_path.is_file() else None,
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
    icon=str(icon_path) if icon_path.is_file() else None,
    bundle_identifier="com.vonbrauss.mlxtranscript",
    info_plist={
        "CFBundleDisplayName": "MLX Transcript",
        "CFBundleName": "MLX Transcript",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": BUILD_NUMBER,
        "LSMinimumSystemVersion": "14.0",
        "LSApplicationCategoryType": "public.app-category.productivity",
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "Copyright © 2026 MLX Transcript contributors",
    },
)
