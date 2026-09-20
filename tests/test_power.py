"""The sleep assertion held while a batch runs."""

from __future__ import annotations

from app.power import CAFFEINATE_COMMAND, SleepBlocker


class FakeProcess:
    """Stands in for the caffeinate child process."""

    def __init__(self) -> None:
        self.terminated = False
        self.waited = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return 0


class Spawner:
    """Records every command it is asked to launch."""

    def __init__(self, error: Exception | None = None) -> None:
        self.commands: list[list[str]] = []
        self.processes: list[FakeProcess] = []
        self.error = error

    def __call__(self, command: list[str]) -> FakeProcess:
        if self.error is not None:
            raise self.error
        self.commands.append(list(command))
        process = FakeProcess()
        self.processes.append(process)
        return process


def blocker(spawner: Spawner, platform: str = "darwin") -> SleepBlocker:
    return SleepBlocker(popen_factory=spawner, platform=platform)


def test_acquiring_starts_caffeinate():
    spawner = Spawner()
    assert blocker(spawner).acquire() is True
    assert spawner.commands == [CAFFEINATE_COMMAND]


def test_the_assertion_prevents_idle_and_disk_sleep():
    assert CAFFEINATE_COMMAND[0] == "caffeinate"
    assert "-i" in CAFFEINATE_COMMAND
    assert "-m" in CAFFEINATE_COMMAND


def test_releasing_terminates_the_process():
    spawner = Spawner()
    sleep_blocker = blocker(spawner)
    sleep_blocker.acquire()
    sleep_blocker.release()

    assert spawner.processes[0].terminated is True
    assert spawner.processes[0].waited is True
    assert sleep_blocker.active is False


def test_acquiring_twice_holds_only_one_assertion():
    spawner = Spawner()
    sleep_blocker = blocker(spawner)
    sleep_blocker.acquire()
    sleep_blocker.acquire()
    assert len(spawner.commands) == 1


def test_releasing_without_acquiring_is_safe():
    sleep_blocker = blocker(Spawner())
    sleep_blocker.release()
    assert sleep_blocker.active is False


def test_a_failed_spawn_is_reported_rather_than_raised():
    spawner = Spawner(error=OSError("caffeinate is missing"))
    sleep_blocker = blocker(spawner)
    assert sleep_blocker.acquire() is False
    assert sleep_blocker.active is False
    assert "caffeinate is missing" in sleep_blocker.last_error


def test_other_platforms_are_a_no_op():
    spawner = Spawner()
    sleep_blocker = blocker(spawner, platform="linux")
    assert sleep_blocker.supported is False
    assert sleep_blocker.acquire() is False
    assert spawner.commands == []


def test_the_context_manager_releases_on_exit():
    spawner = Spawner()
    sleep_blocker = blocker(spawner)
    with sleep_blocker:
        assert sleep_blocker.active is True
    assert sleep_blocker.active is False
    assert spawner.processes[0].terminated is True
