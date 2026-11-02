"""modi-only os_utils tests (``--disable-ffmpeg-check`` of jasna-framegen).

modi 独自。上流の tests/test_os_utils.py には追記しない。
"""
from jasna import os_utils


def test_check_required_executables_skips_checks_when_disabled(monkeypatch) -> None:
    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run must not run when the ffmpeg check is disabled")

    def fail_find(exe):
        raise AssertionError("find_executable must not run when the ffmpeg check is disabled")

    monkeypatch.setattr(os_utils.subprocess, "run", fail_run)
    monkeypatch.setattr(os_utils, "find_executable", fail_find)

    # Returns without raising or exiting; backs jasna-framegen --disable-ffmpeg-check.
    os_utils.check_required_executables(disable_ffmpeg_check=True)
