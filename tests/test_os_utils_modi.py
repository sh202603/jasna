"""modi-only os_utils tests (``--disable-ffmpeg-check`` of jasna-postprocess).

modi 独自。上流の tests/test_os_utils.py には追記しない。
"""
from jasna import system_requirements


def test_check_required_executables_skips_checks_when_disabled(monkeypatch) -> None:
    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run must not run when the ffmpeg check is disabled")

    def fail_status():
        raise AssertionError("ffprobe must not be looked up when the ffmpeg check is disabled")

    monkeypatch.setattr("subprocess.run", fail_run)
    monkeypatch.setattr(system_requirements, "ffprobe_status", fail_status)

    # Returns without raising or exiting; backs jasna-postprocess --disable-ffmpeg-check.
    system_requirements.check_required_executables(disable_ffmpeg_check=True)
