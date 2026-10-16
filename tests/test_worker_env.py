import os
import sys

from jasna.restorer.worker_env import external_worker_env


def test_drops_pythonpath_and_own_library_dirs(tmp_path) -> None:
    own = os.path.join(sys.prefix, "lib", "python3", "site-packages", "nvidia", "cudnn", "lib")
    other = str(tmp_path / "system-libs")
    env = external_worker_env(
        {"PYTHONPATH": "/x", "LD_LIBRARY_PATH": os.pathsep.join([own, other, ""]), "KEEP": "1"}
    )

    assert "PYTHONPATH" not in env
    assert env["LD_LIBRARY_PATH"] == other
    assert env["KEEP"] == "1"


def test_removes_library_path_made_only_of_own_dirs() -> None:
    env = external_worker_env({"LD_LIBRARY_PATH": os.path.join(sys.prefix, "lib")})

    assert "LD_LIBRARY_PATH" not in env


def test_leaves_the_parent_environment_untouched() -> None:
    parent = {"PYTHONPATH": "/x"}
    external_worker_env(parent)

    assert parent == {"PYTHONPATH": "/x"}
