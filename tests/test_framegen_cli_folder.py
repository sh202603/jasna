"""Unit tests for jasna-framegen folder-batch planning (_plan_folder_jobs).

GPU-free: only filesystem classification + naming. Verifies videos-only
selection, the {original} output-pattern, and collision/empty-folder errors,
reusing jasna.media.media_files helpers (classify_folder / folder_output_path).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from jasna.framegen_cli import _plan_folder_jobs


def _touch(*paths: Path) -> None:
    for p in paths:
        p.touch()


def test_videos_selected_images_skipped(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _touch(in_dir / "a.mp4", in_dir / "b.mkv", in_dir / "c.png", in_dir / "d.jpg")
    out_dir = tmp_path / "out"

    jobs, skipped = _plan_folder_jobs(in_dir, out_dir, None)

    assert [vid.name for vid, _ in jobs] == ["a.mp4", "b.mkv"]
    # Default pattern: <stem>_out<ext> in the output dir.
    assert [out.name for _, out in jobs] == ["a_out.mp4", "b_out.mkv"]
    assert all(out.parent == out_dir for _, out in jobs)
    assert [img.name for img in skipped] == ["c.png", "d.jpg"]


def test_output_pattern_with_extension(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _touch(in_dir / "a.mp4", in_dir / "b.mkv")
    out_dir = tmp_path / "out"

    jobs, _ = _plan_folder_jobs(in_dir, out_dir, "{original}_2x.mkv")

    assert [out.name for _, out in jobs] == ["a_2x.mkv", "b_2x.mkv"]


def test_pattern_collision_raises(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _touch(in_dir / "a.mp4", in_dir / "b.mp4")
    out_dir = tmp_path / "out"

    # A constant pattern (no {original}) maps every input to the same output.
    with pytest.raises(ValueError, match="multiple inputs to the same output"):
        _plan_folder_jobs(in_dir, out_dir, "out.mkv")


def test_pattern_would_overwrite_input_raises(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _touch(in_dir / "a.mp4", in_dir / "b.mkv")

    # Output dir == input dir and a pattern that reproduces an input name.
    with pytest.raises(ValueError, match="overwrite an input file"):
        _plan_folder_jobs(in_dir, in_dir, "{original}.mp4")


def test_no_videos_only_images_raises(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _touch(in_dir / "c.png", in_dir / "d.jpg")
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="video-only"):
        _plan_folder_jobs(in_dir, out_dir, None)


def test_empty_folder_raises(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="No video files found"):
        _plan_folder_jobs(in_dir, out_dir, None)


def test_output_existing_file_raises(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    _touch(in_dir / "a.mp4")
    out_file = tmp_path / "out.txt"
    out_file.touch()

    with pytest.raises(ValueError, match="must be a folder"):
        _plan_folder_jobs(in_dir, out_file, None)
