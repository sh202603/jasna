"""modi-only: the TensorRT-RTX flavor tag in the engine system suffix.

modi 独自。上流の tests/test_torch_tensorrt_export.py には追記しない。
"""
import os
from unittest.mock import patch

from jasna import engine_paths
from jasna.engine_paths import engine_system_suffix


class TestEngineSystemSuffixRtx:
    def test_linux_rtx(self):
        with patch.object(os, "name", "posix"), patch.object(engine_paths, "_trt_flavor_cache", "rtx"):
            assert engine_system_suffix() == ".rtx.linux"

    def test_windows_rtx(self):
        with patch.object(os, "name", "nt"), patch.object(engine_paths, "_trt_flavor_cache", "rtx"):
            assert engine_system_suffix() == ".rtx.win"

    def test_standard_flavor_has_no_tag(self):
        with patch.object(os, "name", "posix"), patch.object(engine_paths, "_trt_flavor_cache", "standard"):
            assert engine_system_suffix() == ".linux"
