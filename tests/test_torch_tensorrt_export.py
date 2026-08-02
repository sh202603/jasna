import os
from unittest.mock import patch

from jasna import engine_paths
from jasna.trt.torch_tensorrt_export import engine_system_suffix, engine_precision_name


class TestEngineSystemSuffix:
    # OS-part assertions; pin the flavor so they hold in RTX venvs too
    # (flavor tagging itself is covered by test_trt_flavor.py).
    def test_windows(self):
        with patch.object(os, "name", "nt"), patch.object(engine_paths, "_trt_flavor_cache", "standard"):
            assert engine_system_suffix() == ".win"

    def test_linux(self):
        with patch.object(os, "name", "posix"), patch.object(engine_paths, "_trt_flavor_cache", "standard"):
            assert engine_system_suffix() == ".linux"

    def test_linux_rtx(self):
        with patch.object(os, "name", "posix"), patch.object(engine_paths, "_trt_flavor_cache", "rtx"):
            assert engine_system_suffix() == ".rtx.linux"


class TestEnginePrecisionName:
    def test_fp16(self):
        assert engine_precision_name(fp16=True) == "fp16"

    def test_fp32(self):
        assert engine_precision_name(fp16=False) == "fp32"
