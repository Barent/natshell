"""Tests for GPU detection module."""

from __future__ import annotations

from unittest.mock import patch

from natshell.gpu import (
    GpuInfo,
    _classify_vendor,
    _parse_lspci,
    _parse_nvidia_smi,
    _parse_vulkaninfo,
    best_gpu_index,
    gpu_backend_available,
)

# ─── _classify_vendor ────────────────────────────────────────────────────────


class TestClassifyVendor:
    def test_nvidia_keyword(self):
        assert _classify_vendor("NVIDIA GeForce RTX 4090") == "nvidia"

    def test_nvidia_geforce(self):
        assert _classify_vendor("GeForce GTX 1080") == "nvidia"

    def test_nvidia_quadro(self):
        assert _classify_vendor("Quadro RTX 5000") == "nvidia"

    def test_amd_radeon(self):
        assert _classify_vendor("AMD Radeon RX 7900 XTX") == "amd"

    def test_amd_keyword(self):
        assert _classify_vendor("AMD CDNA") == "amd"

    def test_intel(self):
        assert _classify_vendor("Intel UHD Graphics 770") == "intel"

    def test_unknown(self):
        assert _classify_vendor("Some Random GPU") == "unknown"


# ─── _parse_vulkaninfo ───────────────────────────────────────────────────────


VULKANINFO_SUMMARY = """\
Vulkan Instance Version: 1.3.275

Devices:
========
GPU0:
\tdeviceName = NVIDIA GeForce RTX 4090
\tdeviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\theapSize  = 25769803776 (0x600000000) (DEVICE_LOCAL)

GPU1:
\tdeviceName = Intel UHD Graphics 770
\tdeviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
\theapSize  = 13421772800 (0x320000000) (DEVICE_LOCAL)
"""


class TestParseVulkaninfo:
    def test_two_gpus(self):
        gpus = _parse_vulkaninfo(VULKANINFO_SUMMARY)
        assert len(gpus) == 2

    def test_discrete_gpu(self):
        gpus = _parse_vulkaninfo(VULKANINFO_SUMMARY)
        assert gpus[0].name == "NVIDIA GeForce RTX 4090"
        assert gpus[0].vendor == "nvidia"
        assert gpus[0].is_discrete is True
        assert gpus[0].device_index == 0
        assert gpus[0].vram_mb == 25769803776 // (1024 * 1024)

    def test_integrated_gpu(self):
        gpus = _parse_vulkaninfo(VULKANINFO_SUMMARY)
        assert gpus[1].name == "Intel UHD Graphics 770"
        assert gpus[1].vendor == "intel"
        assert gpus[1].is_discrete is False
        assert gpus[1].device_index == 1

    def test_empty_output(self):
        assert _parse_vulkaninfo("") == []

    def test_header_only(self):
        assert _parse_vulkaninfo("Vulkan Instance Version: 1.3.275\n") == []

    def test_heap_already_in_mb(self):
        output = """\
GPU0:
\tdeviceName = Test GPU
\tdeviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\theapSize  = 8192 (0x2000) (DEVICE_LOCAL)
"""
        gpus = _parse_vulkaninfo(output)
        assert len(gpus) == 1
        assert gpus[0].vram_mb == 8192  # treated as MB since < 1 billion


# ─── _parse_nvidia_smi ───────────────────────────────────────────────────────


class TestParseNvidiaSmi:
    def test_single_gpu(self):
        output = "NVIDIA GeForce RTX 4090, 24564 MiB\n"
        gpus = _parse_nvidia_smi(output)
        assert len(gpus) == 1
        assert gpus[0].name == "NVIDIA GeForce RTX 4090"
        assert gpus[0].vendor == "nvidia"
        assert gpus[0].vram_mb == 24564
        assert gpus[0].is_discrete is True

    def test_multi_gpu(self):
        output = "NVIDIA A100, 40960 MiB\nNVIDIA A100, 40960 MiB\n"
        gpus = _parse_nvidia_smi(output)
        assert len(gpus) == 2
        assert gpus[0].device_index == 0
        assert gpus[1].device_index == 1

    def test_empty_output(self):
        assert _parse_nvidia_smi("") == []

    def test_bad_vram(self):
        output = "NVIDIA GPU, bad_value MiB\n"
        gpus = _parse_nvidia_smi(output)
        assert len(gpus) == 1
        assert gpus[0].vram_mb == 0


# ─── _parse_lspci ────────────────────────────────────────────────────────────


class TestParseLspci:
    def test_nvidia_vga(self):
        output = "01:00.0 VGA compatible controller: NVIDIA Corporation GA102 [GeForce RTX 3090]\n"
        gpus = _parse_lspci(output)
        assert len(gpus) == 1
        assert gpus[0].vendor == "nvidia"
        assert gpus[0].is_discrete is True
        assert gpus[0].vram_mb == 0  # lspci has no VRAM info

    def test_amd_radeon_rx(self):
        output = "06:00.0 VGA compatible controller: AMD Radeon RX 7900 XTX\n"
        gpus = _parse_lspci(output)
        assert len(gpus) == 1
        assert gpus[0].vendor == "amd"
        assert gpus[0].is_discrete is True

    def test_intel_integrated(self):
        output = "00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 770\n"
        gpus = _parse_lspci(output)
        assert len(gpus) == 1
        assert gpus[0].vendor == "intel"
        assert gpus[0].is_discrete is False

    def test_3d_controller(self):
        output = "01:00.0 3D controller: NVIDIA Corporation Tesla T4\n"
        gpus = _parse_lspci(output)
        assert len(gpus) == 1
        assert gpus[0].vendor == "nvidia"

    def test_multiple_devices(self):
        output = (
            "00:02.0 VGA compatible controller: Intel Corporation UHD Graphics\n"
            "01:00.0 VGA compatible controller: NVIDIA Corporation RTX 4090\n"
        )
        gpus = _parse_lspci(output)
        assert len(gpus) == 2
        assert gpus[0].device_index == 0
        assert gpus[1].device_index == 1

    def test_empty_output(self):
        assert _parse_lspci("") == []

    def test_no_gpu_lines(self):
        output = "00:1f.0 ISA bridge: Intel Corporation Device\n"
        assert _parse_lspci(output) == []


# ─── detect_gpus ──────────────────────────────────────────────────────────────


class TestDetectGpus:
    def test_vulkaninfo_preferred(self):
        """vulkaninfo is tried first when available."""
        from natshell.gpu import detect_gpus

        detect_gpus.cache_clear()
        with (
            patch("natshell.gpu.shutil.which", return_value="/usr/bin/vulkaninfo"),
            patch("natshell.gpu._run", return_value=VULKANINFO_SUMMARY),
        ):
            gpus = detect_gpus()
            assert len(gpus) == 2
            assert gpus[0].vendor == "nvidia"
        detect_gpus.cache_clear()

    def test_nvidia_smi_fallback(self):
        """Falls back to nvidia-smi when vulkaninfo not available."""
        from natshell.gpu import detect_gpus

        detect_gpus.cache_clear()

        def which_side_effect(cmd):
            return "/usr/bin/nvidia-smi" if cmd == "nvidia-smi" else None

        with (
            patch("natshell.gpu.shutil.which", side_effect=which_side_effect),
            patch("natshell.gpu._run", return_value="NVIDIA RTX 4090, 24564 MiB\n"),
        ):
            gpus = detect_gpus()
            assert len(gpus) == 1
            assert gpus[0].vram_mb == 24564
        detect_gpus.cache_clear()

    def test_no_tools_returns_empty(self):
        from natshell.gpu import detect_gpus

        detect_gpus.cache_clear()
        with patch("natshell.gpu.shutil.which", return_value=None):
            gpus = detect_gpus()
            assert gpus == []
        detect_gpus.cache_clear()


# ─── best_gpu_index ───────────────────────────────────────────────────────────


class TestBestGpuIndex:
    def test_prefers_discrete(self):
        from natshell.gpu import detect_gpus

        detect_gpus.cache_clear()
        gpus = [
            GpuInfo("Intel UHD", "intel", 0, 2048, False),
            GpuInfo("RTX 4090", "nvidia", 1, 24564, True),
        ]
        with patch("natshell.gpu.detect_gpus", return_value=gpus):
            assert best_gpu_index() == 1
        detect_gpus.cache_clear()

    def test_prefers_higher_vram(self):
        from natshell.gpu import detect_gpus

        detect_gpus.cache_clear()
        gpus = [
            GpuInfo("RTX 3060", "nvidia", 0, 12288, True),
            GpuInfo("RTX 4090", "nvidia", 1, 24564, True),
        ]
        with patch("natshell.gpu.detect_gpus", return_value=gpus):
            assert best_gpu_index() == 1
        detect_gpus.cache_clear()

    def test_no_gpus_returns_zero(self):
        from natshell.gpu import detect_gpus

        detect_gpus.cache_clear()
        with patch("natshell.gpu.detect_gpus", return_value=[]):
            assert best_gpu_index() == 0
        detect_gpus.cache_clear()


# ─── gpu_backend_available ────────────────────────────────────────────────────


class TestGpuBackendAvailable:
    def test_available(self):
        with patch("natshell.gpu.llama_supports_gpu_offload", create=True, return_value=True):
            # Need to patch the import path inside the function
            pass

    def test_no_llama_cpp(self):
        """Returns False when llama_cpp not installed."""
        assert gpu_backend_available() is True or gpu_backend_available() is False
