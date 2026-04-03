"""GPU and NPU hardware detection and best-device selection.

Follows the same cached-detection pattern as ``platform.py``.
Tries ``vulkaninfo``, ``nvidia-smi``, ``lspci`` (Linux), and
``WMI`` (Windows) in order for GPUs.  Detects Qualcomm NPUs on Windows.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class GpuInfo:
    """Describes a single GPU visible to the system."""

    name: str
    vendor: str  # "nvidia", "amd", "intel", "qualcomm", "unknown"
    device_index: int
    vram_mb: int
    is_discrete: bool


@dataclass
class NpuInfo:
    """Describes a detected NPU (Neural Processing Unit)."""

    name: str
    vendor: str  # "qualcomm", "intel", "unknown"
    sdk_available: bool  # True if the vendor SDK (e.g. QNN) is found


def _run(cmd: list[str], timeout: int = 5) -> str | None:
    """Run *cmd* and return stdout, or ``None`` on any failure."""
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode == 0:
            return r.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _classify_vendor(name: str) -> str:
    low = name.lower()
    if "nvidia" in low or "geforce" in low or "rtx" in low or "gtx" in low or "quadro" in low:
        return "nvidia"
    if "amd" in low or "radeon" in low:
        return "amd"
    if "intel" in low:
        return "intel"
    if "qualcomm" in low or "adreno" in low:
        return "qualcomm"
    return "unknown"


def _parse_vulkaninfo(output: str) -> list[GpuInfo]:
    """Parse ``vulkaninfo --summary`` output into GpuInfo entries."""
    gpus: list[GpuInfo] = []
    # Split by GPU blocks — each starts with a line like "GPU0:" or "GPU 0:"
    blocks = re.split(r"(?m)^GPU\s*(\d+)\s*:", output)
    # blocks[0] is header, then alternating (index, body)
    if len(blocks) < 3:
        return gpus

    for i in range(1, len(blocks), 2):
        idx = int(blocks[i])
        body = blocks[i + 1]

        name = ""
        device_type = ""
        vram_mb = 0
        _pending_heap_size: int = 0

        for line in body.splitlines():
            line_s = line.strip()
            if "deviceName" in line_s:
                name = line_s.split("=", 1)[-1].strip()
            elif "deviceType" in line_s:
                device_type = line_s.split("=", 1)[-1].strip().lower()
            elif "heapSize" in line_s:
                m = re.search(r"heapSize\s*=\s*(\d+)", line_s)
                if m:
                    _pending_heap_size = int(m.group(1))
                    # Handle single-line format: "heapSize = N (...) (DEVICE_LOCAL)"
                    if "DEVICE_LOCAL" in line_s:
                        heap_bytes = _pending_heap_size
                        _pending_heap_size = 0
                        if heap_bytes > 1_000_000_000:
                            vram_mb = heap_bytes // (1024 * 1024)
                        else:
                            # Already in MB in some builds
                            vram_mb = heap_bytes
            elif "DEVICE_LOCAL" in line_s and _pending_heap_size > 0:
                # Two-line format: heapSize on previous line, DEVICE_LOCAL on this line
                # e.g. "heapFlags = DEVICE_LOCAL_BIT" (modern vulkan-tools 1.3.x)
                heap_bytes = _pending_heap_size
                _pending_heap_size = 0
                if heap_bytes > 1_000_000_000:
                    vram_mb = heap_bytes // (1024 * 1024)
                else:
                    vram_mb = heap_bytes

        is_discrete = "discrete" in device_type
        vendor = _classify_vendor(name)
        if name:
            gpus.append(
                GpuInfo(
                    name=name,
                    vendor=vendor,
                    device_index=idx,
                    vram_mb=vram_mb,
                    is_discrete=is_discrete,
                )
            )

    return gpus


def _parse_nvidia_smi(output: str) -> list[GpuInfo]:
    """Parse ``nvidia-smi --query-gpu`` CSV output."""
    gpus: list[GpuInfo] = []
    for idx, line in enumerate(output.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        vram_mb = 0
        mem_str = parts[1].replace("MiB", "").strip()
        try:
            vram_mb = int(mem_str)
        except ValueError:
            pass
        gpus.append(
            GpuInfo(
                name=name,
                vendor="nvidia",
                device_index=idx,
                vram_mb=vram_mb,
                is_discrete=True,
            )
        )
    return gpus


def _parse_lspci(output: str) -> list[GpuInfo]:
    """Parse ``lspci`` output for VGA/3D controllers."""
    gpus: list[GpuInfo] = []
    idx = 0
    for line in output.splitlines():
        if "VGA" in line or "3D" in line or "Display" in line:
            # e.g. "01:00.0 VGA compatible controller: NVIDIA Corporation ..."
            match = re.search(r":\s+(.+)$", line)
            name = match.group(1).strip() if match else line
            vendor = _classify_vendor(name)
            # Heuristic: NVIDIA and AMD discrete GPUs typically have PCI bus > 00
            is_discrete = vendor in ("nvidia",) or ("Radeon RX" in name)
            gpus.append(
                GpuInfo(
                    name=name,
                    vendor=vendor,
                    device_index=idx,
                    vram_mb=0,
                    is_discrete=is_discrete,
                )
            )
            idx += 1
    return gpus


def _parse_wmi_gpu(output: str) -> list[GpuInfo]:
    """Parse PowerShell ``Get-CimInstance Win32_VideoController`` output.

    Expected format (tab-separated):
        Name<TAB>AdapterRAM
        NVIDIA GeForce RTX 4060<TAB>8589934592
        Qualcomm Adreno GPU<TAB>0
    """
    gpus: list[GpuInfo] = []
    lines = output.strip().splitlines()
    # Skip header line(s)
    for idx, line in enumerate(lines):
        if idx == 0 and ("Name" in line or "----" in line):
            continue
        if line.startswith("---"):
            continue
        parts = line.split("\t")
        name = parts[0].strip() if parts else ""
        if not name:
            continue
        vram_mb = 0
        if len(parts) >= 2:
            try:
                vram_bytes = int(parts[1].strip())
                if vram_bytes > 0:
                    vram_mb = vram_bytes // (1024 * 1024)
            except (ValueError, IndexError):
                pass
        vendor = _classify_vendor(name)
        # On Windows: discrete GPUs typically have non-zero VRAM, or are
        # NVIDIA/AMD.  Integrated (Intel/Qualcomm Adreno) report 0 or low VRAM.
        is_discrete = vendor in ("nvidia",) or (
            vendor == "amd" and vram_mb > 512
        )
        gpus.append(
            GpuInfo(
                name=name,
                vendor=vendor,
                device_index=idx,
                vram_mb=vram_mb,
                is_discrete=is_discrete,
            )
        )
    return gpus


@lru_cache(maxsize=1)
def detect_gpus() -> list[GpuInfo]:
    """Detect GPU hardware. Tries vulkaninfo, nvidia-smi, lspci/WMI in order."""
    from natshell.platform import is_windows

    # 1. vulkaninfo — cross-vendor, gives device type (discrete/integrated)
    if shutil.which("vulkaninfo"):
        out = _run(["vulkaninfo", "--summary"])
        if out:
            gpus = _parse_vulkaninfo(out)
            if gpus:
                logger.debug("GPU detection via vulkaninfo: %d device(s)", len(gpus))
                return gpus

    # 2. nvidia-smi — NVIDIA-specific, gives VRAM
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        if out:
            gpus = _parse_nvidia_smi(out)
            if gpus:
                logger.debug("GPU detection via nvidia-smi: %d device(s)", len(gpus))
                return gpus

    # 3. Platform-specific fallback
    if is_windows():
        # WMI via PowerShell
        out = _run(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_VideoController"
                " | Select-Object Name,AdapterRAM"
                " | Format-Table -HideTableHeaders"
                " | Out-String",
            ],
            timeout=10,
        )
        if out:
            gpus = _parse_wmi_gpu(out)
            if gpus:
                logger.debug("GPU detection via WMI: %d device(s)", len(gpus))
                return gpus
    else:
        # lspci — Linux, no VRAM info
        if shutil.which("lspci"):
            out = _run(["lspci"])
            if out:
                gpus = _parse_lspci(out)
                if gpus:
                    logger.debug("GPU detection via lspci: %d device(s)", len(gpus))
                    return gpus

    logger.debug("No GPUs detected")
    return []


def best_gpu_index() -> int:
    """Return device index of the best GPU for inference.

    Prefers discrete over integrated, then highest VRAM.
    Returns 0 if no GPUs detected.
    """
    gpus = detect_gpus()
    if not gpus:
        return 0

    # Sort: discrete first, then by VRAM descending
    ranked = sorted(gpus, key=lambda g: (g.is_discrete, g.vram_mb), reverse=True)
    return ranked[0].device_index


def gpu_backend_available() -> bool:
    """Check if llama-cpp-python was built with GPU offload support."""
    try:
        from llama_cpp import llama_supports_gpu_offload

        return llama_supports_gpu_offload()
    except ImportError:
        return False


# ── NPU detection ───────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def detect_npu() -> NpuInfo | None:
    """Detect a Neural Processing Unit (NPU).

    Currently detects:
    - Qualcomm Hexagon NPU (Snapdragon X Elite/Plus) via WMI and QNN SDK
    - Intel NPU (Core Ultra series) via WMI and OpenVINO SDK

    Returns NpuInfo if found, None otherwise.
    """
    from natshell.platform import is_windows

    if not is_windows():
        return None

    # Check for vendor SDKs
    qnn_sdk = bool(os.environ.get("QNN_SDK_ROOT"))
    openvino_sdk = bool(os.environ.get("INTEL_OPENVINO_DIR"))

    # Query Windows for NPU devices via PowerShell
    out = _run(
        [
            "powershell", "-NoProfile", "-Command",
            "Get-PnpDevice -Class 'System' -Status 'OK'"
            " -ErrorAction SilentlyContinue"
            " | Where-Object {"
            " $_.FriendlyName -match 'NPU|Hexagon|Neural' }"
            " | Select-Object -First 1"
            " -ExpandProperty FriendlyName",
        ],
        timeout=10,
    )

    if out and out.strip():
        name = out.strip().splitlines()[0].strip()
        low = name.lower()
        if "qualcomm" in low or "hexagon" in low:
            vendor = "qualcomm"
            sdk_found = qnn_sdk
        elif "intel" in low:
            vendor = "intel"
            sdk_found = openvino_sdk
        else:
            vendor = "unknown"
            sdk_found = qnn_sdk or openvino_sdk
        return NpuInfo(
            name=name,
            vendor=vendor,
            sdk_available=sdk_found,
        )

    # Fallback: check SDK presence even without WMI detection
    if qnn_sdk:
        return NpuInfo(
            name="Qualcomm NPU (detected via QNN SDK)",
            vendor="qualcomm",
            sdk_available=True,
        )
    if openvino_sdk:
        return NpuInfo(
            name="Intel NPU (detected via OpenVINO SDK)",
            vendor="intel",
            sdk_available=True,
        )

    return None
