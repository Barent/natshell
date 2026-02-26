"""GPU hardware detection and best-device selection.

Follows the same cached-detection pattern as ``platform.py``.
Tries ``vulkaninfo``, ``nvidia-smi``, and ``lspci`` in order.
"""

from __future__ import annotations

import logging
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
    vendor: str  # "nvidia", "amd", "intel", "unknown"
    device_index: int
    vram_mb: int
    is_discrete: bool


def _run(cmd: list[str], timeout: int = 5) -> str | None:
    """Run *cmd* and return stdout, or ``None`` on any failure."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
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

        for line in body.splitlines():
            line_s = line.strip()
            if "deviceName" in line_s:
                name = line_s.split("=", 1)[-1].strip()
            elif "deviceType" in line_s:
                device_type = line_s.split("=", 1)[-1].strip().lower()
            elif "DEVICE_LOCAL" in line_s and "heapSize" in line_s:
                # e.g. "heapSize  = 8192 (0x200000000) (DEVICE_LOCAL)"
                m = re.search(r"heapSize\s*=\s*(\d+)", line_s)
                if m:
                    heap_bytes = int(m.group(1))
                    # vulkaninfo prints heap size in bytes
                    if heap_bytes > 1_000_000_000:
                        vram_mb = heap_bytes // (1024 * 1024)
                    else:
                        # Already in MB in some builds
                        vram_mb = heap_bytes

        is_discrete = "discrete" in device_type
        vendor = _classify_vendor(name)
        if name:
            gpus.append(GpuInfo(
                name=name,
                vendor=vendor,
                device_index=idx,
                vram_mb=vram_mb,
                is_discrete=is_discrete,
            ))

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
        gpus.append(GpuInfo(
            name=name,
            vendor="nvidia",
            device_index=idx,
            vram_mb=vram_mb,
            is_discrete=True,
        ))
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
            gpus.append(GpuInfo(
                name=name,
                vendor=vendor,
                device_index=idx,
                vram_mb=0,
                is_discrete=is_discrete,
            ))
            idx += 1
    return gpus


@lru_cache(maxsize=1)
def detect_gpus() -> list[GpuInfo]:
    """Detect GPU hardware. Tries vulkaninfo, nvidia-smi, lspci in order."""

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

    # 3. lspci — last resort, no VRAM info
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
