"""Gather system context for injection into the LLM system prompt."""

from __future__ import annotations

import asyncio
import os
import subprocess
from dataclasses import dataclass, field

from natshell.platform import is_macos


@dataclass
class DiskInfo:
    mount: str
    total: str
    used: str
    available: str
    use_percent: str


@dataclass
class NetInfo:
    name: str
    ip: str
    subnet: str


@dataclass
class SystemContext:
    hostname: str = ""
    distro: str = ""
    kernel: str = ""
    arch: str = ""
    cpu: str = ""
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    username: str = ""
    is_root: bool = False
    has_sudo: bool = False
    shell: str = ""
    package_manager: str = ""
    cwd: str = ""
    disks: list[DiskInfo] = field(default_factory=list)
    network: list[NetInfo] = field(default_factory=list)
    default_gateway: str = ""
    installed_tools: dict[str, bool] = field(default_factory=dict)
    running_services: list[str] = field(default_factory=list)
    containers: list[str] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Format system context as a compact text block for the system prompt."""
        lines = []

        # Host info
        lines.append(f"Host: {self.hostname} | {self.distro} | {self.kernel} | {self.arch}")
        lines.append(
            f"CPU: {self.cpu} | RAM: {self.ram_total_gb:.1f}GB total, "
            f"{self.ram_available_gb:.1f}GB available"
        )
        sudo_str = "yes" if self.has_sudo else "no"
        lines.append(
            f"User: {self.username} (sudo: {sudo_str}) | Shell: {self.shell} | "
            f"Pkg: {self.package_manager}"
        )
        lines.append(f"CWD: {self.cwd}")

        # Disks
        if self.disks:
            disk_parts = [f"{d.mount} {d.total} ({d.use_percent} used)" for d in self.disks]
            lines.append(f"Disks: {', '.join(disk_parts)}")

        # Network
        if self.network:
            net_parts = [f"{n.name} {n.ip}/{n.subnet}" for n in self.network]
            lines.append(f"Network: {' | '.join(net_parts)}")
        if self.default_gateway:
            lines.append(f"Gateway: {self.default_gateway}")

        # Tools
        if self.installed_tools:
            present = [k + "✓" for k, v in self.installed_tools.items() if v]
            missing = [k + "✗" for k, v in self.installed_tools.items() if not v]
            lines.append(f"Tools: {' '.join(present + missing)}")

        # Containers
        if self.containers:
            lines.append(f"Containers: {', '.join(self.containers[:10])}")

        # Services
        if self.running_services:
            lines.append(f"Services: {', '.join(self.running_services[:15])}")

        return "\n".join(lines)


async def _run(cmd: str) -> str:
    """Run a shell command and return stripped stdout, or empty string on failure."""
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _parse_linux_df(df_output: str) -> list[DiskInfo]:
    """Parse Linux ``df -h --output=...`` output into DiskInfo entries."""
    disks = []
    for line in df_output.splitlines():
        parts = line.split()
        if len(parts) >= 5:
            disks.append(
                DiskInfo(
                    mount=parts[0],
                    total=parts[1],
                    used=parts[2],
                    available=parts[3],
                    use_percent=parts[4],
                )
            )
    return disks


def _parse_macos_df(df_output: str) -> list[DiskInfo]:
    """Parse macOS ``df -h`` output (no ``--output`` flag) into DiskInfo entries."""
    disks = []
    for line in df_output.splitlines():
        parts = line.split()
        # macOS df -h: Filesystem Size Used Avail Capacity ... Mounted on
        if len(parts) >= 9 and parts[4].endswith("%"):
            mount = " ".join(parts[8:])  # Mounted on may contain spaces
            disks.append(
                DiskInfo(
                    mount=mount,
                    total=parts[1],
                    used=parts[2],
                    available=parts[3],
                    use_percent=parts[4],
                )
            )
    return disks


def _parse_linux_ip(ip_output: str) -> list[NetInfo]:
    """Parse Linux ``ip -4 -o addr show`` output into NetInfo entries."""
    interfaces = []
    for line in ip_output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and "/" in parts[1]:
            ip, prefix = parts[1].split("/")
            if not ip.startswith("127."):
                interfaces.append(NetInfo(name=parts[0], ip=ip, subnet=prefix))
    return interfaces


def _parse_macos_ifconfig(ifconfig_output: str) -> list[NetInfo]:
    """Parse macOS ``ifconfig | grep 'inet '`` output into NetInfo entries."""
    interfaces = []
    for line in ifconfig_output.splitlines():
        line = line.strip()
        # Format: "inet 192.168.1.5 netmask 0xffffff00 broadcast ..."
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "inet" and not parts[1].startswith("127."):
            ip = parts[1]
            netmask = parts[3] if parts[2] == "netmask" else ""
            interfaces.append(NetInfo(name="", ip=ip, subnet=netmask))
    return interfaces


async def _gather_linux(ctx: SystemContext) -> None:
    """Gather system context using Linux commands."""
    (
        ctx.hostname,
        distro_line,
        ctx.kernel,
        ctx.arch,
        cpu_line,
        mem_line,
        sudo_check,
        gateway_line,
        df_output,
        ip_output,
        services_output,
        docker_output,
    ) = await asyncio.gather(
        _run("hostname"),
        _run("grep PRETTY_NAME /etc/os-release | cut -d'\"' -f2"),
        _run("uname -r"),
        _run("uname -m"),
        _run("lscpu | grep 'Model name' | sed 's/Model name:\\s*//'"),
        _run("free -b | grep Mem"),
        _run("sudo -n true 2>/dev/null && echo yes || echo no"),
        _run("ip -4 route show default | awk '{print $3}'"),
        _run(
            "df -h --output=target,size,used,avail,pcent"
            " -x tmpfs -x devtmpfs -x squashfs 2>/dev/null | tail -n +2"
        ),
        _run("ip -4 -o addr show | awk '{print $2, $4}'"),
        _run(
            "systemctl list-units --type=service --state=running"
            " --no-pager -q 2>/dev/null"
            " | awk '{print $1}' | sed 's/.service$//' | head -20"
        ),
        _run("docker ps --format '{{.Names}} ({{.Image}})' 2>/dev/null | head -10"),
    )

    ctx.distro = distro_line or "Unknown"
    ctx.cpu = cpu_line or "Unknown"
    ctx.has_sudo = sudo_check.strip() == "yes"
    ctx.default_gateway = gateway_line

    # Parse memory (free -b output: Mem: total used free shared buff/cache available)
    if mem_line:
        parts = mem_line.split()
        if len(parts) >= 7:
            try:
                ctx.ram_total_gb = int(parts[1]) / (1024**3)
                ctx.ram_available_gb = int(parts[6]) / (1024**3)
            except (ValueError, IndexError):
                pass

    ctx.disks = _parse_linux_df(df_output)
    ctx.network = _parse_linux_ip(ip_output)

    if services_output:
        ctx.running_services = [s.strip() for s in services_output.splitlines() if s.strip()]
    if docker_output:
        ctx.containers = [c.strip() for c in docker_output.splitlines() if c.strip()]


async def _gather_macos(ctx: SystemContext) -> None:
    """Gather system context using macOS commands."""
    (
        ctx.hostname,
        distro_line,
        ctx.kernel,
        ctx.arch,
        cpu_line,
        ram_total_raw,
        ram_avail_raw,
        sudo_check,
        gateway_line,
        df_output,
        ifconfig_output,
        services_output,
        docker_output,
    ) = await asyncio.gather(
        _run("hostname"),
        _run("sw_vers -productName -productVersion"),
        _run("uname -r"),
        _run("uname -m"),
        _run("sysctl -n machdep.cpu.brand_string"),
        _run("sysctl -n hw.memsize"),
        _run("vm_stat | awk '/Pages free|Pages inactive/ {sum += $NF} END {print sum * 4096}'"),
        _run("sudo -n true 2>/dev/null && echo yes || echo no"),
        _run("route -n get default 2>/dev/null | awk '/gateway/{print $2}'"),
        _run("df -h | grep -vE '^(devfs|map |Filesystem)' | tail -n +1"),
        _run("ifconfig | grep 'inet '"),
        _run("launchctl list 2>/dev/null | awk 'NR>1 {print $3}' | head -20"),
        _run("docker ps --format '{{.Names}} ({{.Image}})' 2>/dev/null | head -10"),
    )

    ctx.distro = distro_line.replace("\n", " ") if distro_line else "macOS"
    ctx.cpu = cpu_line or "Unknown"
    ctx.has_sudo = sudo_check.strip() == "yes"
    ctx.default_gateway = gateway_line

    # Parse memory
    if ram_total_raw:
        try:
            ctx.ram_total_gb = int(ram_total_raw) / (1024**3)
        except ValueError:
            pass
    if ram_avail_raw:
        try:
            ctx.ram_available_gb = int(ram_avail_raw) / (1024**3)
        except ValueError:
            pass

    ctx.disks = _parse_macos_df(df_output)
    ctx.network = _parse_macos_ifconfig(ifconfig_output)

    if services_output:
        ctx.running_services = [s.strip() for s in services_output.splitlines() if s.strip()]
    if docker_output:
        ctx.containers = [c.strip() for c in docker_output.splitlines() if c.strip()]


async def gather_system_context() -> SystemContext:
    """Gather system information. Non-blocking, tolerates failures."""
    ctx = SystemContext()

    if is_macos():
        await _gather_macos(ctx)
    else:
        await _gather_linux(ctx)

    # Common fields
    ctx.username = os.environ.get("USER", "unknown")
    ctx.is_root = os.geteuid() == 0
    ctx.shell = os.environ.get("SHELL", "/bin/sh")
    ctx.cwd = os.getcwd()

    # Detect package manager
    pm_list = ["brew", "apt", "dnf", "yum", "pacman", "zypper", "apk", "emerge"]
    if not is_macos():
        # On Linux, don't prioritize brew
        pm_list = ["apt", "dnf", "yum", "pacman", "zypper", "apk", "emerge", "brew"]
    for pm in pm_list:
        check = await _run(f"which {pm} 2>/dev/null")
        if check:
            ctx.package_manager = pm
            break

    # Check common tools
    tools_to_check = [
        "docker",
        "git",
        "nmap",
        "curl",
        "wget",
        "ssh",
        "python3",
        "node",
        "go",
        "rsync",
        "tmux",
        "vim",
        "htop",
        "jq",
        # Languages & compilers
        "rustc",
        "gcc",
        "g++",
        "clang",
        "javac",
        "ruby",
        "php",
        "perl",
        # Package managers & build tools
        "pip",
        "npm",
        "yarn",
        "cargo",
        "composer",
        "gem",
        "make",
        "cmake",
    ]
    tool_checks = await asyncio.gather(
        *[_run(f"which {tool} 2>/dev/null") for tool in tools_to_check]
    )
    ctx.installed_tools = {tool: bool(result) for tool, result in zip(tools_to_check, tool_checks)}

    return ctx
