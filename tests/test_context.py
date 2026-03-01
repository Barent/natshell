"""Tests for system context gathering module."""

from __future__ import annotations

from unittest.mock import patch

from natshell.agent.context import (
    DiskInfo,
    NetInfo,
    SystemContext,
    _parse_linux_df,
    _parse_linux_ip,
    _parse_macos_df,
    _parse_macos_ifconfig,
    gather_system_context,
)

# ─── _parse_linux_df ─────────────────────────────────────────────────────────


class TestParseLinuxDf:
    def test_single_disk(self):
        output = "/   50G   20G   28G  42%\n"
        disks = _parse_linux_df(output)
        assert len(disks) == 1
        assert disks[0].mount == "/"
        assert disks[0].total == "50G"
        assert disks[0].used == "20G"
        assert disks[0].available == "28G"
        assert disks[0].use_percent == "42%"

    def test_multiple_disks(self):
        output = "/   50G   20G   28G  42%\n/home   200G   100G   90G  53%\n"
        disks = _parse_linux_df(output)
        assert len(disks) == 2
        assert disks[1].mount == "/home"

    def test_empty_output(self):
        assert _parse_linux_df("") == []

    def test_short_line_skipped(self):
        output = "foo bar\n/   50G   20G   28G  42%\n"
        disks = _parse_linux_df(output)
        assert len(disks) == 1


# ─── _parse_macos_df ─────────────────────────────────────────────────────────


class TestParseMacosDf:
    def test_standard_mount(self):
        output = "/dev/disk1s1  466Gi  200Gi  250Gi    45%  1000  0  100%  /\n"
        disks = _parse_macos_df(output)
        assert len(disks) == 1
        assert disks[0].mount == "/"
        assert disks[0].total == "466Gi"
        assert disks[0].use_percent == "45%"

    def test_mount_with_spaces(self):
        output = "/dev/disk2s1  100Gi  50Gi  45Gi  53%  500  0  100%  /Volumes/My Drive\n"
        disks = _parse_macos_df(output)
        assert len(disks) == 1
        assert disks[0].mount == "/Volumes/My Drive"

    def test_empty_output(self):
        assert _parse_macos_df("") == []

    def test_no_percent_skipped(self):
        output = "devfs  200Ki  200Ki  0Bi  0  0  0  /dev\n"
        disks = _parse_macos_df(output)
        assert len(disks) == 0


# ─── _parse_linux_ip ─────────────────────────────────────────────────────────


class TestParseLinuxIp:
    def test_standard_interface(self):
        output = "eth0 192.168.1.100/24\n"
        nets = _parse_linux_ip(output)
        assert len(nets) == 1
        assert nets[0].name == "eth0"
        assert nets[0].ip == "192.168.1.100"
        assert nets[0].subnet == "24"

    def test_loopback_filtered(self):
        output = "lo 127.0.0.1/8\neth0 10.0.0.5/24\n"
        nets = _parse_linux_ip(output)
        assert len(nets) == 1
        assert nets[0].name == "eth0"

    def test_multiple_interfaces(self):
        output = "eth0 192.168.1.100/24\nwlan0 192.168.1.101/24\n"
        nets = _parse_linux_ip(output)
        assert len(nets) == 2

    def test_empty_output(self):
        assert _parse_linux_ip("") == []


# ─── _parse_macos_ifconfig ───────────────────────────────────────────────────


class TestParseMacosIfconfig:
    def test_standard_interface(self):
        output = "\tinet 192.168.1.50 netmask 0xffffff00 broadcast 192.168.1.255\n"
        nets = _parse_macos_ifconfig(output)
        assert len(nets) == 1
        assert nets[0].ip == "192.168.1.50"
        assert nets[0].subnet == "0xffffff00"

    def test_loopback_filtered(self):
        output = (
            "\tinet 127.0.0.1 netmask 0xff000000\n"
            "\tinet 192.168.1.50 netmask 0xffffff00 broadcast 192.168.1.255\n"
        )
        nets = _parse_macos_ifconfig(output)
        assert len(nets) == 1
        assert nets[0].ip == "192.168.1.50"

    def test_empty_output(self):
        assert _parse_macos_ifconfig("") == []


# ─── SystemContext.to_prompt_text ─────────────────────────────────────────────


class TestToPromptText:
    def test_basic_fields(self):
        ctx = SystemContext(
            hostname="myhost",
            distro="Ubuntu 24.04",
            kernel="6.5.0",
            arch="x86_64",
            cpu="Intel i9",
            ram_total_gb=32.0,
            ram_available_gb=16.0,
            username="user",
            has_sudo=True,
            shell="/bin/bash",
            package_manager="apt",
            cwd="/home/user",
        )
        text = ctx.to_prompt_text()
        assert "myhost" in text
        assert "Ubuntu 24.04" in text
        assert "32.0GB" in text
        assert "sudo: yes" in text

    def test_disks_included(self):
        ctx = SystemContext(disks=[DiskInfo("/", "50G", "20G", "28G", "42%")])
        text = ctx.to_prompt_text()
        assert "/ 50G (42% used)" in text

    def test_network_included(self):
        ctx = SystemContext(network=[NetInfo("eth0", "192.168.1.100", "24")])
        text = ctx.to_prompt_text()
        assert "eth0 192.168.1.100/24" in text

    def test_tools_present_and_missing(self):
        ctx = SystemContext(installed_tools={"git": True, "docker": False})
        text = ctx.to_prompt_text()
        assert "git✓" in text
        assert "docker✗" in text

    def test_containers(self):
        ctx = SystemContext(containers=["web (nginx:latest)", "db (postgres:16)"])
        text = ctx.to_prompt_text()
        assert "web (nginx:latest)" in text

    def test_services(self):
        ctx = SystemContext(running_services=["sshd", "nginx", "docker"])
        text = ctx.to_prompt_text()
        assert "sshd" in text
        assert "docker" in text

    def test_gateway(self):
        ctx = SystemContext(default_gateway="192.168.1.1")
        text = ctx.to_prompt_text()
        assert "Gateway: 192.168.1.1" in text

    def test_empty_context(self):
        ctx = SystemContext()
        text = ctx.to_prompt_text()
        assert "Host:" in text


# ─── gather_system_context ────────────────────────────────────────────────────


class TestGatherSystemContext:
    async def test_linux_path(self):
        """gather_system_context calls _gather_linux on non-macOS."""
        with patch("natshell.agent.context.is_macos", return_value=False):
            ctx = await gather_system_context()
            assert isinstance(ctx, SystemContext)
            assert ctx.username  # should be populated from env

    async def test_macos_path(self):
        """gather_system_context calls _gather_macos on macOS."""

        async def mock_gather_macos(ctx):
            ctx.hostname = "mock-mac"
            ctx.distro = "macOS Sonoma"

        with (
            patch("natshell.agent.context.is_macos", return_value=True),
            patch("natshell.agent.context._gather_macos", side_effect=mock_gather_macos),
        ):
            ctx = await gather_system_context()
            assert ctx.hostname == "mock-mac"
            assert ctx.distro == "macOS Sonoma"
