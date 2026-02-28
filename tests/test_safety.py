"""Tests for the safety classifier."""

from __future__ import annotations

from natshell.config import SafetyConfig
from natshell.safety.classifier import Risk, SafetyClassifier


def _make_classifier(
    mode: str = "confirm",
    always_confirm: list[str] | None = None,
    blocked: list[str] | None = None,
) -> SafetyClassifier:
    """Create a classifier with the default config patterns."""
    config = SafetyConfig(
        mode=mode,
        always_confirm=always_confirm
        or [
            r"^rm\s",
            r"^sudo\s",
            r"^dd\s",
            r"^mkfs",
            r"^shutdown",
            r"^reboot",
            r"^systemctl\s+(stop|disable|mask|restart|enable|start)",
            r"^chmod\s+[0-7]*7",
            r"^chown",
            r"\|\s*tee\s",
            r">\s*/etc/",
            r"^kill",
            r"^wipefs",
            r"^fdisk",
            r"^parted",
            r"^apt\s+(install|remove|purge|autoremove)",
            r"^dnf\s+(install|remove|erase)",
            r"^pacman\s+-[SRU]",
            r"^pip\s+install",
            r"^docker\s+(rm|rmi|stop|kill|system\s+prune)",
            r"^iptables",
            r"^ufw",
            r"^crontab",
        ],
        blocked=blocked
        or [
            r":(){ :|:& };:",
            r"^rm\s+-[rR]f\s+/\s*$",
            r"^rm\s+-[rR]f\s+/\*",
            r"^mv\s+/\s",
            r"^dd\s+.*of=/dev/[sh]d[a-z]\s*$",
            r"^mkfs.*\s/dev/[sh]d[a-z][0-9]?\s*$",
            r"> /dev/[sh]d[a-z]",
        ],
    )
    return SafetyClassifier(config)


# ─── Blocked commands ────────────────────────────────────────────────────────


class TestBlockedCommands:
    def test_fork_bomb(self):
        c = _make_classifier()
        assert c.classify_command(":(){ :|:& };:") == Risk.BLOCKED

    def test_rm_rf_root(self):
        c = _make_classifier()
        assert c.classify_command("rm -rf /") == Risk.BLOCKED
        assert c.classify_command("rm -Rf /") == Risk.BLOCKED

    def test_rm_rf_root_star(self):
        c = _make_classifier()
        assert c.classify_command("rm -rf /*") == Risk.BLOCKED

    def test_mv_root(self):
        c = _make_classifier()
        assert c.classify_command("mv / /tmp") == Risk.BLOCKED

    def test_dd_to_disk(self):
        c = _make_classifier()
        assert c.classify_command("dd if=/dev/zero of=/dev/sda") == Risk.BLOCKED

    def test_mkfs_disk(self):
        c = _make_classifier()
        assert c.classify_command("mkfs.ext4 /dev/sda1") == Risk.BLOCKED

    def test_redirect_to_disk(self):
        c = _make_classifier()
        assert c.classify_command("> /dev/sda") == Risk.BLOCKED


# ─── Confirm commands ────────────────────────────────────────────────────────


class TestConfirmCommands:
    def test_rm(self):
        c = _make_classifier()
        assert c.classify_command("rm file.txt") == Risk.CONFIRM

    def test_sudo(self):
        c = _make_classifier()
        assert c.classify_command("sudo apt update") == Risk.CONFIRM

    def test_dd(self):
        c = _make_classifier()
        assert c.classify_command("dd if=a of=b") == Risk.CONFIRM

    def test_mkfs(self):
        c = _make_classifier()
        # mkfs without /dev/sd* is confirm, not blocked
        assert c.classify_command("mkfs.ext4 /dev/loop0") == Risk.CONFIRM

    def test_systemctl_stop(self):
        c = _make_classifier()
        assert c.classify_command("systemctl stop nginx") == Risk.CONFIRM

    def test_systemctl_restart(self):
        c = _make_classifier()
        assert c.classify_command("systemctl restart sshd") == Risk.CONFIRM

    def test_kill(self):
        c = _make_classifier()
        assert c.classify_command("kill -9 1234") == Risk.CONFIRM

    def test_apt_install(self):
        c = _make_classifier()
        assert c.classify_command("apt install nginx") == Risk.CONFIRM

    def test_pip_install(self):
        c = _make_classifier()
        assert c.classify_command("pip install requests") == Risk.CONFIRM

    def test_docker_rm(self):
        c = _make_classifier()
        assert c.classify_command("docker rm mycontainer") == Risk.CONFIRM

    def test_iptables(self):
        c = _make_classifier()
        assert c.classify_command("iptables -A INPUT -j DROP") == Risk.CONFIRM

    def test_redirect_to_etc(self):
        c = _make_classifier()
        assert c.classify_command("echo x > /etc/hostname") == Risk.CONFIRM

    def test_tee_pipe(self):
        c = _make_classifier()
        assert c.classify_command("echo x | tee /tmp/out") == Risk.CONFIRM

    def test_crontab(self):
        c = _make_classifier()
        assert c.classify_command("crontab -e") == Risk.CONFIRM

    def test_sudo_heuristic(self):
        c = _make_classifier()
        assert c.classify_command("sudo ls") == Risk.CONFIRM

    def test_redirect_to_system_path_heuristic(self):
        c = _make_classifier()
        assert c.classify_command("echo x > /boot/grub/grub.cfg") == Risk.CONFIRM


# ─── Safe commands ───────────────────────────────────────────────────────────


class TestSafeCommands:
    def test_ls(self):
        c = _make_classifier()
        assert c.classify_command("ls -la") == Risk.SAFE

    def test_cat(self):
        c = _make_classifier()
        assert c.classify_command("cat /etc/hostname") == Risk.SAFE

    def test_df(self):
        c = _make_classifier()
        assert c.classify_command("df -h") == Risk.SAFE

    def test_grep(self):
        c = _make_classifier()
        assert c.classify_command("grep -r TODO .") == Risk.SAFE

    def test_echo(self):
        c = _make_classifier()
        assert c.classify_command("echo hello") == Risk.SAFE

    def test_uname(self):
        c = _make_classifier()
        assert c.classify_command("uname -a") == Risk.SAFE

    def test_ps(self):
        c = _make_classifier()
        assert c.classify_command("ps aux") == Risk.SAFE

    def test_systemctl_status(self):
        c = _make_classifier()
        assert c.classify_command("systemctl status nginx") == Risk.SAFE

    def test_ip_addr(self):
        c = _make_classifier()
        assert c.classify_command("ip addr show") == Risk.SAFE

    def test_apt_list(self):
        c = _make_classifier()
        assert c.classify_command("apt list --installed") == Risk.SAFE

    def test_docker_ps(self):
        c = _make_classifier()
        assert c.classify_command("docker ps") == Risk.SAFE


# ─── Tool call classification ────────────────────────────────────────────────


class TestToolCallClassification:
    def test_write_file_always_confirm(self):
        c = _make_classifier()
        assert (
            c.classify_tool_call("write_file", {"path": "/tmp/x", "content": "hi"}) == Risk.CONFIRM
        )

    def test_edit_file_always_confirm(self):
        c = _make_classifier()
        assert (
            c.classify_tool_call("edit_file", {"path": "/tmp/x", "old_text": "a", "new_text": "b"})
            == Risk.CONFIRM
        )

    def test_edit_file_sensitive_path(self):
        c = _make_classifier()
        assert (
            c.classify_tool_call(
                "edit_file", {"path": "/home/user/.ssh/config", "old_text": "a", "new_text": "b"}
            )
            == Risk.CONFIRM
        )

    def test_run_code_always_confirm(self):
        c = _make_classifier()
        assert (
            c.classify_tool_call("run_code", {"language": "python", "code": "print(1)"})
            == Risk.CONFIRM
        )

    def test_read_file_always_safe(self):
        c = _make_classifier()
        assert c.classify_tool_call("read_file", {"path": "/etc/passwd"}) == Risk.SAFE

    def test_list_directory_always_safe(self):
        c = _make_classifier()
        assert c.classify_tool_call("list_directory", {"path": "/"}) == Risk.SAFE

    def test_search_files_always_safe(self):
        c = _make_classifier()
        assert c.classify_tool_call("search_files", {"pattern": "TODO"}) == Risk.SAFE

    def test_execute_shell_delegates(self):
        c = _make_classifier()
        assert c.classify_tool_call("execute_shell", {"command": "rm foo"}) == Risk.CONFIRM
        assert c.classify_tool_call("execute_shell", {"command": "ls"}) == Risk.SAFE


# ─── YOLO mode ───────────────────────────────────────────────────────────────


class TestYoloMode:
    def test_yolo_downgrades_confirm_to_safe(self):
        c = _make_classifier(mode="yolo")
        assert c.classify_tool_call("execute_shell", {"command": "rm foo"}) == Risk.SAFE
        assert c.classify_tool_call("execute_shell", {"command": "sudo apt install x"}) == Risk.SAFE

    def test_yolo_does_not_downgrade_blocked(self):
        c = _make_classifier(mode="yolo")
        assert c.classify_tool_call("execute_shell", {"command": "rm -rf /"}) == Risk.BLOCKED

    def test_yolo_downgrades_edit_file(self):
        c = _make_classifier(mode="yolo")
        assert (
            c.classify_tool_call("edit_file", {"path": "/tmp/x", "old_text": "a", "new_text": "b"})
            == Risk.SAFE
        )

    def test_yolo_downgrades_run_code(self):
        c = _make_classifier(mode="yolo")
        assert (
            c.classify_tool_call("run_code", {"language": "python", "code": "print(1)"})
            == Risk.SAFE
        )

    def test_yolo_edit_file_sensitive_path_still_confirm(self):
        """Even in yolo mode, sensitive paths on edit_file stay CONFIRM."""
        c = _make_classifier(mode="yolo")
        assert (
            c.classify_tool_call(
                "edit_file", {"path": "/home/user/.env", "old_text": "a", "new_text": "b"}
            )
            == Risk.CONFIRM
        )
