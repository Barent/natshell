---
name: system-admin
description: Linux/macOS administration: services, processes, package managers, and log triage.
---

# System administration skill

## When to use
- User asks to install, update, or remove packages
- User asks to start, stop, or restart services
- User asks to investigate a running process, port, or resource usage
- User asks to read or triage system logs

## When NOT to use
- The task is primarily about code editing — use coding skill
- The task is a simple shell script — use execute_shell directly

## Procedure
1. **Detect the platform** first (macOS vs Linux): check output of `uname -s` or use system info from the prompt.
2. **Detect the package manager** via `command -v` cascade (see below).
3. **Check before acting**: verify state before installing, stopping, or deleting.
4. **Confirm destructive actions** with the user (package removal, service stop in production).
5. **Check logs** when diagnosing issues — don't guess, read the actual error.

## Package manager detection
```bash
command -v brew    && echo "homebrew"
command -v apt     && echo "apt (Debian/Ubuntu)"
command -v dnf     && echo "dnf (Fedora/RHEL)"
command -v pacman  && echo "pacman (Arch)"
command -v zypper  && echo "zypper (openSUSE)"
command -v apk     && echo "apk (Alpine)"
```

## Recipes

**Install a package (Linux):**
```bash
sudo apt install -y <package>   # Debian/Ubuntu
sudo dnf install -y <package>   # Fedora/RHEL
sudo pacman -S <package>        # Arch
```

**Install a package (macOS):**
```bash
brew install <package>
```

**Check if a package is installed:**
```bash
dpkg -l <package> 2>/dev/null   # Debian/Ubuntu
rpm -q <package> 2>/dev/null    # RPM-based
brew list <package> 2>/dev/null # macOS
```

**Service control (Linux systemd):**
```bash
systemctl status <service>
sudo systemctl start <service>
sudo systemctl stop <service>
sudo systemctl restart <service>
sudo systemctl enable <service>   # start on boot
sudo systemctl disable <service>
```

**Service control (macOS launchd):**
```bash
launchctl list | grep <service>
sudo launchctl start <label>
sudo launchctl stop <label>
```

**View recent system logs:**
```bash
journalctl -u <service> -n 50 --no-pager   # Linux: last 50 lines for service
journalctl -xe --no-pager                   # Linux: recent errors
log show --last 5m --predicate 'process == "nginx"'  # macOS
tail -100 /var/log/syslog                   # fallback
```

**Find process by name:**
```bash
ps aux | grep <name>
pgrep -a <name>
```

**Show what's listening on a port:**
```bash
ss -tlnp | grep :8080     # Linux
lsof -i :8080             # macOS / Linux
```

**Disk usage:**
```bash
df -h                          # filesystem overview
du --max-depth=1 -h /var/log   # dir breakdown (Linux)
du -sh * | sort -hr | head -20 # top consumers
```

**CPU and memory:**
```bash
top -bn1 | head -20    # snapshot
free -h                # Linux memory
vm_stat                # macOS memory
```

## Pitfalls
- `systemctl` is Linux only — on macOS, use `launchctl`.
- `apt` requires `sudo`; always confirm before running package changes.
- `journalctl` may require `sudo` to see other users' service logs.
- Never `kill -9` without understanding the process; use `kill` (SIGTERM) first.
- macOS System Integrity Protection blocks many operations on `/System/` even with sudo.

## References
- `references/systemd.md` — systemd unit files, journalctl patterns
- `references/macos-launchctl.md` — launchd plist format, common labels
