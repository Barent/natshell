# macOS launchd / launchctl reference

## Plist locations
| Domain | Path | When runs |
|--------|------|-----------|
| User agents | `~/Library/LaunchAgents/` | At user login |
| System agents | `/Library/LaunchAgents/` | At user login (all users) |
| System daemons | `/Library/LaunchDaemons/` | At boot (root) |
| Apple daemons | `/System/Library/LaunchDaemons/` | Do not edit |

## Minimal LaunchAgent plist
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.example.myapp</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/myapp</string>
        <string>--config</string>
        <string>/etc/myapp/config.toml</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/myapp.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/myapp.stderr.log</string>
</dict>
</plist>
```

## launchctl commands
```bash
# Load/unload (older API — still works)
launchctl load ~/Library/LaunchAgents/com.example.myapp.plist
launchctl unload ~/Library/LaunchAgents/com.example.myapp.plist

# Bootstrap/bootout (modern API, macOS 10.11+)
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.example.myapp.plist
launchctl bootout  gui/$(id -u) ~/Library/LaunchAgents/com.example.myapp.plist

# Start/stop
launchctl start com.example.myapp
launchctl stop  com.example.myapp

# List all loaded agents
launchctl list | grep example

# Check status / last exit code
launchctl list com.example.myapp
```

## Logs
```bash
log show --last 5m --predicate 'process == "myapp"'
log stream --predicate 'process == "myapp"'  # live tail
tail -f /tmp/myapp.stderr.log
```
