# systemd reference

## Unit file locations
- `/etc/systemd/system/` — user-managed units (highest priority)
- `/lib/systemd/system/` — package-installed units
- After editing: `sudo systemctl daemon-reload`

## Minimal service unit file
```ini
[Unit]
Description=My App
After=network.target

[Service]
Type=simple
User=myuser
WorkingDirectory=/opt/myapp
ExecStart=/opt/myapp/bin/myapp --config /etc/myapp/config.toml
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## journalctl patterns
```bash
journalctl -u myapp.service -n 100 --no-pager    # last 100 lines
journalctl -u myapp.service -f                   # follow (tail)
journalctl -u myapp.service --since "1 hour ago"
journalctl -u myapp.service --since "2024-01-01" --until "2024-01-02"
journalctl -p err -xe --no-pager                 # errors only
journalctl --disk-usage                          # log disk usage
```

## Service dependencies
```bash
systemctl list-dependencies myapp.service
systemctl list-units --state=failed
```
