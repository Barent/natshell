#!/usr/bin/env bash
# NatShell uninstaller
# Usage: bash uninstall.sh
set -euo pipefail

INSTALL_DIR="$HOME/.local/share/natshell/app"
SYMLINK="$HOME/.local/bin/natshell"

info()  { echo -e "\033[1;34m==>\033[0m $*"; }
ok()    { echo -e "\033[1;32m==>\033[0m $*"; }

if [[ -L "$SYMLINK" ]]; then
    info "Removing symlink $SYMLINK..."
    rm -f "$SYMLINK"
fi

if [[ -d "$INSTALL_DIR" ]]; then
    info "Removing $INSTALL_DIR..."
    rm -rf "$INSTALL_DIR"
fi

ok "NatShell uninstalled."
echo ""
echo "  User data preserved at:"
echo "    ~/.local/share/natshell/models/  (downloaded models)"
echo "    ~/.config/natshell/              (config)"
echo ""
echo "  To remove everything:"
echo "    rm -rf ~/.local/share/natshell ~/.config/natshell"
echo ""
