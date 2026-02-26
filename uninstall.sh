#!/usr/bin/env bash
# NatShell uninstaller
# Usage: sudo bash uninstall.sh
set -euo pipefail

INSTALL_DIR="/opt/natshell"
SYMLINK="/usr/local/bin/natshell"

info()  { echo -e "\033[1;34m==>\033[0m $*"; }
ok()    { echo -e "\033[1;32m==>\033[0m $*"; }
die()   { echo -e "\033[1;31mERROR:\033[0m $*" >&2; exit 1; }

if [[ $EUID -ne 0 ]]; then
    die "This script must be run as root (sudo bash uninstall.sh)"
fi

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
echo "    ~/.local/share/natshell/  (models)"
echo "    ~/.config/natshell/       (config)"
echo ""
echo "  To remove user data too:"
echo "    rm -rf ~/.local/share/natshell ~/.config/natshell"
echo ""
