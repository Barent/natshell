#!/usr/bin/env bash
# NatShell global installer
# Usage: sudo bash install.sh
set -euo pipefail

INSTALL_DIR="/opt/natshell"
VENV_DIR="$INSTALL_DIR/.venv"
SYMLINK="/usr/local/bin/natshell"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Helpers ──────────────────────────────────────────────────────────────────

info()  { echo -e "\033[1;34m==>\033[0m $*"; }
ok()    { echo -e "\033[1;32m==>\033[0m $*"; }
warn()  { echo -e "\033[1;33mWARN:\033[0m $*"; }
die()   { echo -e "\033[1;31mERROR:\033[0m $*" >&2; exit 1; }

# ─── Preflight checks ────────────────────────────────────────────────────────

if [[ $EUID -ne 0 ]]; then
    die "This script must be run as root (sudo bash install.sh)"
fi

# Check Python version
PYTHON="python3"
if ! command -v "$PYTHON" &>/dev/null; then
    die "python3 not found. Install Python 3.11+ first."
fi

PY_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$("$PYTHON" -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$("$PYTHON" -c 'import sys; print(sys.version_info.minor)')

if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 11 ]]; }; then
    die "Python 3.11+ required (found $PY_VERSION)"
fi

info "Python $PY_VERSION — OK"

# ─── System dependencies ─────────────────────────────────────────────────────

info "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq python3-venv xclip git >/dev/null 2>&1
ok "System dependencies installed (python3-venv, xclip, git)"

# ─── Get source code ─────────────────────────────────────────────────────────

if [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    # Running from an existing checkout — copy it to /opt/natshell
    if [[ "$SCRIPT_DIR" != "$INSTALL_DIR" ]]; then
        info "Copying source from $SCRIPT_DIR to $INSTALL_DIR..."
        mkdir -p "$INSTALL_DIR"
        rsync -a --exclude='.venv' --exclude='__pycache__' --exclude='*.egg-info' \
            "$SCRIPT_DIR/" "$INSTALL_DIR/"
    else
        info "Already in $INSTALL_DIR"
    fi
elif [[ -d "$INSTALL_DIR/.git" ]]; then
    info "Updating existing installation..."
    cd "$INSTALL_DIR"
    git pull --ff-only
else
    info "Cloning NatShell to $INSTALL_DIR..."
    git clone https://github.com/natshell/natshell.git "$INSTALL_DIR"
fi

# ─── Virtual environment ─────────────────────────────────────────────────────

info "Creating virtual environment..."
"$PYTHON" -m venv "$VENV_DIR"
ok "Virtual environment created at $VENV_DIR"

# ─── Install NatShell package ─────────────────────────────────────────────────

info "Installing NatShell..."
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -e "$INSTALL_DIR" -q
ok "NatShell package installed"

# ─── Detect GPU and install llama-cpp-python ──────────────────────────────────

CMAKE_ARGS=""

if command -v vulkaninfo &>/dev/null 2>&1; then
    info "Vulkan detected — building llama-cpp-python with Vulkan support"
    CMAKE_ARGS="-DGGML_VULKAN=on"
elif command -v nvidia-smi &>/dev/null 2>&1; then
    info "NVIDIA GPU detected — building llama-cpp-python with CUDA support"
    CMAKE_ARGS="-DGGML_CUDA=on"
else
    info "No GPU detected — building llama-cpp-python for CPU"
fi

info "Installing llama-cpp-python (this may take a few minutes)..."
if [[ -n "$CMAKE_ARGS" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS" "$VENV_DIR/bin/pip" install llama-cpp-python --no-cache-dir -q
else
    "$VENV_DIR/bin/pip" install llama-cpp-python --no-cache-dir -q
fi
ok "llama-cpp-python installed"

# ─── Symlink ──────────────────────────────────────────────────────────────────

ln -sf "$VENV_DIR/bin/natshell" "$SYMLINK"
ok "Symlink created: $SYMLINK -> $VENV_DIR/bin/natshell"

# ─── Model download ──────────────────────────────────────────────────────────

echo ""
read -rp "Download the default model now (~2.5 GB)? [y/N] " answer
if [[ "${answer,,}" == "y" ]]; then
    # Run as the real user if invoked via sudo, otherwise as root
    if [[ -n "${SUDO_USER:-}" ]]; then
        info "Downloading model as $SUDO_USER..."
        sudo -u "$SUDO_USER" "$SYMLINK" --download
    else
        "$SYMLINK" --download
    fi
    ok "Model downloaded"
else
    info "Skipping model download. Run 'natshell --download' later."
fi

# ─── Done ─────────────────────────────────────────────────────────────────────

echo ""
ok "NatShell installed successfully!"
echo ""
echo "  Run:       natshell"
echo "  Config:    ~/.config/natshell/config.toml"
echo "  Models:    ~/.local/share/natshell/models/"
echo "  Uninstall: sudo bash $INSTALL_DIR/uninstall.sh"
echo ""
