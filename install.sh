#!/usr/bin/env bash
# NatShell installer
# Usage: bash install.sh
set -euo pipefail

INSTALL_DIR="$HOME/.local/share/natshell/app"
VENV_DIR="$INSTALL_DIR/.venv"
BIN_DIR="$HOME/.local/bin"
SYMLINK="$BIN_DIR/natshell"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Helpers ──────────────────────────────────────────────────────────────────

info()  { echo -e "\033[1;34m==>\033[0m $*"; }
ok()    { echo -e "\033[1;32m==>\033[0m $*"; }
warn()  { echo -e "\033[1;33mWARN:\033[0m $*"; }
die()   { echo -e "\033[1;31mERROR:\033[0m $*" >&2; exit 1; }

# ─── Preflight checks ────────────────────────────────────────────────────────

if [[ $EUID -eq 0 ]]; then
    warn "Running as root is not recommended — NatShell installs to your home directory."
    echo "  Consider running without sudo:  bash install.sh"
    echo ""
    read -rp "  Continue as root anyway? [y/N]: " root_answer
    if [[ "${root_answer,,}" != "y" ]]; then
        exit 1
    fi
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

# ─── System dependency checks ─────────────────────────────────────────────────

# Detect package manager
PKG_MGR=""
if command -v apt-get &>/dev/null; then
    PKG_MGR="apt"
elif command -v dnf &>/dev/null; then
    PKG_MGR="dnf"
elif command -v pacman &>/dev/null; then
    PKG_MGR="pacman"
fi

# Helper: offer to install a system package, or show manual instructions
install_pkg() {
    local apt_pkg="$1" dnf_pkg="$2" pacman_pkg="$3"

    case "$PKG_MGR" in
        apt)
            read -rp "  Install $apt_pkg now? (requires sudo) [Y/n]: " answer
            if [[ -z "$answer" || "${answer,,}" == "y" ]]; then
                sudo apt-get install -y "$apt_pkg"
                return $?
            fi
            ;;
        dnf)
            read -rp "  Install $dnf_pkg now? (requires sudo) [Y/n]: " answer
            if [[ -z "$answer" || "${answer,,}" == "y" ]]; then
                sudo dnf install -y "$dnf_pkg"
                return $?
            fi
            ;;
        pacman)
            read -rp "  Install $pacman_pkg now? (requires sudo) [Y/n]: " answer
            if [[ -z "$answer" || "${answer,,}" == "y" ]]; then
                sudo pacman -S --noconfirm "$pacman_pkg"
                return $?
            fi
            ;;
    esac

    # Unknown distro or user declined
    die "Please install it manually:
  Debian/Ubuntu:  sudo apt install $apt_pkg
  Fedora:         sudo dnf install $dnf_pkg
  Arch:           sudo pacman -S $pacman_pkg"
}

# Verify python3-venv is available (separate package on Debian/Ubuntu)
if ! "$PYTHON" -m venv --help &>/dev/null; then
    warn "python3-venv is required but not installed."
    install_pkg "python3-venv" "python3-libs" "python"
    # Re-check after install
    if ! "$PYTHON" -m venv --help &>/dev/null; then
        die "python3-venv still not available after install attempt."
    fi
fi
ok "python3-venv — OK"

# Verify a C++ compiler is available (needed to build llama-cpp-python)
if ! command -v g++ &>/dev/null && ! command -v c++ &>/dev/null && ! command -v clang++ &>/dev/null; then
    warn "A C++ compiler is required to build llama-cpp-python."
    install_pkg "g++" "gcc-c++" "gcc"
    # Re-check after install
    if ! command -v g++ &>/dev/null && ! command -v c++ &>/dev/null && ! command -v clang++ &>/dev/null; then
        die "C++ compiler still not available after install attempt."
    fi
fi
ok "C++ compiler — OK"

# ─── Get source code ─────────────────────────────────────────────────────────

if [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    # Running from an existing checkout — copy to install dir
    if [[ "$SCRIPT_DIR" != "$INSTALL_DIR" ]]; then
        info "Copying source from $SCRIPT_DIR to $INSTALL_DIR..."
        mkdir -p "$INSTALL_DIR"
        cp -a "$SCRIPT_DIR/." "$INSTALL_DIR/"
        # Clean build artifacts from the copy
        rm -rf "$INSTALL_DIR/.venv"
        find "$INSTALL_DIR" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
        find "$INSTALL_DIR" -name '*.egg-info' -type d -exec rm -rf {} + 2>/dev/null || true
    else
        info "Already in $INSTALL_DIR"
    fi
elif [[ -d "$INSTALL_DIR/.git" ]]; then
    info "Updating existing installation..."
    cd "$INSTALL_DIR"
    git pull --ff-only
else
    info "Cloning NatShell to $INSTALL_DIR..."
    git clone https://github.com/Barent/natshell.git "$INSTALL_DIR"
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
GPU_DETECTED=false

if command -v vulkaninfo &>/dev/null 2>&1; then
    info "Vulkan detected — building llama-cpp-python with Vulkan support"
    CMAKE_ARGS="-DGGML_VULKAN=on"
    GPU_DETECTED=true
elif command -v nvidia-smi &>/dev/null 2>&1; then
    info "NVIDIA GPU detected — building llama-cpp-python with CUDA support"
    CMAKE_ARGS="-DGGML_CUDA=on"
    GPU_DETECTED=true
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

mkdir -p "$BIN_DIR"
ln -sf "$VENV_DIR/bin/natshell" "$SYMLINK"
ok "Symlink created: $SYMLINK"

# Check if ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    warn "$BIN_DIR is not in your PATH"
    echo "  Add this to your shell profile (~/.bashrc or ~/.zshrc):"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
fi

# ─── Interactive Setup ────────────────────────────────────────────────────────

CONFIG_DIR="$HOME/.config/natshell"
CONFIG_FILE="$CONFIG_DIR/config.toml"

echo ""
echo "  ─── NatShell Setup ───"
echo ""
echo "  Select a model preset:"
echo ""
echo "    1) Light    — Qwen3-4B  (~2.5 GB, low RAM)"
echo "    2) Standard — Qwen3-8B  (~5 GB, better quality)"
echo "    3) Both     — download Light + Standard (~7.5 GB)"
echo "    4) Remote only — use an Ollama server (no local download)"
echo "    5) Skip — configure later"
echo ""
read -rp "  Choice [1]: " model_choice
model_choice="${model_choice:-1}"

DOWNLOAD_MODEL=false
DOWNLOAD_BOTH=false
SETUP_OLLAMA=false
WRITE_MODEL_CONFIG=false
HF_REPO=""
HF_FILE=""

case "$model_choice" in
    1)
        info "Light preset selected (Qwen3-4B)"
        DOWNLOAD_MODEL=true
        ;;
    2)
        info "Standard preset selected (Qwen3-8B)"
        DOWNLOAD_MODEL=true
        WRITE_MODEL_CONFIG=true
        HF_REPO="Qwen/Qwen3-8B-GGUF"
        HF_FILE="Qwen3-8B-Q4_K_M.gguf"
        ;;
    3)
        info "Both models selected (Qwen3-4B + Qwen3-8B)"
        DOWNLOAD_BOTH=true
        WRITE_MODEL_CONFIG=true
        HF_REPO="Qwen/Qwen3-8B-GGUF"
        HF_FILE="Qwen3-8B-Q4_K_M.gguf"
        ;;
    4)
        info "Remote only — skipping local model download"
        SETUP_OLLAMA=true
        ;;
    5)
        info "Skipping setup. Run 'natshell' later to configure."
        ;;
    *)
        warn "Invalid choice '$model_choice', defaulting to Light preset"
        model_choice=1
        DOWNLOAD_MODEL=true
        ;;
esac

# Offer Ollama setup for local model options
if [[ "$model_choice" == "1" || "$model_choice" == "2" || "$model_choice" == "3" ]]; then
    echo ""
    read -rp "  Configure a remote Ollama server too? [y/N]: " ollama_answer
    if [[ "${ollama_answer,,}" == "y" ]]; then
        SETUP_OLLAMA=true
    fi
fi

# ─── Ollama Setup ────────────────────────────────────────────────────────────

OLLAMA_URL=""
OLLAMA_MODEL=""

if [[ "$SETUP_OLLAMA" == true ]]; then
    echo ""
    echo "  ─── Ollama Server Setup ───"
    echo ""
    read -rp "  Server URL [http://localhost:11434]: " ollama_url_input
    OLLAMA_URL="${ollama_url_input:-http://localhost:11434}"

    # Ping the server
    echo ""
    info "Checking server at $OLLAMA_URL..."
    if curl -sf "${OLLAMA_URL%/}/" --connect-timeout 5 >/dev/null 2>&1; then
        ok "Server is reachable"

        # List available models
        models_json=$(curl -sf "${OLLAMA_URL%/}/api/tags" --connect-timeout 5 2>/dev/null || true)
        if [[ -n "$models_json" ]]; then
            echo ""
            echo "$models_json" | "$PYTHON" -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = data.get('models', [])
    if models:
        print('  Available models:')
        for m in models:
            name = m.get('name', '?')
            size_gb = m.get('size', 0) / (1024**3)
            print(f'    - {name}  ({size_gb:.1f} GB)')
    else:
        print('  No models found on server.')
except Exception:
    print('  Could not parse model list.')
" 2>/dev/null || echo "  Could not query model list."
        fi

        echo ""
        read -rp "  Default model name [qwen3:4b]: " ollama_model_input
        OLLAMA_MODEL="${ollama_model_input:-qwen3:4b}"
    else
        warn "Server not reachable at $OLLAMA_URL"
        echo ""
        read -rp "  Save this URL anyway for later? [Y/n]: " save_anyway
        if [[ -z "$save_anyway" || "${save_anyway,,}" == "y" ]]; then
            read -rp "  Default model name [qwen3:4b]: " ollama_model_input
            OLLAMA_MODEL="${ollama_model_input:-qwen3:4b}"
        else
            info "Skipping Ollama configuration"
            OLLAMA_URL=""
        fi
    fi
fi

# ─── Write Config ────────────────────────────────────────────────────────────

# Write config if any section needs non-default values
if [[ "$WRITE_MODEL_CONFIG" == true || "$GPU_DETECTED" == true || -n "$OLLAMA_URL" ]]; then
    # Back up existing config if present
    if [[ -f "$CONFIG_FILE" ]]; then
        cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
        info "Existing config backed up to config.toml.bak"
    fi

    # Build config content (only sections that differ from defaults)
    config_content=""

    # [model] section — written for 8B preset and/or GPU detection
    if [[ "$WRITE_MODEL_CONFIG" == true || "$GPU_DETECTED" == true ]]; then
        config_content="[model]"$'\n'
        if [[ "$WRITE_MODEL_CONFIG" == true ]]; then
            config_content+="hf_repo = \"${HF_REPO}\""$'\n'
            config_content+="hf_file = \"${HF_FILE}\""$'\n'
        fi
        if [[ "$GPU_DETECTED" == true ]]; then
            config_content+="n_gpu_layers = -1"$'\n'
        fi
        config_content+=$'\n'
    fi

    if [[ -n "$OLLAMA_URL" ]]; then
        config_content+="[ollama]
url = \"${OLLAMA_URL}\"
default_model = \"${OLLAMA_MODEL}\"
"
    fi

    mkdir -p "$CONFIG_DIR"
    printf '%s' "$config_content" > "$CONFIG_FILE"

    ok "Config written to $CONFIG_FILE"
fi

# ─── Model Download ──────────────────────────────────────────────────────────

if [[ "$DOWNLOAD_BOTH" == true ]]; then
    echo ""
    info "Downloading Light model (Qwen3-4B)..."
    "$VENV_DIR/bin/python" -c "
from huggingface_hub import hf_hub_download
from pathlib import Path
model_dir = Path.home() / '.local' / 'share' / 'natshell' / 'models'
model_dir.mkdir(parents=True, exist_ok=True)
hf_hub_download(repo_id='Qwen/Qwen3-4B-GGUF', filename='Qwen3-4B-Q4_K_M.gguf', local_dir=str(model_dir))
print('Done.')
"
    ok "Light model downloaded"

    info "Downloading Standard model (Qwen3-8B)..."
    "$SYMLINK" --download
    ok "Standard model downloaded"
elif [[ "$DOWNLOAD_MODEL" == true ]]; then
    echo ""
    info "Downloading model..."
    "$SYMLINK" --download
    ok "Model downloaded"
fi

# ─── Done ─────────────────────────────────────────────────────────────────────

echo ""
ok "NatShell installed successfully!"
echo ""
echo "  Run:       natshell"
echo "  Config:    ~/.config/natshell/config.toml"
echo "  Models:    ~/.local/share/natshell/models/"
echo "  Uninstall: bash $INSTALL_DIR/uninstall.sh"
echo ""
