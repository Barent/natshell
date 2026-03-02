"""NatShell entry point — CLI argument parsing, model setup, and app launch."""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

from natshell.config import load_config


def _get_version() -> str:
    """Return the installed package version, or fall back to 'unknown'."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"natshell {version('natshell')}"
    except PackageNotFoundError:
        return "natshell (unknown version — not installed as package)"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="natshell",
        description="NatShell — Natural language shell interface for Linux",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=_get_version(),
    )
    parser.add_argument(
        "--config",
        "-c",
        help="Path to config.toml file",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Path to a GGUF model file (overrides config)",
    )
    parser.add_argument(
        "--remote",
        help="URL of an OpenAI-compatible API to use instead of local model "
        "(e.g., http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--remote-model",
        help="Model name for the remote API (e.g., qwen3:4b)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force local model (ignore remote/Ollama configuration)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the default model and exit",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Pull latest code and reinstall (git installs only)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        help="Write all log messages (DEBUG level) to a file. "
        "Useful for real-time monitoring with tail -f.",
    )
    parser.add_argument(
        "--headless",
        metavar="PROMPT",
        help="Run a single prompt without the TUI and exit. "
        "Response text goes to stdout (pipeable), diagnostics to stderr.",
    )
    parser.add_argument(
        "--danger-fast",
        action="store_true",
        help="Skip all confirmation dialogs. BLOCKED commands are still blocked. "
        "Only use this on VMs or test environments.",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Run as an MCP (Model Context Protocol) server over stdio. "
        "Requires the 'mcp' package: pip install 'natshell[mcp]'",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s: %(message)s")

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle self-update
    if args.update:
        _self_update()
        return

    # Load config
    config = load_config(args.config)

    # Conflict check
    if args.local and args.remote:
        print("Error: --local and --remote cannot be used together.")
        sys.exit(1)

    # Override config with CLI args
    if args.model:
        config.model.path = args.model
    if args.remote:
        config.remote.url = args.remote
    if args.remote_model:
        config.remote.model = args.remote_model

    # Handle model download
    if args.download or config.model.path == "auto":
        model_path = _ensure_model(config)
        if args.download:
            print(f"Model ready at: {model_path}")
            return
        config.model.path = model_path

    # Determine remote URL and model (CLI --remote > [remote] > [ollama])
    remote_url = config.remote.url
    remote_model = config.remote.model
    remote_api_key = config.remote.api_key
    use_remote = bool(remote_url)
    fallback_config = None

    # Ensure remote URL has a scheme
    if remote_url and not remote_url.startswith(("http://", "https://")):
        remote_url = f"http://{remote_url}"

    if not remote_url and config.ollama.url:
        from natshell.inference.ollama import normalize_base_url

        base = normalize_base_url(config.ollama.url)
        remote_url = f"{base}/v1"
        remote_model = remote_model or config.ollama.default_model or "qwen3:4b"
        use_remote = True

    if not remote_model:
        remote_model = "qwen3:4b"

    # Apply --local override before preference logic
    if args.local:
        use_remote = False

    # Apply persisted engine preference (CLI flags override)
    cli_forced_remote = bool(args.remote)
    cli_forced_local = bool(args.model) or args.local
    if not cli_forced_remote and not cli_forced_local:
        if config.engine.preferred == "local":
            use_remote = False
        elif config.engine.preferred == "remote":
            pass  # keep use_remote as-is (try remote if URL exists)

    # Build the inference engine
    if use_remote:
        from natshell.inference.ollama import ping_server

        print(f"Checking remote server: {remote_url}...")
        reachable = asyncio.run(ping_server(remote_url))

        if reachable:
            from natshell.inference.ollama import get_model_context_length
            from natshell.inference.remote import RemoteEngine

            if config.remote.n_ctx > 0:
                n_ctx = config.remote.n_ctx
            elif config.ollama.n_ctx > 0:
                n_ctx = config.ollama.n_ctx
            else:
                n_ctx = asyncio.run(get_model_context_length(remote_url, remote_model))
            engine = RemoteEngine(
                base_url=remote_url,
                model=remote_model,
                api_key=remote_api_key,
                n_ctx=n_ctx,
            )
            fallback_config = config.model
            print(f"Using remote model: {remote_model} at {remote_url}")
        else:
            print(f"Remote server unreachable at {remote_url}. Falling back to local model.")
            use_remote = False

    if not use_remote:
        from natshell.inference.local import LocalEngine

        print(f"Loading model: {config.model.path}...")
        engine = LocalEngine(
            model_path=config.model.path,
            n_ctx=config.model.n_ctx,
            n_threads=config.model.n_threads,
            n_gpu_layers=config.model.n_gpu_layers,
            main_gpu=config.model.main_gpu,
            prompt_cache=config.model.prompt_cache,
            prompt_cache_mb=config.model.prompt_cache_mb,
        )
        try:
            from llama_cpp import llama_supports_gpu_offload

            if config.model.n_gpu_layers != 0 and not llama_supports_gpu_offload():
                from natshell.gpu import detect_gpus
                from natshell.platform import is_macos

                gpus = detect_gpus()
                if is_macos():
                    gpu_flag = "-DGGML_METAL=on"
                else:
                    gpu_flag = "-DGGML_VULKAN=on"

                print(
                    "WARNING: GPU offloading requested but llama-cpp-python"
                    " was built without GPU support."
                )
                if gpus:
                    gpu = gpus[0]
                    vram = f" ({gpu.vram_mb} MB VRAM)" if gpu.vram_mb else ""
                    print(f"  Detected GPU: {gpu.name}{vram}")

                print(
                    f"  Reinstall with: CMAKE_ARGS=\"{gpu_flag}\""
                    " pip install llama-cpp-python"
                    " --no-binary llama-cpp-python --force-reinstall"
                )

                if not is_macos():
                    _print_vulkan_dep_hint()
        except ImportError:
            pass
        print("Model loaded.")

    # Build the tool registry
    from natshell.tools.registry import create_default_registry

    tools = create_default_registry()

    # Load plugins
    from natshell.plugins import load_plugins

    loaded = load_plugins(tools)
    if loaded:
        print(f"Loaded {loaded} plugin(s)")

    # Build the safety classifier
    from natshell.safety.classifier import SafetyClassifier

    safety = SafetyClassifier(config.safety)

    # Inject safety config into the natshell_help tool
    from natshell.tools.natshell_help import set_safety_config

    set_safety_config(config.safety)

    # MCP server mode — run over stdio and exit
    if args.mcp:
        try:
            from natshell.mcp_server import run_mcp_server

            asyncio.run(run_mcp_server(tools, safety, config.mcp))
        except ImportError:
            print("MCP support requires the 'mcp' package: pip install 'natshell[mcp]'")
            sys.exit(1)
        sys.exit(0)

    # Build the agent
    from natshell.agent.loop import AgentLoop

    agent = AgentLoop(
        engine=engine,
        tools=tools,
        safety=safety,
        config=config.agent,
        fallback_config=fallback_config,
    )

    # Gather system context and initialize agent
    from natshell.agent.context import gather_system_context

    print("Gathering system information...")
    context = asyncio.run(gather_system_context())
    agent.initialize(context)

    # Headless mode — single-shot CLI, no TUI
    if args.headless:
        from natshell.headless import run_headless

        exit_code = asyncio.run(
            run_headless(agent, args.headless, auto_approve=args.danger_fast)
        )
        sys.exit(exit_code)

    # Launch the TUI
    from natshell.app import NatShellApp

    if args.danger_fast:
        print("WARNING: --danger-fast is active. All confirmations will be skipped.")
    app = NatShellApp(agent=agent, config=config, skip_permissions=args.danger_fast)
    app.run()


def _print_vulkan_dep_hint() -> None:
    """Print distro-specific instructions for installing Vulkan build deps."""
    import shutil
    from pathlib import Path

    if shutil.which("rpm-ostree") and Path("/run/ostree-booted").exists():
        print("  Install Vulkan build deps: sudo rpm-ostree install vulkan-devel glslc")
        print("  Then reboot and re-run install.sh")
    elif shutil.which("dnf"):
        print("  Install Vulkan build deps: sudo dnf install vulkan-devel glslc")
    elif shutil.which("apt-get"):
        print("  Install Vulkan build deps: sudo apt install libvulkan-dev glslang-tools")
    elif shutil.which("pacman"):
        print("  Install Vulkan build deps: sudo pacman -S vulkan-headers glslang")
    else:
        print("  Ensure Vulkan development headers and a GLSL shader compiler are installed.")


def _self_update() -> None:
    """Pull latest code and reinstall from the git checkout."""
    # Navigate from src/natshell/__main__.py -> project root
    root = Path(__file__).resolve().parent.parent.parent
    if not (root / ".git").exists():
        print("Not a git install. Re-run install.sh to update.")
        sys.exit(1)

    print(f"Updating NatShell from {root}...")
    result = subprocess.run(
        ["git", "-C", str(root), "pull", "--ff-only"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"git pull failed:\n{result.stderr.strip()}")
        sys.exit(1)
    print(result.stdout.strip())

    # Reinstall using the same Python that's running us
    pip = str(Path(sys.executable).parent / "pip")
    result = subprocess.run(
        [pip, "install", "-e", str(root), "-q"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"pip install failed:\n{result.stderr.strip()}")
        sys.exit(1)

    print("Update complete.")


def _ensure_model(config) -> str:
    """Ensure the default model is downloaded. Returns the model path."""
    model_dir = Path.home() / ".local" / "share" / "natshell" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    target = model_dir / config.model.hf_file
    if target.exists():
        return str(target)

    # Prompt user
    print("\nNo local model found.")
    print(f"Download {config.model.hf_file} from {config.model.hf_repo}?")
    print("This is approximately 2.5 GB.\n")

    response = input("Download now? [Y/n]: ").strip().lower()
    if response and response != "y":
        print("No model available. Use --model or --remote to specify one.")
        sys.exit(1)

    try:
        from huggingface_hub import hf_hub_download

        print("Downloading from HuggingFace...")
        path = hf_hub_download(
            repo_id=config.model.hf_repo,
            filename=config.model.hf_file,
            local_dir=str(model_dir),
        )
        print(f"Model saved to: {path}")
        return path

    except ImportError:
        print("huggingface-hub is required for model download.")
        print("Install it: pip install huggingface-hub")
        sys.exit(1)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
