"""NatShell entry point — CLI argument parsing, model setup, and app launch."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
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
    parser.add_argument(
        "--plan",
        metavar="DESCRIPTION",
        help="Generate a PLAN.md from a natural language description and exit. "
        "Plan text goes to stdout, diagnostics to stderr.",
    )
    parser.add_argument(
        "--exeplan",
        metavar="FILE",
        help="Execute a plan file step by step and exit. "
        "Use --danger-fast to auto-approve all confirmations.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously interrupted --exeplan run from the last failed step. "
        "Reads state from PLAN.state.json alongside the plan file.",
    )
    parser.add_argument(
        "--no-setup",
        action="store_true",
        help="Skip the first-run setup wizard",
    )
    parser.add_argument(
        "--profile",
        metavar="NAME",
        help="Apply a named configuration profile before starting",
    )

    args = parser.parse_args()

    # Setup logging — nothing must reach stderr while the Textual TUI owns
    # the terminal, because any stderr output corrupts the display.
    root_logger = logging.getLogger()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG if args.verbose else logging.WARNING)

    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.DEBUG)

    # Handle self-update
    if args.update:
        _self_update()
        return

    # Conflict check (before wizard to avoid unnecessary interaction)
    if args.local and args.remote:
        print("Error: --local and --remote cannot be used together.")
        sys.exit(1)

    # First-run setup wizard (skip when CLI specifies engine explicitly)
    from natshell.setup_wizard import should_run_wizard

    if should_run_wizard(
        config_path=Path(args.config) if args.config else None,
        no_setup=args.no_setup or bool(args.local) or bool(args.remote)
        or bool(args.model),
        headless=bool(args.headless) or bool(args.plan) or bool(args.exeplan),
        mcp=args.mcp,
        download=args.download,
    ):
        from natshell.setup_wizard import run_setup_wizard

        run_setup_wizard()

    # Load config
    config = load_config(args.config)

    # Apply profile if specified
    if args.profile:
        from natshell.config import apply_profile, list_profiles

        try:
            apply_profile(config, args.profile)
            print(f"Applied profile: {args.profile}")
        except KeyError:
            available = list_profiles(config)
            if available:
                print(f"Unknown profile: {args.profile}")
                print(f"Available profiles: {', '.join(available)}")
            else:
                print(f"Unknown profile: {args.profile}")
                print("No profiles defined. Add [profiles.*] sections to config.toml.")
            sys.exit(1)

    # Override config with CLI args
    if args.model:
        config.model.path = args.model
    if args.remote:
        config.remote.url = args.remote
    if args.remote_model:
        config.remote.model = args.remote_model

    # TODO: Remove this Windows ARM64 override once llama-cpp-python builds
    # reliably with MSVC on ARM64 (currently requires clang-cl).
    from natshell.platform import is_arm64, is_windows

    if is_windows() and is_arm64() and not args.local and not args.model:
        if config.engine.preferred == "auto":
            config.engine.preferred = "remote"
        if not config.ollama.url and not config.remote.url:
            config.ollama.url = "http://localhost:11434"

    # Handle model download — skip when engine is set to remote
    # (avoids downloading a multi-GB model the user doesn't need)
    _prefers_remote = (
        config.engine.preferred == "remote"
        and not args.model
        and not args.local
        and (config.remote.url or config.ollama.url)
    )
    if args.download or (config.model.path == "auto" and not _prefers_remote):
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

    # Diagnostic: show whether an API key was found
    if remote_url:
        if remote_api_key:
            print("API key: configured")
        else:
            print("API key: not set — set NATSHELL_API_KEY or api_key in config.toml")
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
        reachable = asyncio.run(ping_server(remote_url, api_key=remote_api_key))

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
        try:
            engine = LocalEngine(
                model_path=config.model.path,
                n_ctx=config.model.n_ctx,
                n_threads=config.model.n_threads,
                n_gpu_layers=config.model.n_gpu_layers,
                main_gpu=config.model.main_gpu,
                prompt_cache=config.model.prompt_cache,
                prompt_cache_mb=config.model.prompt_cache_mb,
            )
        except ModuleNotFoundError:
            from natshell.platform import is_arm64, is_windows

            print("\nERROR: llama-cpp-python is not installed.\n")
            if is_windows() and is_arm64():
                print(
                    "On Windows ARM64, MSVC cannot compile llama.cpp.\n"
                    "Build with clang-cl instead:\n"
                    "\n"
                    '  $env:CMAKE_ARGS="-G Ninja'
                    " -DCMAKE_C_COMPILER=clang-cl"
                    ' -DCMAKE_CXX_COMPILER=clang-cl"\n'
                    "  pip install llama-cpp-python"
                    " --no-binary llama-cpp-python"
                    " --no-cache-dir\n"
                    "\n"
                    "Requires: Visual Studio 2022 with"
                    ' "C++ Clang Compiler for Windows"'
                    " component.\n"
                )
            elif is_windows():
                print(
                    "Install it with:\n"
                    "  pip install llama-cpp-python\n"
                    "\n"
                    "For GPU acceleration (CUDA or Vulkan), build"
                    " from source:\n"
                    '  $env:CMAKE_ARGS="-DGGML_CUDA=on"  '
                    " # NVIDIA CUDA\n"
                    '  $env:CMAKE_ARGS="-DGGML_VULKAN=on" '
                    " # AMD/Intel Vulkan\n"
                    "  pip install llama-cpp-python"
                    " --no-binary llama-cpp-python"
                    " --no-cache-dir\n"
                    "\n"
                    "Requires: Visual Studio 2022 with"
                    ' "Desktop development with C++"'
                    " workload.\n"
                )
            else:
                print(
                    "Install it with:\n"
                    "  pip install llama-cpp-python\n"
                )
            print(
                "Or use Ollama instead"
                + (" (recommended on Windows):\n" if is_windows()
                   else ":\n")
                + "  1. Install Ollama from https://ollama.com\n"
                "  2. ollama pull qwen3:8b\n"
                '  3. Set preferred = "remote" and'
                ' url = "http://localhost:11434/v1"'
                " in config.toml\n"
            )
            # Show NPU hint if detected
            if is_windows():
                try:
                    from natshell.gpu import detect_npu

                    npu = detect_npu()
                    if npu:
                        npu_backend = (
                            "Snapdragon NPU" if npu.vendor == "qualcomm"
                            else "NPU"
                        )
                        print(
                            f"  Detected NPU: {npu.name}\n"
                            f"  Ollama supports {npu_backend}"
                            " acceleration out of the box.\n"
                        )
                except Exception:
                    pass
            sys.exit(1)
        try:
            from llama_cpp import llama_supports_gpu_offload

            if config.model.n_gpu_layers != 0 and not llama_supports_gpu_offload():
                from natshell.gpu import detect_gpus
                from natshell.platform import is_macos, is_windows

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

                if is_windows():
                    _print_windows_gpu_hint()
                elif not is_macos():
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

    # Inject live config into the update_config tool
    from natshell.tools.update_config import set_live_config

    set_live_config(config)

    # Inject kiwix URL into the kiwix_search tool and auto-discover if needed
    from natshell.tools.kiwix_search import discover_and_set_kiwix_url, set_kiwix_url

    set_kiwix_url(config.kiwix.url)
    asyncio.run(discover_and_set_kiwix_url())

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
        prompt_config=config.prompt,
        memory_config=config.memory,
        memory_store_config=config.memory_store,
    )

    # Gather system context and initialize agent
    from natshell.agent.context import gather_system_context

    print("Gathering system information...")
    context = asyncio.run(gather_system_context())
    agent.initialize(context)

    try:
        # Headless plan generation
        if args.plan:
            from natshell.headless import run_headless_plan

            exit_code = asyncio.run(
                run_headless_plan(agent, args.plan, auto_approve=args.danger_fast)
            )
            sys.exit(exit_code)

        # Headless plan execution
        if args.exeplan:
            from natshell.headless import run_headless_exeplan

            exit_code = asyncio.run(
                run_headless_exeplan(
                    agent, args.exeplan,
                    auto_approve=args.danger_fast,
                    resume=args.resume,
                )
            )
            sys.exit(exit_code)

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
        # Silence the console handler before the TUI takes the terminal —
        # ANY stderr output during Textual rendering corrupts the display.
        console_handler.setLevel(logging.CRITICAL)
        app.run()
    finally:
        if hasattr(engine, "close"):
            try:
                asyncio.run(engine.close())
            except Exception:
                pass


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


def _print_windows_gpu_hint() -> None:
    """Print Windows-specific instructions for GPU-accelerated builds."""
    from natshell.platform import is_arm64

    if is_arm64():
        print(
            "  Install the Vulkan SDK from"
            " https://vulkan.lunarg.com/sdk/home\n"
            "  On ARM64: ensure the Qualcomm Adreno GPU"
            " driver is up to date.\n"
            "  Build with clang-cl: set CMAKE_ARGS to include"
            " -DCMAKE_C_COMPILER=clang-cl"
            " -DCMAKE_CXX_COMPILER=clang-cl"
        )
    else:
        print(
            "  For NVIDIA GPUs: install CUDA Toolkit from"
            " https://developer.nvidia.com/cuda-downloads\n"
            "  For AMD/Intel GPUs: install the Vulkan SDK"
            " from https://vulkan.lunarg.com/sdk/home"
        )


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
        print("Fast-forward merge failed — you may have local changes or the history has diverged.")
        print()
        print("To update manually:")
        print("  git stash && git pull --rebase && git stash pop")
        print()
        print("To force-update (discards local changes):")
        print(f"  git -C {root} fetch origin && git -C {root} reset --hard origin/main")
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
    from natshell.platform import data_dir

    model_dir = data_dir() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    target = model_dir / config.model.hf_file
    if target.exists():
        return str(target)

    # Estimate download size from filename
    fname = config.model.hf_file.lower()
    if "mistral" in fname and "nemo" in fname:
        size_hint = "~7.5 GB"
    elif "8b" in fname:
        size_hint = "~5 GB"
    else:
        size_hint = "~2.5 GB"

    # Check if llama-cpp-python is available before downloading
    from natshell.platform import is_arm64, is_windows

    llama_ok = True
    try:
        import llama_cpp  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        llama_ok = False

    if not llama_ok:
        print("\n⚠ llama-cpp-python is not installed.")
        if is_windows():
            print(
                "  Local models require llama-cpp-python"
                " to run.\n"
                "\n"
                "  Recommended: use Ollama instead:\n"
                "    1. Install from https://ollama.com\n"
                "    2. ollama pull qwen3:4b\n"
                "    3. Re-run natshell — it will auto-detect"
                " Ollama.\n"
            )
            if is_arm64():
                print(
                    "  Or build llama-cpp-python with clang-cl"
                    " (see natshell docs).\n"
                )
            else:
                print(
                    "  Or: pip install llama-cpp-python\n"
                )
        else:
            print("  Install: pip install llama-cpp-python\n")

        print(f"Download {config.model.hf_file} anyway"
              f" ({size_hint})?")
        response = input("Download now? [y/N]: ").strip().lower()
        if response != "y":
            print(
                "No model downloaded."
                " Use --remote or install Ollama."
            )
            sys.exit(1)
    else:
        # Prompt user — llama-cpp-python is available
        print("\nNo local model found.")
        print(
            f"Download {config.model.hf_file}"
            f" from {config.model.hf_repo}?"
        )
        print(f"This is approximately {size_hint}.\n")

        response = input(
            "Download now? [Y/n]: "
        ).strip().lower()
        if response and response != "y":
            print(
                "No model available."
                " Use --model or --remote to specify one."
            )
            sys.exit(1)

    from huggingface_hub import hf_hub_download

    # Disable hf_xet — its native subprocess spawning triggers
    # "bad value(s) in fds_to_keep" on Python 3.14+ (forkserver default).
    prev_xet = os.environ.get("HF_HUB_DISABLE_XET")
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    try:
        print("Downloading from HuggingFace...")
        path = hf_hub_download(
            repo_id=config.model.hf_repo,
            filename=config.model.hf_file,
            local_dir=str(model_dir),
        )
        print(f"Model saved to: {path}")
        return path

    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)
    finally:
        if prev_xet is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = prev_xet


if __name__ == "__main__":
    main()
