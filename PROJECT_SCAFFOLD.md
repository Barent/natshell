# NatShell — Original Design Specification

> **Note**: This is the original architectural specification used to build NatShell.
> The actual implementation has evolved beyond this document. For current architecture
> and features, see **CLAUDE.md** and **README.md**. Key differences from this spec:
>
> - **Cross-platform**: Now supports Linux, macOS, and WSL (not just Debian)
> - **Ollama integration**: New `inference/ollama.py` module for server discovery and model listing
> - **GPU support**: New `gpu.py` module for GPU detection and best-device selection
> - **Runtime model switching**: `/model` slash commands allow switching engines without restart
> - **Clipboard integration**: New `ui/clipboard.py` with macOS/WSL/Wayland/X11/OSC52 support
> - **Command palette**: New `ui/commands.py` for Ctrl+P model switching
> - **Platform detection**: New `platform.py` module with cached `is_macos()`/`is_wsl()`/`is_linux()`
> - **Security hardening**: Markup escaping, command chaining detection, env var filtering,
>   sudo password timeout, sensitive file path gating, HTTPS warnings, config permissions checks
> - **Tool calling**: Uses plain-text tool injection + `<tool_call>` XML parsing (not llama-cpp-python's built-in tool format)
> - **Context auto-sizing**: `n_ctx = 0` auto-detects from model parameter count in filename
> - **Config persistence**: `save_ollama_default()` and `save_model_config()` for TOML editing
> - **Self-update**: `--update` flag for git-based installations

## Project Overview

NatShell is a self-contained, local-first agentic TUI that lets users interact with their system through natural language. It bundles a small quantized LLM (via llama.cpp) and uses a ReAct-style agent loop to plan, execute, and iterate on shell commands to fulfill user requests.

---

## Target Environment

- **OS:** Linux, macOS, and WSL
- **Python:** 3.11+
- **Inference:** llama-cpp-python (bundled), with optional Ollama or OpenAI-compatible API fallback
- **Default Model:** Qwen3-4B-Q4_K_M.gguf (~2.5GB)
- **TUI:** Textual

---

## Project Structure

```
natshell/
├── pyproject.toml              # Project metadata, dependencies, entry point
├── README.md
├── CLAUDE.md                   # Claude Code build instructions
├── config.default.toml         # Default configuration with safety patterns
├── install.sh                  # Cross-platform installer
├── src/
│   └── natshell/
│       ├── __init__.py
│       ├── __main__.py         # Entry point: CLI args, model download, engine wiring
│       ├── app.py              # Textual TUI with slash commands and model switching
│       ├── config.py           # TOML config loading, env var support, persistence
│       ├── gpu.py              # GPU detection (vulkaninfo/nvidia-smi/lspci)
│       ├── platform.py         # Platform detection (Linux/macOS/WSL)
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── loop.py         # ReAct agent loop with safety, sudo, fallback
│       │   ├── system_prompt.py # Platform-aware system prompt builder
│       │   └── context.py      # System context gathering (per-platform)
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── engine.py       # Protocol types (CompletionResult, ToolCall, EngineInfo)
│       │   ├── local.py        # llama-cpp-python with GPU, auto ctx, XML tool parsing
│       │   ├── remote.py       # OpenAI-compatible API backend (httpx)
│       │   └── ollama.py       # Ollama server ping, model listing, URL normalization
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── registry.py     # Tool registration and dispatch
│       │   ├── execute_shell.py # Shell exec with sudo, env filtering, truncation
│       │   ├── read_file.py
│       │   ├── write_file.py
│       │   ├── list_directory.py
│       │   └── search_files.py
│       ├── safety/
│       │   ├── __init__.py
│       │   └── classifier.py   # Regex classifier with chaining + path sensitivity
│       └── ui/
│           ├── __init__.py
│           ├── widgets.py      # Widgets with Rich markup escaping
│           ├── commands.py     # Command palette (model switching)
│           ├── clipboard.py    # Cross-platform clipboard
│           └── styles.tcss     # Textual CSS stylesheet
└── tests/
    ├── test_agent.py
    ├── test_safety.py
    ├── test_tools.py
    ├── test_clipboard.py
    ├── test_platform.py
    ├── test_engine_swap.py
    ├── test_ollama.py
    ├── test_ollama_config.py
    └── test_slash_commands.py
```

---

## Dependencies (pyproject.toml)

```toml
[project]
name = "natshell"
version = "0.1.0"
description = "Natural language shell interface for Linux"
requires-python = ">=3.11"
dependencies = [
    "llama-cpp-python>=0.3.0",
    "textual>=1.0.0",
    "tomli>=2.0.0;python_version<'3.11'",
    "tomllib-stubs>=0.1.0",
    "rich>=13.0.0",
    "httpx>=0.27.0",        # For remote API backend
    "huggingface-hub>=0.24", # For model download
]

[project.scripts]
natshell = "natshell.__main__:main"

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "ruff"]
```

---

## File-by-File Specification

### 1. `src/natshell/__main__.py` — Entry Point

```
Responsibilities:
- Parse CLI arguments (--config, --model, --remote-url, --download-model)
- Load configuration
- Download model if not present (huggingface_hub.hf_hub_download)
- Initialize inference engine (local or remote)
- Launch the Textual app

CLI Interface:
  natshell                          # Launch with defaults
  natshell --model ./my-model.gguf  # Use specific model
  natshell --remote http://host:11434/v1  # Use Ollama/remote
  natshell --download                # Download default model and exit
  natshell --config ~/.config/natshell/config.toml

Model Auto-Download:
  On first run, if no model exists at the configured path:
  - Prompt user: "No model found. Download Qwen3-4B-Q4_K_M (~2.5GB)? [Y/n]"
  - Use huggingface_hub.hf_hub_download() to fetch it
  - Save to ~/.local/share/natshell/models/
```

### 2. `config.default.toml` — Configuration

```toml
[model]
# Path to GGUF model file. "auto" triggers download of default model.
path = "auto"
# HuggingFace repo for auto-download
hf_repo = "Qwen/Qwen3-4B-GGUF"
hf_file = "qwen3-4b-q4_k_m.gguf"
# Context window size (tokens)
n_ctx = 8192
# Number of CPU threads (0 = auto-detect)
n_threads = 0
# GPU layers to offload (0 = CPU only, -1 = all)
n_gpu_layers = 0

[remote]
# Set to enable remote inference instead of local model
# url = "http://localhost:11434/v1"
# model = "qwen3:4b"
# api_key = ""

[agent]
# Maximum tool calls per request before forcing a response
max_steps = 15
# Temperature for generation
temperature = 0.3
# Max tokens per generation
max_tokens = 2048

[safety]
# "confirm" = ask before moderate/dangerous commands
# "warn" = show warning but auto-execute
# "yolo" = execute everything without asking
mode = "confirm"
# Commands that always require confirmation (regex patterns)
always_confirm = [
    "^rm\\s",
    "^dd\\s",
    "^mkfs",
    "^shutdown",
    "^reboot",
    "^systemctl\\s+(stop|disable|mask|restart)",
    "^chmod\\s+777",
    "^chown",
    "\\|\\s*tee\\s",
    ">\\s*/etc/",
    "^kill\\s+-9",
    "^wipefs",
    "^fdisk",
    "^parted",
]
# Commands that are blocked entirely
blocked = [
    ":(){ :|:& };:",    # Fork bomb patterns
    "^rm\\s+-rf\\s+/\\s*$",
]

[ui]
# Theme: "dark" or "light"
theme = "dark"
# Show command output inline or in separate panel
output_mode = "inline"
```

### 3. `src/natshell/agent/context.py` — System Context Gathering

```
Responsibilities:
  Collect system information ONCE at startup, refresh on-demand.
  This context is injected into every system prompt so the model
  knows what system it's operating on.

Functions:

  async def gather_system_context() -> SystemContext:
      """Gather all system info, return structured dataclass."""

  @dataclass
  class SystemContext:
      hostname: str
      distro: str           # e.g. "Debian GNU/Linux 13 (trixie)"
      kernel: str           # e.g. "6.12.x-amd64"
      arch: str             # e.g. "x86_64"
      cpu: str              # e.g. "AMD Ryzen AI 9 HX 370"
      ram_total_gb: float
      ram_available_gb: float
      username: str
      is_root: bool
      has_sudo: bool
      shell: str            # e.g. "/bin/bash"
      package_manager: str  # "apt", "dnf", "pacman", etc.
      cwd: str
      disk_usage: list[DiskInfo]   # mount, total, used, free
      network_interfaces: list[NetInfo]  # name, ip, subnet
      default_gateway: str
      installed_tools: dict[str, bool]  # docker, git, nmap, curl, etc.
      running_services: list[str]       # active systemd services (top 20)
      containers: list[str]             # docker ps --format if docker available

Gathering Methods (all via subprocess, non-blocking):
  - distro:        cat /etc/os-release | grep PRETTY_NAME
  - kernel:        uname -r
  - cpu:           lscpu | grep "Model name"
  - ram:           free -b | grep Mem
  - sudo:          sudo -n true 2>/dev/null (exit code check)
  - pkg manager:   which apt || which dnf || which pacman || ...
  - disk:          df -h --output=target,size,used,avail -x tmpfs -x devtmpfs
  - network:       ip -4 -j addr show
  - gateway:       ip -4 route show default
  - tools:         which <tool> for each in checklist
  - services:      systemctl list-units --type=service --state=running --no-pager -q
  - containers:    docker ps --format '{{.Names}} ({{.Image}})' 2>/dev/null

Format as compact text block for system prompt injection.
```

### 4. `src/natshell/agent/system_prompt.py` — System Prompt Builder

```
Responsibilities:
  Build the system prompt that defines the agent's behavior,
  available tools, and system context.

The system prompt has these sections:

ROLE:
  You are NatShell, a Linux system administration assistant running
  directly on the user's machine. You help users accomplish tasks
  by executing shell commands and analyzing results. You are running
  on their actual system — commands you execute are real.

BEHAVIOR RULES:
  - Always plan before acting. State what you intend to do.
  - Execute commands one at a time, observe results, then decide next step.
  - If a command fails, analyze the error and try an alternative approach.
  - When the task is complete, summarize what was done and the results.
  - Never guess at system state — always check first.
  - Prefer non-destructive commands. Use --dry-run flags when available.
  - For package installs, check if the package exists first.
  - If you need elevated privileges, explain why and use sudo.
  - If a task seems risky, warn the user before proceeding.
  - Keep command output analysis concise — highlight what matters.
  - You can execute multiple commands in sequence to accomplish complex goals.
  - When presenting results, format them clearly for the terminal.

TOOL DEFINITIONS:
  (Injected from tool registry — see tools section)

SYSTEM CONTEXT:
  (Injected from context.py output)

  Example:
  <system_info>
  Host: debbythekeeper | Debian GNU/Linux 13 (trixie) | 6.12.6-amd64 | x86_64
  CPU: 12th Gen Intel i5-12600 | RAM: 64.0GB total, 41.2GB available
  User: nicholas (sudo: yes) | Shell: /bin/bash | Pkg: apt
  Disks: / 458G (32% used), /home 916G (61% used)
  Network: eth0 192.168.1.50/24 | tailscale0 100.64.x.x/32
  Gateway: 192.168.1.1
  Tools: docker✓ git✓ nmap✓ curl✓ wget✓ ssh✓ python3✓ node✗ go✗
  Containers: gitea (gitea/gitea:latest), jellyfin (jellyfin/jellyfin)
  Services: docker, sshd, pihole-FTL, tailscaled, netdata
  </system_info>
```

### 5. `src/natshell/tools/registry.py` — Tool Registration

```
Responsibilities:
  Define the tool interface, register all tools, dispatch calls,
  and generate tool definitions for the system prompt.

Key Types:

  @dataclass
  class ToolDefinition:
      name: str
      description: str
      parameters: dict          # JSON Schema
      requires_confirmation: bool  # default False

  @dataclass  
  class ToolResult:
      output: str
      error: str
      exit_code: int
      truncated: bool          # if output was too long and got cut

  class ToolRegistry:
      def register(self, name, handler, definition): ...
      def get_definitions_for_prompt(self) -> str: ...
      async def execute(self, name, arguments) -> ToolResult: ...

Prompt Format (Hermes-style for Qwen3):
  Tools are formatted as JSON function definitions compatible with
  the Qwen3/Hermes tool calling template. The llama-cpp-python
  chat handler will format these into the model's expected template.

  tools = [
      {
          "type": "function",
          "function": {
              "name": "execute_shell",
              "description": "Execute a shell command...",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "command": {"type": "string", "description": "..."},
                      "timeout": {"type": "integer", "description": "..."}
                  },
                  "required": ["command"]
              }
          }
      },
      ...
  ]
```

### 6. `src/natshell/tools/execute_shell.py` — Primary Tool

```
Responsibilities:
  Execute shell commands safely and return structured results.

async def execute_shell(command: str, timeout: int = 30,
                        workdir: str | None = None) -> ToolResult:
    """
    Execute a shell command via subprocess.

    Args:
        command: The shell command to execute (passed to bash -c)
        timeout: Max seconds to wait (default 30, max 300)
        workdir: Working directory (default: user's cwd)

    Returns:
        ToolResult with stdout, stderr, exit_code

    Behavior:
        - Runs via: subprocess.run(["bash", "-c", command], ...)
        - Captures stdout and stderr separately
        - Enforces timeout (kills process on timeout)
        - Truncates output to ~4000 chars with note if longer
        - Sets LC_ALL=C for consistent output parsing
        - Inherits user's PATH and environment
        - Does NOT run in a sandbox — this is intentional
    """

Output Truncation:
  If stdout or stderr exceeds 4000 characters:
  - Keep first 2000 chars
  - Insert "\n... [truncated {N} lines] ...\n"
  - Keep last 1500 chars
  This ensures the model sees both the beginning and end of long output.

Tool Definition for prompt:
  name: execute_shell
  description: >
    Execute a shell command on the user's Linux system and return
    the output. Use this to run any CLI command, install packages,
    check system state, modify files, manage services, etc.
    The command runs as the current user via bash. Use sudo when
    elevated privileges are needed. Prefer single commands; for
    multi-step operations, call this tool multiple times.
  parameters:
    command: (required) The bash command to execute
    timeout: (optional) Timeout in seconds, default 30, max 300
```

### 7. `src/natshell/tools/read_file.py`

```
async def read_file(path: str, max_lines: int = 200) -> ToolResult:
    """
    Read contents of a file. Returns ToolResult with file content
    as output. Truncates to max_lines from the start.
    
    Useful for the model to inspect config files, logs, code, etc.
    without constructing cat/head commands.
    """
```

### 8. `src/natshell/tools/write_file.py`

```
async def write_file(path: str, content: str, 
                     mode: str = "overwrite") -> ToolResult:
    """
    Write content to a file. mode: "overwrite" or "append".
    
    Always requires confirmation (marked in tool definition).
    Creates parent directories if needed.
    """
```

### 9. `src/natshell/tools/list_directory.py`

```
async def list_directory(path: str = ".", 
                         show_hidden: bool = False,
                         max_entries: int = 100) -> ToolResult:
    """
    List directory contents with file sizes and types.
    Returns formatted listing. More structured than raw ls output.
    """
```

### 10. `src/natshell/tools/search_files.py`

```
async def search_files(pattern: str, path: str = ".",
                       file_pattern: str = "*",
                       max_results: int = 50) -> ToolResult:
    """
    Search for text in files (grep -rn wrapper) or find files
    by name (find wrapper). Distinguishes based on whether
    'pattern' looks like a glob or a text search.
    """
```

### 11. `src/natshell/inference/engine.py` — Inference Abstraction

```
Responsibilities:
  Abstract interface so the agent loop doesn't care whether
  inference is local (llama.cpp) or remote (API).

class InferenceEngine(Protocol):
    async def chat_completion(
        self,
        messages: list[dict],       # OpenAI-format messages
        tools: list[dict] | None,   # Tool definitions
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> CompletionResult: ...

@dataclass
class CompletionResult:
    content: str | None           # Text response (if any)
    tool_calls: list[ToolCall]    # Tool calls (if any)
    finish_reason: str            # "stop", "tool_calls", "length"

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict               # Parsed JSON arguments
```

### 12. `src/natshell/inference/local.py` — Local llama.cpp Backend

```
Responsibilities:
  Manage the llama-cpp-python Llama instance and translate
  between our interface and llama-cpp-python's API.

class LocalEngine(InferenceEngine):
    def __init__(self, model_path, n_ctx, n_threads, n_gpu_layers):
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads or os.cpu_count(),
            n_gpu_layers=n_gpu_layers,
            chat_format="chatml-function-calling",  # Hermes/Qwen format
            verbose=False,
        )

    async def chat_completion(self, messages, tools, **kwargs):
        # llama-cpp-python supports tool calling natively
        # via create_chat_completion with tools parameter
        response = self.llm.create_chat_completion(
            messages=messages,
            tools=tools,
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 2048),
        )
        # Parse response into CompletionResult
        # Handle both text responses and tool_calls
        ...

IMPORTANT NOTES:
  - llama-cpp-python's create_chat_completion() is synchronous.
    Wrap in asyncio.to_thread() so it doesn't block the TUI.
  - The chat_format="chatml-function-calling" enables Hermes-style
    tool calling which Qwen3 models support natively.
  - Token streaming: use create_chat_completion(stream=True) and
    yield chunks to the TUI for real-time output display.
```

### 13. `src/natshell/inference/remote.py` — Remote API Backend

```
Responsibilities:
  Connect to any OpenAI-compatible API (Ollama, vLLM, LM Studio, etc.)

class RemoteEngine(InferenceEngine):
    def __init__(self, base_url, model, api_key=""):
        self.client = httpx.AsyncClient(base_url=base_url)
        self.model = model
        self.api_key = api_key

    async def chat_completion(self, messages, tools, **kwargs):
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": False,
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        response = await self.client.post(
            "/chat/completions", json=payload, headers=headers
        )
        # Parse OpenAI-format response into CompletionResult
        ...
```

### 14. `src/natshell/agent/loop.py` — Core Agent Loop

```
Responsibilities:
  The ReAct agent loop. This is the brain of the application.

class AgentLoop:
    def __init__(self, engine: InferenceEngine, tools: ToolRegistry,
                 safety: SafetyClassifier, config: AgentConfig):
        self.engine = engine
        self.tools = tools
        self.safety = safety
        self.config = config
        self.messages: list[dict] = []  # Conversation history

    async def initialize(self, system_context: SystemContext):
        """Build system prompt and set initial messages."""
        system_prompt = build_system_prompt(
            system_context,
            self.tools.get_definitions_for_prompt()
        )
        self.messages = [{"role": "system", "content": system_prompt}]

    async def handle_user_message(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """
        Process a user message through the full agent loop.
        Yields events for the TUI to render.
        """
        self.messages.append({"role": "user", "content": user_input})
        
        for step in range(self.config.max_steps):
            # Get model response
            yield AgentEvent(type="thinking")
            
            result = await self.engine.chat_completion(
                messages=self.messages,
                tools=self.tools.get_tool_schemas(),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            # Case 1: Model wants to call a tool
            if result.tool_calls:
                for tool_call in result.tool_calls:
                    # Safety check
                    risk = self.safety.classify(tool_call)
                    
                    if risk == Risk.BLOCKED:
                        yield AgentEvent(type="blocked", data=tool_call)
                        # Tell model the command was blocked
                        self._append_tool_result(tool_call, "BLOCKED: dangerous")
                        continue
                    
                    if risk == Risk.CONFIRM:
                        yield AgentEvent(type="confirm_needed", data=tool_call)
                        # TUI will await user confirmation
                        confirmed = yield  # Coroutine receives confirmation
                        if not confirmed:
                            self._append_tool_result(tool_call, "User declined")
                            continue
                    
                    # Execute the tool
                    yield AgentEvent(type="executing", data=tool_call)
                    tool_result = await self.tools.execute(
                        tool_call.name, tool_call.arguments
                    )
                    yield AgentEvent(type="tool_result", data=tool_result)
                    
                    # Append to conversation for next iteration
                    self._append_tool_result(tool_call, tool_result)
                
                continue  # Loop back for model to process results
            
            # Case 2: Model responded with text (task complete or needs info)
            if result.content:
                self.messages.append({
                    "role": "assistant", 
                    "content": result.content
                })
                yield AgentEvent(type="response", data=result.content)
                break  # Done
        
        else:
            yield AgentEvent(type="response", 
                           data="Reached maximum steps. Stopping here.")

    def _append_tool_result(self, tool_call, result):
        """Add assistant tool call + tool result to message history."""
        # Append the assistant's tool call message
        self.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments)
                }
            }]
        })
        # Append the tool result
        output = result if isinstance(result, str) else (
            f"Exit code: {result.exit_code}\n"
            f"stdout:\n{result.output}\n"
            f"stderr:\n{result.error}" if result.error else
            f"Exit code: {result.exit_code}\n{result.output}"
        )
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": output
        })

Event Types for TUI:
  - "thinking"       -> Show spinner/indicator
  - "executing"      -> Show the command being run
  - "tool_result"    -> Show command output
  - "confirm_needed" -> Show confirmation dialog
  - "blocked"        -> Show blocked command warning  
  - "response"       -> Show model's text response
  - "error"          -> Show error message
```

### 15. `src/natshell/safety/classifier.py` — Command Safety

```
Responsibilities:
  Classify commands by risk level before execution.

class Risk(Enum):
    SAFE = "safe"           # Auto-execute (ls, cat, df, grep, etc.)
    CONFIRM = "confirm"     # Show confirmation prompt
    BLOCKED = "blocked"     # Never execute

class SafetyClassifier:
    def __init__(self, config):
        self.always_confirm = [re.compile(p) for p in config.always_confirm]
        self.blocked = [re.compile(p) for p in config.blocked]
    
    def classify(self, tool_call: ToolCall) -> Risk:
        if tool_call.name != "execute_shell":
            # read_file, list_directory, search_files are always safe
            # write_file always requires confirmation
            return Risk.CONFIRM if tool_call.name == "write_file" else Risk.SAFE
        
        command = tool_call.arguments.get("command", "")
        
        # Check blocked patterns first
        for pattern in self.blocked:
            if pattern.search(command):
                return Risk.BLOCKED
        
        # Check confirmation patterns
        for pattern in self.always_confirm:
            if pattern.search(command):
                return Risk.CONFIRM
        
        # Heuristic checks for things not in config
        if "sudo" in command:
            return Risk.CONFIRM
        
        return Risk.SAFE

Note: The safety classifier is intentionally pattern-based, not
LLM-based. It must be fast, deterministic, and not dependent on
model intelligence. The config.toml patterns are user-customizable.
```

### 16. `src/natshell/app.py` — Textual TUI Application

```
Responsibilities:
  The main TUI application. Manages layout, user input,
  and rendering of agent events.

Layout:
  ┌──────────────────────────────────────────────────┐
  │  NatShell v0.1.0 | debian 13 | nicholas@debby    │  <- Header
  ├──────────────────────────────────────────────────┤
  │                                                    │
  │  You: scan my local network for computers          │
  │                                                    │
  │  NatShell: I'll scan your local network. First     │
  │  let me check your subnet and available tools.     │
  │                                                    │
  │  ┌─ execute_shell ────────────────────────────┐   │
  │  │ $ ip -4 route show default                  │   │
  │  │ default via 192.168.1.1 dev eth0            │   │
  │  └────────────────────────────────────────────┘   │
  │                                                    │
  │  ┌─ execute_shell ────────────────────────────┐   │
  │  │ $ nmap -sn 192.168.1.0/24                   │   │
  │  │ Starting Nmap 7.94 ...                      │   │
  │  │ Nmap scan report for router (192.168.1.1)   │   │
  │  │ Host is up (0.001s latency).                │   │
  │  │ Nmap scan report for debby (192.168.1.50)   │   │
  │  │ Host is up (0.0001s latency).               │   │  <- Scrollable
  │  │ ...                                         │   │     output area
  │  └────────────────────────────────────────────┘   │
  │                                                    │
  │  NatShell: I found 8 devices on your network:      │
  │                                                    │
  │  192.168.1.1   - router (gateway)                  │
  │  192.168.1.50  - debbythekeeper (this machine)     │
  │  192.168.1.51  - bazzite-old-asus                  │
  │  ...                                               │
  │                                                    │
  ├──────────────────────────────────────────────────┤
  │ > _                                                │  <- Input area
  └──────────────────────────────────────────────────┘

Textual App Structure:

  class NatShellApp(App):
      CSS_PATH = "ui/styles.tcss"
      BINDINGS = [
          ("ctrl+c", "quit", "Quit"),
          ("ctrl+l", "clear", "Clear"),
          ("ctrl+r", "refresh_context", "Refresh system info"),
      ]
      
      def compose(self) -> ComposeResult:
          yield Header(show_clock=True)
          yield ScrollableContainer(id="conversation")
          yield Input(placeholder="Ask me anything about your system...",
                     id="user-input")
          yield Footer()
      
      async def on_input_submitted(self, event):
          user_text = event.value
          self.query_one("#user-input").clear()
          
          # Add user message to conversation
          self.add_user_message(user_text)
          
          # Run agent loop, rendering events as they come
          async for event in self.agent.handle_user_message(user_text):
              match event.type:
                  case "thinking":
                      self.show_thinking_indicator()
                  case "executing":
                      self.add_command_block(event.data)
                  case "tool_result":
                      self.add_result_block(event.data)
                  case "confirm_needed":
                      confirmed = await self.show_confirmation(event.data)
                      # Send confirmation back to agent
                      ...
                  case "response":
                      self.add_assistant_message(event.data)

Custom Widgets:
  - UserMessage:     Right-aligned, distinct color
  - AssistantMessage: Left-aligned, supports markdown-ish formatting
  - CommandBlock:    Bordered box showing $ command and output
  - ConfirmDialog:   Modal overlay for dangerous command confirmation
  - ThinkingSpinner: Animated dots while model is generating
```

### 17. `src/natshell/ui/styles.tcss` — Textual Stylesheet

```
Target aesthetic:
  - Dark background (#1a1a2e or terminal default)
  - Command blocks with subtle borders (#30304a)
  - User messages in one color (cyan/blue)
  - Assistant text in default/white
  - Command prompts in green ($ prefix)
  - Errors/warnings in yellow/red
  - Confirmation dialogs with clear yes/no buttons
```

---

## Agent Loop Walkthrough — "scan my local network"

```
Step 1: User submits "Can you scan my local network for computers?"
        -> Added to messages as user role

Step 2: Model receives system prompt (which includes network info:
        "eth0 192.168.1.50/24, gateway 192.168.1.1") and thinks:
        "I can see the user is on 192.168.1.0/24. Let me check if
         nmap is available and run a ping scan."
        -> Model emits tool_call: execute_shell(command="which nmap")
        
Step 3: Tool returns: "/usr/bin/nmap" (exit 0)
        -> Appended to messages as tool result

Step 4: Model decides nmap exists, runs the scan:
        -> tool_call: execute_shell(
             command="nmap -sn 192.168.1.0/24",
             timeout=60
           )

Step 5: Tool returns nmap output with discovered hosts
        -> Appended to messages

Step 6: Model analyzes output and generates final response:
        "I found 8 active devices on your 192.168.1.0/24 network:
         ..."
        -> yield AgentEvent(type="response", data=formatted_text)

Step 7: Agent loop ends, TUI displays the response.
```

---

## Build Order (Recommended)

Build these in order so you can test incrementally:

1. **config.py** — Load TOML config, merge defaults
2. **tools/** — Implement all 5 tools with standalone tests
3. **safety/classifier.py** — Pattern-based classifier with tests
4. **inference/local.py** — Get llama-cpp-python loading a model and generating
5. **inference/engine.py + remote.py** — Abstraction layer
6. **agent/context.py** — System info gathering
7. **agent/system_prompt.py** — Prompt builder
8. **agent/loop.py** — The ReAct loop (test headless first, print to stdout)
9. **app.py + ui/** — Textual TUI wrapping the agent
10. **__main__.py** — CLI entry point, model download, wiring everything together

---

## Testing Strategy

```
tests/test_tools.py:
  - Test execute_shell with simple commands (echo, ls)
  - Test output truncation with large output
  - Test timeout behavior
  - Test read_file with existing and missing files
  - Test write_file creates directories
  - Test search_files with grep and find modes

tests/test_safety.py:
  - Test blocked patterns catch fork bombs, rm -rf /
  - Test confirm patterns catch rm, dd, sudo, etc.
  - Test safe commands pass through (ls, cat, df)
  - Test custom patterns from config

tests/test_agent.py:
  - Mock InferenceEngine to return scripted responses
  - Test single-tool-call cycle
  - Test multi-step loop with tool calls
  - Test max_steps cutoff
  - Test blocked command handling
  - Test confirmation flow
```

---

## Model Download Helper

```python
# In __main__.py or a setup module
from huggingface_hub import hf_hub_download

def download_default_model(config) -> str:
    """Download the default model, return path."""
    model_dir = Path.home() / ".local" / "share" / "natshell" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    target = model_dir / config.model.hf_file
    if target.exists():
        return str(target)
    
    print(f"Downloading {config.model.hf_file} from {config.model.hf_repo}...")
    path = hf_hub_download(
        repo_id=config.model.hf_repo,
        filename=config.model.hf_file,
        local_dir=str(model_dir),
    )
    return path
```

---

## Future Enhancements (Post-MVP)

- **Conversation persistence** — Save/load chat history
- **Custom tool plugins** — Let users add their own tools via Python files in ~/.config/natshell/tools/
- **Model fine-tuning recipe** — LoRA fine-tune on NL2Bash + sysadmin Q&A datasets
- **Streaming output** — Real-time token display as model generates
- **Tab completion** — Common request suggestions
- **Session context** — Remember cwd changes, installed packages across turns
- **Multi-machine** — Use SSH tool to manage remote hosts via Tailscale
- **Undo tracking** — Log all changes made, offer rollback

### Implemented Since Original Spec

- ~~GPU offloading~~ — Implemented: auto-detection, multi-GPU selection, Vulkan/Metal support
- ~~Remote API fallback~~ — Implemented: Ollama integration, runtime model switching, engine fallback
- ~~Cross-platform~~ — Implemented: macOS + WSL support in installer, context, clipboard, safety patterns
- ~~Clipboard integration~~ — Implemented: macOS/WSL/Wayland/X11/OSC52 with copy buttons on all messages
- ~~Security hardening~~ — Implemented: 9-point audit covering markup injection, env filtering, sudo timeout, etc.
