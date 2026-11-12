Model configuration:
  Seven local model tiers:
    Light:          Qwen3.5-4B      (~2.7 GB, low RAM)
    Standard:       Qwen3.5-9B      (~5.7 GB, general purpose)
    Enhanced:       Mistral Nemo 12B (~7.5 GB, 128K context) — recommended
    Coder (MoE):    Qwen3.5-35B-A3B (~22 GB, needs 24+ GB RAM)
    Gemma Light:    Gemma 4 E2B     (~1.5 GB, 128K context)
    Gemma Standard: Gemma 4 E4B     (~5 GB, 128K context)
    Gemma Enhanced: Gemma 4 12B     (~7.1 GB, 128K context)

  Default: Qwen3.5-4B Q4_K_M GGUF, auto-downloaded on first run.
  Model storage: ~/.local/share/natshell/models/
  Download after setup: /model download <tier>
    (light, standard, enhanced, coder, gemma-light, gemma-standard, gemma-enhanced)

Local model config ([model] section in config.toml):
  path         — Path to .gguf file, or 'auto' for default download
  n_ctx        — Context window (0 = auto: 4096 for ≤4B, 32768 for 9B+, Mistral Nemo / Gemma 4)
  n_gpu_layers — GPU layers (-1 = all, 0 = CPU only)
  main_gpu     — GPU device index (-1 = auto-detect)

Remote/Ollama config:
  [remote] section: url, model, api_key
  [ollama] section: url, default_model
  CLI flags: --remote <url>, --remote-model <name>
  To install Ollama: curl -fsSL https://ollama.com/install.sh | sh
  To pull a model: ollama pull <model-name>