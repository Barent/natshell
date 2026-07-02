Common issues:
  'GPU offloading requested but not supported'
    → Reinstall llama-cpp-python with GPU flags:
      CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --no-binary llama-cpp-python --no-cache-dir

  'Remote server unreachable'
    → Check the URL in [ollama] or [remote] config section.
      Ensure the server is running (ollama serve, or check the API host).

  'No local model found'
    → Run: natshell --download  (downloads the default model)
    → Or use /model download <tier> from within NatShell

  Slow inference / high CPU
    → Check n_gpu_layers in config (set to -1 to offload all layers to GPU)
    → Set n_threads to match physical core count

  Self-update: natshell --update (git installs only)