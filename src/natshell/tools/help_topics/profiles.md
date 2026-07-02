Configuration profiles let you switch between different settings quickly.

Commands:
  /profile           — List available profiles
  /profile <name>    — Apply a profile

Defining profiles in config.toml:
  [profiles.coding]
  engine = "remote"        # Switch engine (local/remote)
  model = "qwen3:30b"     # Remote model name
  n_ctx = 32768            # Context window override
  temperature = 0.3        # Lower = more deterministic
  n_gpu_layers = -1        # GPU layers override

  [profiles.creative]
  engine = "local"
  temperature = 0.9

Profiles can override: engine, model, url, api_key, n_ctx, temperature, n_gpu_layers. Only specified fields are changed.