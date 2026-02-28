"""Command palette providers for NatShell."""

from __future__ import annotations

from pathlib import Path

from textual.command import Hit, Hits, Provider

MODELS_DIR = Path.home() / ".local" / "share" / "natshell" / "models"


class ModelSwitchProvider(Provider):
    """Command palette provider for switching between downloaded local models."""

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        if not MODELS_DIR.is_dir():
            return

        # Get the currently active model name so we can skip it
        try:
            current = self.app.agent.engine.engine_info().model_name
        except Exception:
            current = ""

        for gguf in sorted(MODELS_DIR.glob("*.gguf")):
            if gguf.name == current:
                continue
            label = f"Switch to {gguf.stem}"
            score = matcher.match(label)
            if score > 0:
                model_path = str(gguf)
                yield Hit(
                    score,
                    matcher.highlight(label),
                    self._make_callback(model_path),
                    text=label,
                    help=f"Load {gguf.name} as local model",
                )

    def _make_callback(self, model_path: str):
        """Create a callback that switches to the given model."""

        async def callback() -> None:
            await self.app.switch_local_model(model_path)

        return callback
