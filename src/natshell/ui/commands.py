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


class RemoteModelProvider(Provider):
    """Command palette provider for switching to remote models."""

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        # Use cached model list if available, otherwise fetch
        models = getattr(self.app, "_cached_remote_models", None)
        if models is None:
            try:
                from natshell.inference.ollama import list_models, normalize_base_url

                base_url = self.app._get_remote_base_url()
                if not base_url:
                    return
                models = await list_models(base_url)
                self.app._cached_remote_models = models
            except Exception:
                return

        try:
            current = self.app.agent.engine.engine_info().model_name
        except Exception:
            current = ""

        for m in models:
            if m.name == current:
                continue
            detail = ""
            if m.parameter_size:
                detail = f" [{m.parameter_size}]"
            elif m.size_gb:
                detail = f" ({m.size_gb} GB)"
            label = f"Switch to remote: {m.name}{detail}"
            score = matcher.match(label)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(label),
                    self._make_callback(m.name),
                    text=label,
                    help=f"Switch to {m.name} on remote server",
                )

    def _make_callback(self, model_name: str):
        """Create a callback that switches to the given remote model."""
        async def callback() -> None:
            from textual.containers import ScrollableContainer
            conversation = self.app.query_one("#conversation", ScrollableContainer)
            await self.app._model_use(model_name, conversation)
            conversation.scroll_end()
        return callback
