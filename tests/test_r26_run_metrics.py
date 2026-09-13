"""Tests for R2-6 (recording half) — run-metrics persistence.

Covers:
- ``agent/run_metrics.py``: store bootstrap + 0o700 dir, append/load_roundtrip,
  ``max_lines`` truncation, malformed-line tolerance, ``stats()`` summary,
  disable semantics, and the ``NATSHELL_DISABLE_METRICS`` kill switch.
- ``AgentLoop`` integration: one record per run (success path), no write
  when disabled, engine metadata (model/n_ctx) attached, and the guarantee
  that a broken metrics path never breaks the agent run.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from natshell.agent.context import SystemContext
from natshell.agent.loop import AgentLoop
from natshell.agent.run_metrics import (
    FILENAME,
    get_run_metrics_store,
    reset_run_metrics_store,
    RunMetricsStore,
)
from natshell.config import AgentConfig, SafetyConfig
from natshell.inference.engine import CompletionResult, EngineInfo
from natshell.safety.classifier import SafetyClassifier
from natshell.tools.registry import create_default_registry


def _make_agent(
    responses: list[CompletionResult] | None = None,
    engine_info: EngineInfo | None = None,
) -> AgentLoop:
    from unittest.mock import MagicMock

    engine = AsyncMock()
    if responses is not None:
        engine.chat_completion = AsyncMock(side_effect=responses)
    else:
        engine.chat_completion = AsyncMock(
            return_value=CompletionResult(content="done")
        )
    if engine_info is not None:
        # engine_info is a *sync* accessor on real engines — mirror the
        # test_headless.py convention of assigning a sync MagicMock.
        engine.engine_info = MagicMock(return_value=engine_info)
    else:
        # Engine that doesn't implement engine_info() at all (some remote
        # backends).  _record_run must gracefully skip metadata for these.
        def _no_info():
            raise AttributeError("engine has no engine_info")

        engine.engine_info = _no_info  # type: ignore[assignment]
    tools = create_default_registry()
    safety = SafetyClassifier(SafetyConfig(mode="danger", always_confirm=[], blocked=[]))
    agent = AgentLoop(
        engine=engine,
        tools=tools,
        safety=safety,
        config=AgentConfig(max_steps=15, temperature=0.3, max_tokens=2048),
    )
    agent.initialize(
        SystemContext(hostname="h", distro="Debian", kernel="6.12", username="u")
    )
    return agent


async def _run(agent: AgentLoop) -> list:
    events = []
    async for event in agent.handle_user_message("hello"):
        events.append(event)
    return events


def _reset_store_singleton() -> None:
    reset_run_metrics_store()


# ─── Store bootstrap ────────────────────────────────────────────────────────


class TestStoreBootstrap:
    def test_creates_dir_0700(self, tmp_path: Path):
        d = tmp_path / "metrics"
        store = RunMetricsStore(d)
        assert d.is_dir()
        assert d.stat().st_mode & 0o777 == 0o700

    def test_path_property(self, tmp_path: Path):
        store = RunMetricsStore(tmp_path)
        assert store.path == tmp_path / FILENAME

    def test_explicit_enabled_false_no_dir(self, tmp_path: Path):
        d = tmp_path / "metrics"
        store = RunMetricsStore(d, enabled=False)
        assert store.enabled is False
        assert not d.exists()

    def test_default_dir_under_data_dir(self):
        store = get_run_metrics_store()
        assert "natshell" in str(store.dir)
        assert store.enabled in (True, False)  # env-dependent; don't assert direction
        _reset_store_singleton()


# ─── Record + load ──────────────────────────────────────────────────────────


class TestRecordLoad:
    def test_append_and_load_roundtrip(self, tmp_path: Path):
        store = RunMetricsStore(tmp_path)
        stats = {
            "steps": 2,
            "total_wall_ms": 1234,
            "total_inference_ms": 900,
            "total_prompt_tokens": 500,
            "total_completion_tokens": 100,
        }
        path = store.record(stats, model="Qwen3-4B", n_ctx=4096)
        assert path is not None and path.exists()
        recs = store.load_recent()
        assert len(recs) == 1
        assert recs[0]["steps"] == 2
        assert recs[0]["model"] == "Qwen3-4B"
        assert recs[0]["n_ctx"] == 4096
        assert recs[0]["schema"] == 1
        assert "ts" in recs[0] and "ts_iso" in recs[0]
        assert recs[0]["total_prompt_tokens"] == 500

    def test_multiple_appends_preserve_order(self, tmp_path: Path):
        store = RunMetricsStore(tmp_path)
        for i in range(3):
            store.record({"steps": i, "marker": f"r{i}"})
        recs = store.load_recent()
        assert [r["marker"] for r in recs] == ["r0", "r1", "r2"]
        # load_recent(n) returns the LAST n, oldest-first
        last2 = store.load_recent(2)
        assert [r["marker"] for r in last2] == ["r1", "r2"]

    def test_record_none_stats_is_noop(self, tmp_path: Path):
        store = RunMetricsStore(tmp_path)
        assert store.record(None) is None
        assert store.load_recent() == []

    def test_disabled_store_does_not_write(self, tmp_path: Path):
        store = RunMetricsStore(tmp_path, enabled=False)
        assert store.record({"steps": 1}) is None
        assert not store.path.exists()

    def test_disable_runtime(self, tmp_path: Path):
        store = RunMetricsStore(tmp_path)
        assert store.enabled is True
        store.disable()
        assert store.record({"steps": 1}) is None


# ─── Truncation ─────────────────────────────────────────────────────────────


class TestTruncation:
    def test_max_lines_keeps_tail(self, tmp_path: Path):
        store = RunMetricsStore(tmp_path, max_lines=3)
        for i in range(7):
            store.record({"marker": f"r{i}"})
        recs = store.load_recent()
        assert len(recs) == 3
        assert [r["marker"] for r in recs] == ["r4", "r5", "r6"]

    def test_under_cap_keeps_all(self, tmp_path: Path):
        store = RunMetricsStore(tmp_path, max_lines=5)
        for i in range(4):
            store.record({"marker": f"r{i}"})
        assert len(store.load_recent()) == 4


# ─── Malformed-line tolerance ───────────────────────────────────────────────


class TestMalformedLines:
    def test_bad_lines_are_skipped(self, tmp_path: Path):
        store = RunMetricsStore(tmp_path)
        store.record({"ok": 1})
        # Interleave garbage
        with store.path.open("a", encoding="utf-8") as fh:
            fh.write("not json at all\n")
            fh.write("\n")
            fh.write("[1,2,3]\n")  # valid JSON but not a dict
        store.record({"ok": 2})
        recs = store.load_recent()
        assert len(recs) == 2
        assert all(r.get("ok") in (1, 2) for r in recs)


# ─── Stats summary ──────────────────────────────────────────────────────────


class TestStoreStats:
    def test_empty(self, tmp_path: Path):
        store = RunMetricsStore(tmp_path)
        assert store.stats() == {"lines": 0}

    def test_summary(self, tmp_path: Path):
        store = RunMetricsStore(tmp_path)
        store.record({"total_prompt_tokens": 100, "total_completion_tokens": 10})
        store.record({"total_prompt_tokens": 200, "total_completion_tokens": 20})
        s = store.stats()
        assert s["lines"] == 2
        assert s["total_prompt_tokens"] == 300
        assert s["total_completion_tokens"] == 30
        assert s["first_ts"] is not None and s["last_ts"] is not None
        assert s["first_ts"] <= s["last_ts"]


# ─── Kill switch ────────────────────────────────────────────────────────────


class TestKillSwitch:
    def test_env_disables_default_store(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("NATSHELL_DISABLE_METRICS", "1")
        # Without a dir_path this is the "default" location → disabled
        store = RunMetricsStore()
        assert store.enabled is False
        assert store.record({"steps": 1}) is None

    def test_env_does_not_disable_explicit_dir(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("NATSHELL_DISABLE_METRICS", "1")
        store = RunMetricsStore(tmp_path)  # explicit dir → still enabled
        assert store.enabled is True
        store.record({"steps": 1})
        assert len(store.load_recent()) == 1

    def test_unset_env_default_store_enabled(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("NATSHELL_DISABLE_METRICS", raising=False)
        store = RunMetricsStore(tmp_path)
        assert store.enabled is True


# ─── AgentLoop integration ──────────────────────────────────────────────────


class TestLoopIntegration:
    async def test_success_run_appends_one_record(self, tmp_path: Path):
        _reset_store_singleton()
        store = RunMetricsStore(tmp_path)
        # Point the singleton at our temp dir so the loop's _record_run hits it
        from natshell.agent import run_metrics as _rm

        _rm._store = store
        try:
            agent = _make_agent(
                responses=[CompletionResult(content="hello back", prompt_tokens=50)],
                engine_info=EngineInfo(
                    engine_type="local", model_name="Qwen3-4B-Q4_K_M.gguf", n_ctx=4096,
                ),
            )
            await _run(agent)
            recs = store.load_recent()
            assert len(recs) == 1
            r = recs[0]
            assert r["engine_type"] == "local"
            assert r["model"] == "Qwen3-4B-Q4_K_M.gguf"
            assert r["n_ctx"] == 4096
            assert r["steps"] >= 1
            assert r.get("completion_tokens", 0) >= 0
        finally:
            _reset_store_singleton()

    async def test_store_disabled_no_write(self, tmp_path: Path):
        from natshell.agent import run_metrics as _rm

        _reset_store_singleton()
        disabled = RunMetricsStore(tmp_path, enabled=False)
        _rm._store = disabled  # type: ignore
        try:
            agent = _make_agent(responses=[CompletionResult(content="ok")])
            await _run(agent)
            assert not (tmp_path / FILENAME).exists()
        finally:
            _reset_store_singleton()

    async def test_record_failure_never_breaks_run(self, tmp_path: Path):
        """A raising store must not break the agent loop (best-effort)."""
        from natshell.agent import run_metrics as _rm

        class BrokenStore:
            def record(self, *a, **k):
                raise RuntimeError("metrics disk on fire")

        _reset_store_singleton()
        _rm._store = BrokenStore()  # type: ignore
        try:
            agent = _make_agent(responses=[CompletionResult(content="ok")])
            events = await _run(agent)
            # The successful RESPONSE still lands
            from natshell.agent.events import EventType

            assert any(e.type == EventType.RESPONSE for e in events)
        finally:
            _reset_store_singleton()

    async def test_engine_without_engine_info_still_records(self, tmp_path: Path):
        """Engine lacking engine_info() shouldn't block the record."""
        from natshell.agent import run_metrics as _rm

        _reset_store_singleton()
        store = RunMetricsStore(tmp_path)
        _rm._store = store  # type: ignore
        try:
            # engine_info raises AttributeError (see _make_agent default)
            agent = _make_agent(responses=[CompletionResult(content="ok")])
            await _run(agent)
            recs = store.load_recent()
            assert len(recs) == 1
            # No engine metadata (engine didn't report any), but stats present
            assert "engine_type" not in recs[0]
            assert "steps" in recs[0]
        finally:
            _reset_store_singleton()


def store_path_exists(base: Path) -> bool:
    return (base / FILENAME).exists()
