"""Tests for step-aware shell output truncation.

configure_step_scaling applies a scale factor to a baseline set by
configure_limits.  It used to apply that factor *through* configure_limits,
which reassigned the baseline to the just-scaled value, so the factor
compounded on every step.  These tests pin the intended linear decay and the
invariant that truncation never returns more text than it was given.
"""

from __future__ import annotations

import pytest

from natshell.tools import execute_shell as es


@pytest.fixture(autouse=True)
def _restore_limits():
    es.reset_limits()
    yield
    es.reset_limits()


class TestBaselineStability:
    def test_baseline_is_not_moved_by_step_scaling(self):
        """The scaling baseline must survive a full run of steps."""
        es.configure_limits(4000)
        for step in range(16):
            es.configure_step_scaling(step, 15)
        assert es._base_max_output_chars == 4000

    def test_scaling_is_idempotent_for_a_given_step(self):
        es.configure_limits(4000)
        es.configure_step_scaling(7, 15)
        once = es._max_output_chars
        for _ in range(5):
            es.configure_step_scaling(7, 15)
        assert es._max_output_chars == once

    def test_repeated_runs_start_from_the_same_budget(self):
        es.configure_limits(4000)
        for _ in range(3):
            for step in range(16):
                es.configure_step_scaling(step, 15)
        es.configure_step_scaling(0, 15)
        assert es._max_output_chars == 4000


class TestLinearDecay:
    def test_decays_to_thirty_percent_not_to_zero(self):
        """Documented behaviour: 1.0 at step 0 down to 0.3 at the last step."""
        es.configure_limits(4000)
        es.configure_step_scaling(0, 15)
        assert es._max_output_chars == 4000
        es.configure_step_scaling(15, 15)
        assert es._max_output_chars == pytest.approx(1200, abs=1)

    def test_decay_is_monotonic(self):
        es.configure_limits(4000)
        seen = []
        for step in range(16):
            es.configure_step_scaling(step, 15)
            seen.append(es._max_output_chars)
        assert seen == sorted(seen, reverse=True)
        assert min(seen) >= 1200

    def test_large_context_tier_scales_proportionally(self):
        es.configure_limits(64000)
        es.configure_step_scaling(15, 15)
        assert es._max_output_chars == pytest.approx(19200, abs=2)

    def test_zero_max_steps_is_a_no_op(self):
        es.configure_limits(4000)
        es.configure_step_scaling(5, 0)
        assert es._max_output_chars == 4000


class TestTruncationNeverGrows:
    def test_truncation_shrinks_output_at_the_final_step(self):
        """The regression: at tail 0, text[-0:] returned the entire string."""
        es.configure_limits(4000)
        for step in range(16):
            es.configure_step_scaling(step, 15)

        text = "X" * 50_000
        out, truncated = es._truncate_output(text)
        assert truncated is True
        assert len(out) < len(text)

    @pytest.mark.parametrize("step", range(16))
    def test_truncated_output_never_exceeds_input(self, step):
        es.configure_limits(4000)
        es.configure_step_scaling(step, 15)
        text = "line\n" * 10_000
        out, _ = es._truncate_output(text)
        assert len(out) <= len(text)

    def test_tail_never_reaches_zero(self):
        es.configure_limits(4000)
        for step in range(16):
            es.configure_step_scaling(step, 15)
            assert es._tail_chars > 0

    def test_absurdly_small_limit_is_floored(self):
        es.configure_limits(1)
        assert es._max_output_chars >= es._MIN_OUTPUT_CHARS
        assert es._tail_chars > 0
        out, _ = es._truncate_output("Y" * 10_000)
        assert len(out) < 10_000

    def test_short_output_is_untouched(self):
        es.configure_limits(4000)
        text = "short output"
        out, truncated = es._truncate_output(text)
        assert out == text
        assert truncated is False
