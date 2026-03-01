"""Tests for the FileReadTracker."""

from __future__ import annotations

from natshell.tools.file_tracker import get_tracker, reset_tracker


class TestFileReadTracker:
    def setup_method(self):
        reset_tracker()

    def teardown_method(self):
        reset_tracker()

    def test_never_read_allows_edit(self):
        """Files never read should be editable (agent may know content from other means)."""
        tracker = get_tracker()
        allowed, reason = tracker.can_edit("/tmp/never_read.txt")
        assert allowed is True
        assert reason == ""

    def test_partial_read_blocks_edit(self):
        """A truncated first read should block editing."""
        tracker = get_tracker()
        tracker.record_read("/tmp/test.txt", truncated=True)
        allowed, reason = tracker.can_edit("/tmp/test.txt")
        assert allowed is False
        assert "partially read" in reason

    def test_full_read_allows_edit(self):
        """A complete first read should allow editing."""
        tracker = get_tracker()
        tracker.record_read("/tmp/test.txt", truncated=False)
        allowed, reason = tracker.can_edit("/tmp/test.txt")
        assert allowed is True

    def test_partial_then_continuation_completes(self):
        """partial read + continuation(not truncated) should promote to full."""
        tracker = get_tracker()
        tracker.record_read("/tmp/test.txt", truncated=True)
        tracker.record_continuation("/tmp/test.txt", truncated=False)
        allowed, reason = tracker.can_edit("/tmp/test.txt")
        assert allowed is True

    def test_partial_continuation_still_partial(self):
        """partial read + continuation(still truncated) should remain blocked."""
        tracker = get_tracker()
        tracker.record_read("/tmp/test.txt", truncated=True)
        tracker.record_continuation("/tmp/test.txt", truncated=True)
        allowed, reason = tracker.can_edit("/tmp/test.txt")
        assert allowed is False

    def test_invalidate_after_edit(self):
        """Invalidating a file should remove tracking, making it editable again."""
        tracker = get_tracker()
        tracker.record_read("/tmp/test.txt", truncated=True)
        tracker.invalidate("/tmp/test.txt")
        allowed, reason = tracker.can_edit("/tmp/test.txt")
        assert allowed is True

    def test_clear_resets_all(self):
        """clear() should reset all tracking state."""
        tracker = get_tracker()
        tracker.record_read("/tmp/a.txt", truncated=True)
        tracker.record_read("/tmp/b.txt", truncated=True)
        tracker.clear()
        assert tracker.can_edit("/tmp/a.txt") == (True, "")
        assert tracker.can_edit("/tmp/b.txt") == (True, "")

    def test_path_resolution(self):
        """/tmp/./test.txt and /tmp/test.txt should resolve to the same key."""
        tracker = get_tracker()
        tracker.record_read("/tmp/./test.txt", truncated=True)
        allowed, reason = tracker.can_edit("/tmp/test.txt")
        assert allowed is False

    def test_full_read_not_downgraded_by_partial(self):
        """A full read should not be downgraded to partial by a subsequent truncated read."""
        tracker = get_tracker()
        tracker.record_read("/tmp/test.txt", truncated=False)
        tracker.record_read("/tmp/test.txt", truncated=True)
        allowed, reason = tracker.can_edit("/tmp/test.txt")
        assert allowed is True

    def test_singleton_is_shared(self):
        """get_tracker() should return the same instance."""
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2
