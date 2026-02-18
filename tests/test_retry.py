"""Unit tests for the retry mechanism in AIBackendBase."""

from unittest.mock import MagicMock, patch

from audio2sub.ai import AIBackendBase, GeminiMixin, OpenAIMixin
from audio2sub.common import Usage


class _DummyBackend(AIBackendBase):
    """Minimal concrete subclass for testing _retry."""

    name = "dummy"

    def _create_client(self):
        return None


def test_retry_succeeds_first_try():
    backend = _DummyBackend()
    fn = MagicMock(return_value="ok")
    assert backend._retry(fn, retries=3) == "ok"
    assert fn.call_count == 1


def test_retry_succeeds_after_failures():
    backend = _DummyBackend()
    fn = MagicMock(side_effect=[RuntimeError("fail"), RuntimeError("fail"), "ok"])

    with patch("audio2sub.ai.time.sleep"):  # skip actual sleep
        result = backend._retry(fn, retries=2)

    assert result == "ok"
    assert fn.call_count == 3


def test_retry_exhausted_raises():
    backend = _DummyBackend()
    fn = MagicMock(side_effect=RuntimeError("always fails"))

    try:
        with patch("audio2sub.ai.time.sleep"):
            backend._retry(fn, retries=2)
        assert False, "Should have raised"
    except RuntimeError as exc:
        assert "always fails" in str(exc)

    assert fn.call_count == 3  # initial + 2 retries


def test_retry_zero_means_no_retry():
    backend = _DummyBackend()
    fn = MagicMock(side_effect=RuntimeError("once"))

    try:
        backend._retry(fn, retries=0)
        assert False, "Should have raised"
    except RuntimeError:
        pass

    assert fn.call_count == 1


def test_default_retries_used():
    backend = _DummyBackend()
    assert backend.default_retries == 3

    fn = MagicMock(
        side_effect=[RuntimeError("1"), RuntimeError("2"), RuntimeError("3"), "ok"]
    )

    with patch("audio2sub.ai.time.sleep"):
        result = backend._retry(fn)  # no retries kwarg → use default (3)

    assert result == "ok"
    assert fn.call_count == 4  # 1 initial + 3 retries
