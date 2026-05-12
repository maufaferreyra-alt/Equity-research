"""
Shared retry policy for provider HTTP calls.

Tenacity defaults to retry-on-any-exception, which wastes time
retrying deterministic failures (missing API key, malformed ticker,
parse errors). This policy retries only on TRANSIENT errors (network,
timeouts, 5xx, 429) and fails fast on everything else.

Centralizing the policy here prevents re-introducing the missing
retry_if_exception_type bug when new providers are added.
"""

from __future__ import annotations
import requests

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )
    _TENACITY = True
except ImportError:
    _TENACITY = False
    def retry(**_kw):
        def decorator(fn):
            return fn
        return decorator
    def stop_after_attempt(*_a, **_kw): return None
    def wait_exponential(*_a, **_kw): return None
    def retry_if_exception_type(*_a, **_kw): return None


# Transient errors worth retrying with exponential backoff.
# Deterministic errors (MissingAPIKeyError, TickerNotFoundError,
# parsing failures) are NOT included — those raise immediately.

# RateLimitError lives in core.exceptions. Try-import to avoid a
# potential circular-import edge case if a provider needs the
# retry policy at module-load time before core.exceptions is
# initialized. If unavailable, the tuple gracefully falls back
# to the non-rate-limit transient errors only.
try:
    from core.exceptions import RateLimitError
    _RATE_LIMIT_EXC: tuple = (RateLimitError,)
except ImportError:
    _RATE_LIMIT_EXC = ()

RETRYABLE_EXCEPTIONS = (
    requests.RequestException,
    requests.Timeout,
    requests.ConnectionError,
    TimeoutError,
    *_RATE_LIMIT_EXC,
)


def provider_retry(attempts: int = 3, min_wait: float = 2, max_wait: float = 10):
    """Standard retry decorator for provider HTTP calls.

    Usage:
        from data._retry_policy import provider_retry

        @provider_retry()
        def _get(...):
            ...

        # Custom config when needed (e.g., longer wait for rate-limited APIs):
        @provider_retry(attempts=5, max_wait=30)
        def _heavy_call(...):
            ...
    """
    if not _TENACITY:
        # No-op decorator when tenacity unavailable
        def decorator(fn):
            return fn
        return decorator
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
