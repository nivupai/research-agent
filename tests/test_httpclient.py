from __future__ import annotations

import httpx
import pytest
import respx

from bioagent.common.httpclient import (
    LiteratureAPIError,
    RateLimitedClient,
    parse_retry_after,
)
from tests.conftest import RecordingSleep

URL = "https://api.example.org/thing"


@respx.mock
async def test_retry_on_429_honours_retry_after(
    http: RateLimitedClient, sleeper: RecordingSleep
) -> None:
    route = respx.get(URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "2"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    response = await http.get(URL)
    assert response.json() == {"ok": True}
    assert route.call_count == 2
    assert sleeper.calls == [2.0]


@respx.mock
async def test_retry_on_5xx_and_timeout_with_backoff(
    http: RateLimitedClient, sleeper: RecordingSleep
) -> None:
    route = respx.get(URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.ReadTimeout("slow"),
            httpx.Response(200, text="done"),
        ]
    )
    response = await http.get(URL)
    assert response.text == "done"
    assert route.call_count == 3
    assert len(sleeper.calls) == 2
    # equal-jitter exponential backoff: attempt 0 in [0.25, 0.5], attempt 1 in [0.5, 1.0]
    assert 0.25 <= sleeper.calls[0] <= 0.5
    assert 0.5 <= sleeper.calls[1] <= 1.0


@respx.mock
async def test_gives_up_after_max_attempts(sleeper: RecordingSleep) -> None:
    client = RateLimitedClient({}, default_rate=1000.0, max_attempts=3, sleep=sleeper)
    route = respx.get(URL).mock(return_value=httpx.Response(500))
    with pytest.raises(LiteratureAPIError, match="after 3 attempts"):
        await client.get(URL)
    assert route.call_count == 3
    assert len(sleeper.calls) == 2
    await client.aclose()


@respx.mock
async def test_client_error_not_retried(http: RateLimitedClient) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(400))
    with pytest.raises(LiteratureAPIError, match="HTTP 400") as excinfo:
        await http.get(URL, params={"api_key": "s3cret"})
    assert route.call_count == 1
    assert "s3cret" not in str(excinfo.value)


def test_parse_retry_after() -> None:
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after("999") == 60.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("garbage") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # in the past


def test_per_host_buckets() -> None:
    client = RateLimitedClient({"a.example": 10.0}, default_rate=2.0)
    assert client.bucket_for("https://a.example/x").rate == 10.0
    assert client.bucket_for("https://b.example/y").rate == 2.0
    assert client.bucket_for("https://b.example/z") is client.bucket_for("https://b.example/y")
