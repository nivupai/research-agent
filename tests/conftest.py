from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from bioagent.common.httpclient import RateLimitedClient
from bioagent.mcp_servers.literature.europepmc import EuropePMCClient
from bioagent.mcp_servers.literature.pubmed import PubMedClient

FIXTURES = Path(__file__).parent / "fixtures"


class RecordingSleep:
    """Stand-in for ``asyncio.sleep`` that records requested delays without waiting."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.calls.append(delay)


@pytest.fixture
def efetch_xml() -> bytes:
    return (FIXTURES / "efetch_sample.xml").read_bytes()


@pytest.fixture
def sleeper() -> RecordingSleep:
    return RecordingSleep()


@pytest.fixture
async def http(sleeper: RecordingSleep) -> AsyncIterator[RateLimitedClient]:
    client = RateLimitedClient({}, default_rate=1000.0, sleep=sleeper)
    yield client
    await client.aclose()


@pytest.fixture
def pubmed(http: RateLimitedClient) -> PubMedClient:
    return PubMedClient(http, api_key="test-key", email="dev@example.org")


@pytest.fixture
def europepmc(http: RateLimitedClient) -> EuropePMCClient:
    return EuropePMCClient(http)
