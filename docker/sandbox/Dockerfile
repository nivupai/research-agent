# Throwaway execution environment for the bioagent sandbox MCP server.
# Containers from this image run with no network, a read-only root filesystem, no capabilities
# and resource limits; see src/bioagent/mcp_servers/sandbox/executor.py.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp

RUN pip install \
        "numpy>=2,<3" \
        "pandas>=2,<4" \
        "scipy>=1.13,<2" \
        "matplotlib>=3.9,<4" \
        "rdkit>=2024.9" \
    && groupadd --gid 10001 sandbox \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin sandbox \
    && mkdir -p /work /opt/sandbox \
    && chown sandbox:sandbox /work

COPY --chmod=0444 runner.py /opt/sandbox/runner.py

USER 10001:10001
WORKDIR /work
CMD ["python", "-s", "/opt/sandbox/runner.py"]
