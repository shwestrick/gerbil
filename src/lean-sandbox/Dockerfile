FROM debian:bookworm-slim

# elan needs curl + ca-certificates; lean projects need git; build-essential
# covers the C toolchain lean/leanc may invoke for native targets. python3 +
# venv run the lean-lsp-mcp server; ripgrep powers its local-search tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential \
    python3 python3-venv ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. uid/gid must match SANDBOX_UID/SANDBOX_GID in sandbox.py so
# uploaded files are owned by this user.
RUN useradd -m -u 1000 gerbil
USER gerbil
ENV HOME=/home/gerbil \
    PATH="/home/gerbil/.venv/bin:/home/gerbil/.elan/bin:${PATH}"

# Install elan without a default toolchain. The project's lean-toolchain file
# (uploaded at session start) determines the version, installed on first use.
RUN curl -sSf https://elan.lean-lang.org/elan-init.sh | sh -s -- -y --default-toolchain none

# Pre-bake the lean-lsp-mcp server into a venv so no download happens at session
# start. The host connects to it over `docker exec -i ... lean-lsp-mcp` (on PATH).
RUN python3 -m venv /home/gerbil/.venv \
    && /home/gerbil/.venv/bin/pip install --no-cache-dir lean-lsp-mcp

WORKDIR /workspace/project
