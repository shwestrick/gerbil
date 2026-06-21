FROM debian:bookworm-slim

# elan needs curl + ca-certificates; lean projects need git; build-essential
# covers the C toolchain lean/leanc may invoke for native targets.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. uid/gid must match SANDBOX_UID/SANDBOX_GID in sandbox.py so
# uploaded files are owned by this user.
RUN useradd -m -u 1000 gerbil
USER gerbil
ENV HOME=/home/gerbil \
    PATH="/home/gerbil/.elan/bin:${PATH}"

# Install elan without a default toolchain. The project's lean-toolchain file
# (uploaded at session start) determines the version, installed on first use.
RUN curl -sSf https://elan.lean-lang.org/elan-init.sh | sh -s -- -y --default-toolchain none

WORKDIR /workspace/project
