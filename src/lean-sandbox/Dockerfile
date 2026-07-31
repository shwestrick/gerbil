FROM debian:bookworm-slim

# apt drops privileges to the `_apt` user to fetch packages, which needs a
# second uid inside the build's user namespace. Rootless podman only has one
# unless the host grants a /etc/subuid range, and without it the build dies with
# "setgroups: Operation not permitted". Fetching as root inside a throwaway
# build container costs nothing and keeps the image buildable everywhere.
RUN printf 'APT::Sandbox::User "root";\n' > /etc/apt/apt.conf.d/99gerbil-no-sandbox

# elan needs curl + ca-certificates; lean projects need git; build-essential
# covers the C toolchain lean/leanc may invoke for native targets. python3 +
# venv run the lean-lsp-mcp server; ripgrep powers its local-search tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential \
    python3 python3-venv ripgrep \
    && rm -rf /var/lib/apt/lists/*

# The uid/gid the sandbox runs as. The default, 1000, is the non-root `gerbil`
# user and must match SANDBOX_UID/SANDBOX_GID in sandbox.py.
#
# Build with --build-arg SANDBOX_UID=0 for an image that runs as root *inside
# the container*. That is the only way to run under rootless podman on a host
# with no /etc/subuid range for the user: with a single uid mapped, the build
# cannot chown anything to uid 1000 and the runtime cannot map it. It does not
# make the sandbox more privileged on the host -- rootless podman maps container
# root to the (unprivileged) invoking user. The launcher picks this automatically
# when it detects that situation; everywhere else the non-root default stands.
ARG SANDBOX_UID=1000

# Home and workspace are created and owned explicitly here rather than left to
# `useradd -m` and the WORKDIR below, for two reasons:
#   - every chown is to $SANDBOX_UID, which is a no-op when that is 0, so the
#     build needs only the single uid a bare rootless podman has;
#   - docker materializes a missing WORKDIR when the container starts, but
#     buildah/podman only records the path -- the directory would not exist in
#     the image, and the sandbox would fail to start with
#     `workdir "/workspace/project" does not exist`.
# The group is created explicitly too, so the gid is 1000 rather than whatever
# useradd picks next.
RUN if [ "$SANDBOX_UID" -ne 0 ]; then \
        groupadd -g "$SANDBOX_UID" gerbil \
        && useradd -M -u "$SANDBOX_UID" -g "$SANDBOX_UID" -d /home/gerbil gerbil; \
    fi \
    && mkdir -p /home/gerbil /workspace/project \
    && chown -R "$SANDBOX_UID:$SANDBOX_UID" /home/gerbil /workspace

USER $SANDBOX_UID
# HOME is pinned to /home/gerbil for both uids (root's would otherwise be /root),
# so elan, the venv, and git config live at the same paths in either image.
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
