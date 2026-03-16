"""Docker-specific coding guidelines for the Developer agent."""


def get_prompt() -> str:
    return """\
## Docker Guidelines

### Naming Conventions
- Tag images with semver: `registry/org/image:1.2.3` (e.g. `ghcr.io/org/app:1.2.3`).
- Use `latest` only for local development — never in production deployments or CI pipelines.
- Use `kebab-case` for service names in `docker-compose.yml`.
- Use descriptive stage names in multi-stage builds (`FROM node:20 AS builder`).

### Security
- Run containers as a non-root user: add `USER appuser` after creating the user.
- Use `--read-only` filesystem where possible — write only to explicit volumes.
- Never embed secrets in `ENV`, `ARG`, or `COPY` — inject them at runtime via Docker secrets or environment variables.
- Scan images regularly with `docker scout` or `trivy`.
- Use minimal base images: `distroless`, `alpine`, or `-slim` variants.
- Drop all capabilities and add back only what is needed: `--cap-drop=ALL --cap-add=NET_BIND_SERVICE`.

### Dockerfile Best Practices
- Use multi-stage builds to separate build dependencies from the final runtime image.
- Order layers from least to most frequently changed — install dependencies before copying source code.
- Use `COPY --chown=appuser:appuser` instead of a separate `RUN chown` step.
- Pin base image digests in production: `FROM node:20@sha256:abc123...`.
- Combine `RUN` commands with `&&` to reduce layer count:
  ```dockerfile
  RUN apt-get update && \
      apt-get install -y --no-install-recommends pkg && \
      rm -rf /var/lib/apt/lists/*
  ```

### Useful Patterns
- Use `.dockerignore` to exclude `node_modules`, `.git`, test files, and documentation.
- Add `HEALTHCHECK` for all long-running services to enable orchestrator health monitoring.
- Use `ENTRYPOINT` for the executable and `CMD` for default arguments — allows overriding args at runtime.
- Use build args for version pinning: `ARG NODE_VERSION=20` then `FROM node:${NODE_VERSION}`.

### Anti-Patterns to Avoid
- `RUN apt-get install` without `--no-install-recommends` and cache cleanup — bloats images.
- `ADD` instead of `COPY` for local files — `ADD` has implicit tar extraction and URL fetch that are rarely needed.
- Running `sshd` inside containers — use `docker exec` or orchestrator tools instead.
- Storing application state in the container filesystem — use volumes or external storage.
- `privileged: true` in compose files — grants full host access; almost never necessary.\
"""
