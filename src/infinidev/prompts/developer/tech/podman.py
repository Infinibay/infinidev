"""Podman-specific coding guidelines for the Developer agent."""


def get_prompt() -> str:
    return """\
## Podman Guidelines

Follow all Docker/container best practices, plus these Podman-specific considerations.

### Rootless by Default
- Podman runs rootless by default — no daemon, no root privileges required.
- Use `podman unshare` for operations that need the user namespace (e.g. fixing file ownership in volumes).
- User namespaces map UID 0 inside the container to the host user's UID — be aware of this for file permissions.
- Use `--userns=keep-id` when bind-mounting host directories to preserve file ownership.

### Systemd Integration
- Generate systemd unit files: `podman generate systemd --new --name <container> > ~/.config/systemd/user/<name>.service`.
- Enable automatic image updates with the `io.containers.autoupdate=registry` label and `podman auto-update`.
- Use `loginctl enable-linger <user>` so rootless containers survive user logout.
- Manage container lifecycle with `systemctl --user start/stop/enable <service>`.

### Pods
- Use `podman pod create --name <pod> -p 8080:80` to group related containers sharing a network namespace (analogous to Kubernetes pods).
- Use `podman play kube <manifest.yaml>` to run Kubernetes YAML manifests directly.
- Use `podman generate kube <pod>` to export running pods as Kubernetes manifests.

### Security
- Use `--security-opt=no-new-privileges` to prevent privilege escalation inside containers.
- Use SELinux labels for volume mounts: `:z` for shared, `:Z` for private.
- Use `podman secret create` and `--secret` for sensitive data instead of environment variables.
- Use `podman image sign` for image signature verification in trusted environments.

### Anti-Patterns to Avoid
- Assuming 100% Docker CLI compatibility — some flags and behaviours differ (e.g. networking, volume permissions).
- Mounting `docker.sock` — use `podman.sock` or the `--remote` flag instead.
- Ignoring rootless networking limitations — binding to ports below 1024 requires `net.ipv4.ip_unprivileged_port_start` sysctl adjustment.
- Using `podman` with `sudo` habitually — leverage rootless mode and only escalate when truly needed.
- Running Podman inside Docker without proper nested namespace configuration.\
"""
