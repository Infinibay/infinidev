"""PyInstaller entry point for the bundled engine sidecar.

Produces the `infinidev-server` binary the Tauri shell launches in packaged
builds. It simply delegates to the same CLI as `infinidev serve`, so the
binary accepts `--host` / `--port` / `--workdir`.
"""

from infinidev.server.cli import main

if __name__ == "__main__":
    main()
