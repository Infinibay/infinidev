# Records service

Three modules: `src/auth.py` verifies session tokens, `src/storage.py` reads and
writes the record file, and `src/api.py` exposes the two handlers the service
uses. There is no HTTP server in this repository; the handlers are called
directly by the deployment.
