"""Prompt for the /init command — project exploration and documentation."""

INIT_TASK_DESCRIPTION = """\
## Task: Explore and Document This Project

You are initializing the knowledge base for this project. Explore the codebase
thoroughly and record everything a developer would need to know to start
working on it.

## What to Explore

1. **Project root** — Read README, CHANGELOG, LICENSE, config files
   (package.json, pyproject.toml, Cargo.toml, Makefile, docker-compose.yml,
   etc.). Understand what this project IS.
2. **Directory structure** — Use list_directory on the root and key
   subdirectories. Understand how the project is organized.
3. **Entry points** — Find the main entry point(s): main(), app.run(),
   CLI commands, HTTP server setup, etc.
4. **Core modules** — Read the most important source files. Understand the
   core abstractions, classes, and how they connect.
5. **Dependencies** — Check the dependency file (requirements.txt,
   package.json, Cargo.toml, go.mod). Note key libraries and their purpose.
6. **Tests** — Check if tests exist, what framework is used, how to run them.
7. **Build/run commands** — How to install, build, run, and test the project.
8. **Configuration** — Environment variables, config files, settings.

## How to Document — Standard Format

ALL documentation goes to the DATABASE, not to files. Do NOT create, write, or
modify any files. Use these tools:

- **record_finding** — For each piece of project knowledge (structure, classes,
  patterns, config, etc.). Use finding_type="project_context", confidence=1.0.
- **update_documentation** — If you find key external libraries, fetch their
  docs and store them in the library_docs table for future reference.

### Finding type: `project_context` (for all structural knowledge)

Use these EXACT topic prefixes so findings are searchable and consistent:

| Topic prefix | What to document | Example |
|---|---|---|
| `[project] Overview` | What the project does, its purpose, tech stack | "FastAPI web app for inventory management, Python 3.11, PostgreSQL" |
| `[project] Structure` | Directory layout and what each dir contains | "src/ = app code, tests/ = pytest tests, migrations/ = alembic" |
| `[project] Entry points` | How the app starts, main files | "Entry: src/main.py:main(), CLI via click, runs uvicorn server" |
| `[project] Build & run` | Install, build, test, deploy commands | "Install: pip install -e ., Run: uvicorn src.main:app, Test: pytest" |
| `[project] Config` | Environment vars, config files, settings | "DB_URL in .env, settings in src/config.py via pydantic-settings" |
| `[project] Dependencies` | Key libraries and why they're used | "SQLAlchemy for ORM, Pydantic for validation, httpx for HTTP client" |
| `[class] ClassName` | Purpose, key methods, file location | "UserService in src/services/user.py — handles registration, auth, profile CRUD" |
| `[module] module_name` | What the module does, key exports | "src/auth/ — JWT token generation, middleware, permission decorators" |
| `[pattern] Pattern name` | Design patterns and conventions used | "Repository pattern: all DB access through src/repositories/, never direct queries" |
| `[api] Endpoint group` | API routes, request/response format | "POST /auth/login — takes {email, password}, returns {token, refresh_token}" |
| `[test] Testing setup` | Test framework, how to run, conventions | "pytest + pytest-asyncio, fixtures in conftest.py, DB reset per test" |

### Rules for content quality

- **Be specific.** File paths, line numbers, function signatures, actual values.
- **Be complete.** Each finding should be self-contained — useful without
  needing to read other findings.
- **Set confidence to 1.0** — you are reading the actual code, not guessing.
- **Tag findings** with relevant keywords for searchability.
- **One finding per concept.** Do not cram everything into one giant finding.
  Separate structure from classes from patterns.

## Workflow

1. Start with list_directory on the project root
2. Read the main config file (pyproject.toml, package.json, etc.)
3. Read the README if it exists
4. Explore the source directory structure
5. Read key source files (entry points, core modules)
6. Record findings as you go — do not wait until the end
7. After exploring, review your notes and add any missing context

## Important

- Do NOT create, write, or modify any files. This is read-only exploration.
  ALL output goes to the database via record_finding and update_documentation.
- Do NOT skip recording findings. Every important discovery must be recorded.
- Use add_note to track what you've already explored so you don't repeat yourself.
- If the project is large, focus on the most important parts first (entry points,
  core logic, public API) and note what remains to be explored.
- For key external dependencies (frameworks, libraries), use update_documentation
  to fetch and store their documentation for future developer reference.
"""

INIT_EXPECTED_OUTPUT = (
    "Comprehensive project documentation recorded to the knowledge base. "
    "Report a summary of what was documented and any areas that need "
    "further exploration."
)
