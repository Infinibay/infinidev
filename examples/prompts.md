# Example Apps Built with Infinidev

These mini applications were built entirely by Infinidev in a single prompt using `qwen3.5:27b` running locally via Ollama.

Each example was generated with:
```bash
infinidev --prompt "<prompt below>" --model ollama_chat/qwen3.5:27b --no-tui
```

---

## 1. URL Shortener API

**Directory:** `examples/url-shortener/`
**Model:** `qwen3.5:27b`

```
Build a URL shortener web service from scratch.

Requirements:
1. Backend: FastAPI with SQLite database
2. Endpoints: POST /shorten (accepts a URL, returns a short code), GET /{code} (redirects to original URL), GET /stats/{code} (returns click count, creation date, last accessed)
3. Features: auto-generate 6-character alphanumeric codes, validate URLs, track click counts and timestamps, rate limiting (10 requests/minute per IP)
4. Frontend: single-page HTML form to shorten URLs and display the result, served at GET /
5. Include a requirements.txt and a run.sh script
6. Make it runnable with: bash run.sh
```

---

## 2. System Monitor TUI

**Directory:** `examples/sysmonitor/`
**Model:** `qwen3.5:27b`

```
Build a terminal-based system monitor dashboard using Python and the rich library.

Requirements:
1. Display live-updating panels for: CPU usage (per-core bar chart), RAM usage (used/total with bar), Disk usage (per mount point), Network I/O (bytes sent/received per second), Top 10 processes by CPU usage
2. Use rich.live and rich.table for the TUI layout
3. Update every 2 seconds
4. Support --interval flag to change refresh rate
5. Support --no-color flag for piping output
6. Use psutil for system metrics
7. Handle Ctrl+C gracefully with a clean exit
8. Include requirements.txt and make it runnable with: python monitor.py
```

---

## 3. Markdown Static Site Generator

**Directory:** `examples/sitegen/`
**Model:** `qwen3.5:27b`

```
Build a minimal static site generator that converts Markdown files into a styled HTML website.

Requirements:
1. CLI interface with Click: sitegen build --input ./content --output ./dist
2. Read all .md files from input directory (supports nested folders)
3. Parse YAML frontmatter (title, date, tags, template) from each markdown file
4. Convert markdown to HTML using the markdown library with extensions (fenced code blocks, tables, footnotes)
5. Apply Jinja2 HTML templates: base.html (layout), page.html (single page), index.html (lists all pages sorted by date)
6. Include a default template set with clean CSS (no framework, just custom styles)
7. Generate an index page listing all posts with titles, dates, and tags
8. Copy any static/ directory contents to output as-is
9. Include 3 sample markdown posts in content/ demonstrating different features
10. Include requirements.txt and make it runnable with: python -m sitegen build
```

---

## 4. Task Queue with Worker Pool

**Directory:** `examples/taskqueue/`
**Model:** `qwen3.5:27b`

```
Build a simple in-process task queue system with a worker pool in Python.

Requirements:
1. Core: TaskQueue class that accepts callables with arguments, distributes work to a configurable pool of worker threads
2. Features: priority levels (high/medium/low), task retry with exponential backoff (max 3 retries), task timeout, progress tracking, task status (pending/running/completed/failed)
3. API: queue.submit(fn, args, priority), queue.status(task_id), queue.results(), queue.wait_all()
4. CLI demo script that submits 20 mixed tasks (some that succeed, some that fail, some slow) and displays a live progress dashboard using rich
5. Include proper logging, graceful shutdown on Ctrl+C, and thread-safe operations
6. Project structure: taskqueue/ package with queue.py, worker.py, task.py modules
7. Include tests using pytest that verify: task execution, priority ordering, retry behavior, timeout handling
8. Include requirements.txt and make the demo runnable with: python demo.py
```

---

## 5. Project Management Dashboard (Full-Stack)

**Directory:** `examples/project-dashboard/`
**Model:** `qwen3.5:27b`

```
Build a full-stack project management dashboard with a FastAPI backend and React frontend.

BACKEND (FastAPI + SQLite):
1. Models: Project (id, name, description, color, created_at), Task (id, project_id, title, description, status, priority, assignee, due_date, created_at, updated_at), Activity (id, project_id, task_id, action, details, timestamp)
2. REST API: full CRUD for projects and tasks, GET /api/activity (recent activity feed), GET /api/stats (dashboard statistics: total tasks, completed, overdue, by priority, by status, completion rate per project)
3. Auto-log activity on task create/update/delete (e.g., "Task 'Fix login bug' moved to In Progress")
4. Seed data: 3 projects with 15-20 tasks across different statuses and priorities
5. Serve the React build from static files

FRONTEND (React + Tailwind CSS via CDN, no build step):
1. Use React and ReactDOM from CDN with Babel standalone (no npm/webpack needed)
2. Use Tailwind CSS from CDN
3. Dashboard page: stat cards (total tasks, completed, overdue, in progress) with icons, a donut chart showing tasks by status (use Chart.js), recent activity feed with timestamps
4. Kanban board page: drag-and-drop columns for Backlog, To Do, In Progress, Review, Done. Each card shows title, priority badge (color-coded), assignee avatar, due date. Moving a card updates the task status via API
5. Projects page: grid of project cards with progress bars, task counts, and color coding
6. Navigation: clean sidebar with icons, active state highlighting
7. Design: modern, clean UI with shadows, rounded corners, smooth transitions, and a professional color palette. Use Heroicons from CDN for icons
8. Make all API calls with fetch(), handle loading and error states

INFRASTRUCTURE:
1. Single run.sh that installs Python deps, seeds the database, and starts the server
2. Frontend files served from static/ directory (index.html, app.js, styles.css)
3. requirements.txt for Python dependencies
4. Make it runnable with: bash run.sh then open http://localhost:8000
```
