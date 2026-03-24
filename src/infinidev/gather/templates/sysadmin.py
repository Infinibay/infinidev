"""Fixed questions for sysadmin tickets."""

from infinidev.gather.models import Question

QUESTIONS = [
    Question(
        id="current_config",
        question="What is the current configuration relevant to this task?",
        context_prompt=(
            "Read configuration files, environment variables, and settings relevant "
            "to this task. Check .env files, config directories, and settings modules.\n\n"
            "Task:\n{ticket_description}"
        ),
    ),
    Question(
        id="infrastructure",
        question="What infrastructure components are involved?",
        context_prompt=(
            "Check for infrastructure definitions:\n"
            "- Dockerfiles, docker-compose.yml\n"
            "- CI/CD configs (.github/workflows, .gitlab-ci.yml, Jenkinsfile)\n"
            "- Makefiles, justfiles, taskfiles\n"
            "- Deployment scripts\n\n"
            "Task:\n{ticket_description}"
        ),
    ),
    Question(
        id="dependencies",
        question="What system dependencies and versions are currently in use?",
        context_prompt=(
            "Check dependency files for current versions:\n"
            "- requirements.txt, pyproject.toml, setup.py (Python)\n"
            "- package.json (Node)\n"
            "- Cargo.toml (Rust)\n"
            "- go.mod (Go)\n"
            "- System package requirements\n\n"
            "Task:\n{ticket_description}"
        ),
    ),
    Question(
        id="scripts",
        question="What existing scripts or automation handle this concern?",
        context_prompt=(
            "Search for shell scripts, Makefiles, or CI workflows related to this task.\n"
            "Read them to understand what automation already exists.\n\n"
            "Task:\n{ticket_description}"
        ),
    ),
]
