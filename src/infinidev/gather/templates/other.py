"""Fallback questions for unclassified tickets."""

from infinidev.gather.models import Question

QUESTIONS = [
    Question(
        id="project_structure",
        question="What is the overall project structure?",
        context_prompt=(
            "List the top-level directories and key files.\n"
            "Read README or CONTRIBUTING if they exist.\n"
            "Understand what this project does and how it's organized.\n\n"
            "Task:\n{ticket_description}"
        ),
    ),
    Question(
        id="related_files",
        question="Which files define the symbols or behavior named by this task?",
        context_prompt=(
            "Search for keywords from the task description in the codebase.\n"
            "Read the files that define or test the matched symbols and behavior.\n"
            "Identify classes, functions, and modules that relate to the task.\n\n"
            "Task:\n{ticket_description}"
        ),
    ),
    Question(
        id="context",
        question="What additional context from the codebase helps understand this task?",
        context_prompt=(
            "Look for documentation, comments, or code that provides context:\n"
            "- README, CONTRIBUTING, CHANGELOG\n"
            "- Docstrings on matched modules and symbols\n"
            "- Existing tests that show expected behavior\n\n"
            "Task:\n{ticket_description}"
        ),
    ),
]
