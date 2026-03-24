"""Fixed questions for refactor tickets."""

from infinidev.gather.models import Question

QUESTIONS = [
    Question(
        id="current_state",
        question="What does the code currently look like? What is being refactored?",
        context_prompt=(
            "Read all files mentioned or implied by the refactoring request.\n"
            "For each file: understand the current structure, identify what needs to change, "
            "and note the current patterns in use.\n\n"
            "Refactoring request:\n{ticket_description}"
        ),
    ),
    Question(
        id="callers",
        question="What code calls or depends on the code being refactored?",
        context_prompt=(
            "Search for all usages, imports, and references to the code being refactored:\n"
            "- Function/method callers\n"
            "- Import statements\n"
            "- Type annotations referencing the code\n"
            "- Configuration referencing class/function names\n"
            "All of these will need updating when the refactor happens.\n\n"
            "Refactoring request:\n{ticket_description}"
        ),
    ),
    Question(
        id="test_coverage",
        question="What tests exist for the code being refactored?",
        context_prompt=(
            "Find all tests that cover the code being refactored:\n"
            "- Direct unit tests\n"
            "- Integration tests that exercise the code path\n"
            "- Tests that will need updating after the refactor\n"
            "- Tests that should continue to pass unchanged (regression guards)\n\n"
            "Refactoring request:\n{ticket_description}"
        ),
    ),
    Question(
        id="patterns",
        question="What patterns and conventions does the surrounding code follow?",
        context_prompt=(
            "Read neighboring functions, classes, and modules to understand:\n"
            "- Naming conventions\n"
            "- Error handling patterns\n"
            "- Code organization style\n"
            "- Design patterns in use\n"
            "The refactored code should follow these same conventions.\n\n"
            "Refactoring request:\n{ticket_description}"
        ),
    ),
]
