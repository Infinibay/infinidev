"""Fixed questions for bug tickets."""

from infinidev.gather.models import Question

QUESTIONS = [
    Question(
        id="reproduction",
        question="How can this bug be reproduced? What error messages or tracebacks are involved?",
        context_prompt=(
            "Search the codebase for the error messages, function names, or code paths "
            "described in this bug report. Find test files that execute the "
            "affected functionality. Read the code to understand the reproduction path.\n\n"
            "Bug report:\n{ticket_description}"
        ),
    ),
    Question(
        id="expected_vs_actual",
        question="What is the expected behavior vs. the actual behavior?",
        context_prompt=(
            "Read the source code and tests related to this bug to determine:\n"
            "1. The expected behavior from tests, docstrings, or the bug report\n"
            "2. What the code ACTUALLY does (trace the code path)\n"
            "3. Where the behavior diverges\n\n"
            "Bug report:\n{ticket_description}"
        ),
    ),
    Question(
        id="related_files",
        question="What files, classes, and functions are directly involved in this bug?",
        context_prompt=(
            "Start with project_structure() to understand the project layout.\n"
            "Then use search_symbols and find_references to trace the code involved.\n"
            "For each file containing an affected symbol, use list_symbols to inspect it.\n"
            "Use get_symbol_code to read specific functions/methods.\n"
            "Be thorough — most bugs require changes in MULTIPLE locations.\n\n"
            "Bug report:\n{ticket_description}"
        ),
    ),
    Question(
        id="root_cause_candidates",
        question="What are the likely root causes? What code paths need to change?",
        context_prompt=(
            "Based on the files and behavior found, identify:\n"
            "1. The PRIMARY root cause (where the bug originates)\n"
            "2. SECONDARY locations that also need fixing (same pattern, related checks, "
            "string comparisons, etc.)\n"
            "3. Callers or dependents whose behavior crosses the failing path\n"
            "Search for the same failing pattern in other code paths.\n\n"
            "Bug report:\n{ticket_description}"
        ),
    ),
    Question(
        id="test_coverage",
        question="Which existing tests execute the affected code, and which command runs them?",
        context_prompt=(
            "Find test files that test the code affected by this bug:\n"
            "- Search for test files in tests/ or test directories\n"
            "- Read the tests to understand what they assert\n"
            "- Identify which tests would need to pass after the fix\n"
            "- Note how tests are structured (framework, fixtures, patterns)\n\n"
            "Bug report:\n{ticket_description}"
        ),
    ),
]
