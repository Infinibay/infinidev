"""Fixed questions for bug tickets."""

from infinidev.gather.models import Question

QUESTIONS = [
    Question(
        id="reproduction",
        question="How can this bug be reproduced? What error messages or tracebacks are involved?",
        context_prompt=(
            "Search the codebase for the error messages, function names, or code paths "
            "described in this bug report. Find relevant test files that exercise the "
            "affected functionality. Read the code to understand the reproduction path.\n\n"
            "Bug report:\n{ticket_description}"
        ),
    ),
    Question(
        id="expected_vs_actual",
        question="What is the expected behavior vs. the actual behavior?",
        context_prompt=(
            "Read the source code and tests related to this bug to determine:\n"
            "1. What the code SHOULD do (from tests, docstrings, or the bug report)\n"
            "2. What the code ACTUALLY does (trace the code path)\n"
            "3. Where the behavior diverges\n\n"
            "Bug report:\n{ticket_description}"
        ),
    ),
    Question(
        id="related_files",
        question="What files, classes, and functions are directly involved in this bug?",
        context_prompt=(
            "Search for ALL files related to this bug. For each file found:\n"
            "- Read it to understand its role\n"
            "- Identify the specific functions/classes involved\n"
            "- Note line numbers of relevant code\n"
            "- Check imports to find connected modules\n"
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
            "3. Any callers or dependents that might be affected\n"
            "Search for similar patterns in the codebase that might have the same bug.\n\n"
            "Bug report:\n{ticket_description}"
        ),
    ),
    Question(
        id="test_coverage",
        question="What existing tests cover the affected code? How should the fix be verified?",
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
