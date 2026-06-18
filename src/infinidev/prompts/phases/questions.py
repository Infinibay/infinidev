"""Question generation prompts per task type.

Each task type has examples of good/bad questions and fallback defaults.
Used in QUESTIONS phase — direct LLM call to generate investigation questions.
"""

BUG_QUESTIONS = """\
You need to understand a bug before you can fix it. Generate questions that
will help you reproduce, locate, and understand the root cause.

## EXAMPLES OF GOOD QUESTIONS

For "Users get a 500 error when uploading files larger than 10MB":
1. What error message or stack trace appears in logs when the 500 occurs?
2. Which function handles file uploads, and what file is it in?
3. Is there a file size limit configured, and where is it set?
4. Are there existing tests for file upload that cover large files?
5. What is the current test baseline (how many pass/fail)?

For "Sort order is wrong in the task list — shows oldest first instead of newest":
1. Which function builds the task list query, and what ORDER BY does it use?
2. Are there tests that verify sort order? Do they pass or fail?
3. Is the sort order configurable or hardcoded?

## EXAMPLES OF BAD QUESTIONS (DO NOT GENERATE THESE)

- "What is this project?" (too broad, not relevant to the bug)
- "What programming language is used?" (obvious from the code)
- "Can you explain the architecture?" (not targeted — ask about the specific area)
- "What should I fix?" (the task already tells you)
"""

BUG_FALLBACK = [
    "What error message, stack trace, or failing test reproduces this bug?",
    "What function and file contains the buggy code?",
    "Are there existing tests that cover this behavior, and do they pass or fail?",
]


FEATURE_QUESTIONS = """\
You need to understand the codebase and the specification before implementing.
Generate questions that will reveal the API, patterns, dependencies, and current state.

## EXAMPLES OF GOOD QUESTIONS

For "Add a /users/stats endpoint that returns activity metrics":
1. What patterns do existing endpoints follow? (decorators, auth, response format)
2. What data models exist for users and their activity?
3. Are there tests for the new endpoint already written? What do they expect?
4. What is the current test baseline?
5. Are there similar stats/aggregation endpoints I can use as reference?

For "Implement a caching layer for database queries":
1. How are database queries currently made? (ORM, raw SQL, query builder)
2. What existing caching infrastructure exists? (Redis, in-memory, none)
3. Which queries are most frequently called and would benefit from caching?
4. Are there tests that verify query results? Will caching change behavior?
5. What is the project structure and where should new modules go?

For "Build a module from scratch based on a test specification":
1. What is the full public API expected? (classes, methods, signatures, return types)
2. How many tests are there and what categories do they cover?
3. What are the dependencies between features? (what must be built first)
4. Are there existing files or patterns in the project to follow?
5. What is the current test baseline?

## EXAMPLES OF BAD QUESTIONS (DO NOT GENERATE THESE)

- "What should I implement?" (the task already says this)
- "Is there documentation?" (go read the files instead of asking)
- "How does everything work?" (too broad — ask about specific areas)
- "What tests should I write?" (read the test file to find out)
"""

FEATURE_FALLBACK = [
    "What is the expected API and behavior? (read tests/specs if they exist)",
    "What existing code patterns should the implementation follow?",
    "What is the current test baseline?",
    "What are the dependencies between components? (build order)",
]


REFACTOR_QUESTIONS = """\
Before refactoring, you must understand the code AND all its consumers.
Generate questions that map dependencies and establish a safety baseline.

## EXAMPLES OF GOOD QUESTIONS

For "Split the monolithic process() function into smaller helpers":
1. What does process() do and what are its logical sections?
2. Who calls process()? (all importers and callers across the project)
3. What is the current test baseline? All tests must keep passing.
4. Are there internal variables shared across the sections that complicate extraction?

For "Move UserService from services/ to a new domain/ directory":
1. What files import UserService? (all references across the project)
2. Does UserService depend on other services that would also need to move?
3. What is the current test baseline?

## EXAMPLES OF BAD QUESTIONS (DO NOT GENERATE THESE)

- "What does the code do?" (too vague — ask about the specific function/class)
- "Should I refactor this?" (the task already asks you to)
"""

REFACTOR_FALLBACK = [
    "What is the full test baseline? (must not regress after refactoring)",
    "What files and functions need to change?",
    "Who calls/imports the code being refactored? (all dependents)",
]


OTHER_QUESTIONS = """\
Generate questions to understand the current state before making changes.

## EXAMPLES OF GOOD QUESTIONS

For "Change the API timeout from 30s to 60s":
1. Where is the timeout configured? (file, line, env var)
2. Are there other timeout settings that might need to change too?

For "Figure out why the deploy is failing":
1. What does the deploy error log say?
2. What changed recently that could cause the failure?
3. What is the deploy process? (scripts, CI config, commands)

## EXAMPLES OF BAD QUESTIONS

- "What is the project?" (too broad)
"""

OTHER_FALLBACK = [
    "What is the current state of the system/config related to the task?",
    "What needs to change and where?",
]
