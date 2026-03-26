"""Question generation prompts per task type.

Each task type has examples of good/bad questions and fallback defaults.
Used in QUESTIONS phase — direct LLM call to generate investigation questions.
"""

BUG_QUESTIONS = """\
You need to understand a bug before you can fix it. Generate questions that
will help you reproduce, locate, and understand the root cause.

## EXAMPLES OF GOOD QUESTIONS

For "Users get a 500 error when uploading files larger than 10MB":
[
  {{"question": "What error message or stack trace appears in logs when the 500 occurs?", "intent": "reproduce"}},
  {{"question": "Which function handles file uploads, and what file is it in?", "intent": "find_code"}},
  {{"question": "Is there a file size limit configured, and where is it set?", "intent": "find_config"}},
  {{"question": "Are there existing tests for file upload that cover large files?", "intent": "check_tests"}},
  {{"question": "What is the current test baseline (how many pass/fail)?", "intent": "baseline"}}
]

For "Sort order is wrong in the task list — shows oldest first instead of newest":
[
  {{"question": "Which function builds the task list query, and what ORDER BY does it use?", "intent": "find_code"}},
  {{"question": "Are there tests that verify sort order? Do they pass or fail?", "intent": "check_tests"}},
  {{"question": "Is the sort order configurable or hardcoded?", "intent": "find_config"}}
]

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
[
  {{"question": "What patterns do existing endpoints follow? (decorators, auth, response format)", "intent": "find_patterns"}},
  {{"question": "What data models exist for users and their activity?", "intent": "find_code"}},
  {{"question": "Are there tests for the new endpoint already written? What do they expect?", "intent": "check_tests"}},
  {{"question": "What is the current test baseline?", "intent": "baseline"}},
  {{"question": "Are there similar stats/aggregation endpoints I can use as reference?", "intent": "find_patterns"}}
]

For "Implement a caching layer for database queries":
[
  {{"question": "How are database queries currently made? (ORM, raw SQL, query builder)", "intent": "find_code"}},
  {{"question": "What existing caching infrastructure exists? (Redis, in-memory, none)", "intent": "find_code"}},
  {{"question": "Which queries are most frequently called and would benefit from caching?", "intent": "find_code"}},
  {{"question": "Are there tests that verify query results? Will caching change behavior?", "intent": "check_tests"}},
  {{"question": "What is the project structure and where should new modules go?", "intent": "find_patterns"}}
]

For "Build a module from scratch based on a test specification":
[
  {{"question": "What is the full public API expected? (classes, methods, signatures, return types)", "intent": "understand_spec"}},
  {{"question": "How many tests are there and what categories do they cover?", "intent": "check_tests"}},
  {{"question": "What are the dependencies between features? (what must be built first)", "intent": "understand_spec"}},
  {{"question": "Are there existing files or patterns in the project to follow?", "intent": "find_patterns"}},
  {{"question": "What is the current test baseline?", "intent": "baseline"}}
]

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
[
  {{"question": "What does process() do and what are its logical sections?", "intent": "find_code"}},
  {{"question": "Who calls process()? (all importers and callers across the project)", "intent": "find_dependents"}},
  {{"question": "What is the current test baseline? All tests must keep passing", "intent": "baseline"}},
  {{"question": "Are there internal variables shared across the sections that complicate extraction?", "intent": "find_code"}}
]

For "Move UserService from services/ to a new domain/ directory":
[
  {{"question": "What files import UserService? (all references across the project)", "intent": "find_dependents"}},
  {{"question": "Does UserService depend on other services that would also need to move?", "intent": "find_code"}},
  {{"question": "What is the current test baseline?", "intent": "baseline"}}
]

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
[
  {{"question": "Where is the timeout configured? (file, line, env var)", "intent": "find_config"}},
  {{"question": "Are there other timeout settings that might need to change too?", "intent": "find_config"}}
]

For "Figure out why the deploy is failing":
[
  {{"question": "What does the deploy error log say?", "intent": "reproduce"}},
  {{"question": "What changed recently that could cause the failure?", "intent": "find_code"}},
  {{"question": "What is the deploy process? (scripts, CI config, commands)", "intent": "find_code"}}
]

## EXAMPLES OF BAD QUESTIONS

- "What is the project?" (too broad)
"""

OTHER_FALLBACK = [
    "What is the current state of the system/config related to the task?",
    "What needs to change and where?",
]
