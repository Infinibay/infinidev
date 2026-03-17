"""Developer task prompts used in flows."""

from infinidev.prompts.shared import GIT_WORKFLOW_INSTRUCTIONS
from infinidev.prompts.team import build_conversation_context, build_state_context


def implement_code(
    task_id: int,
    task_title: str,
    task_description: str,
    conversation_context: str = "",
    project_id: int = 0,
    project_name: str = "",
) -> tuple[str, str]:
    """Return (description, expected_output) for code implementation."""

    state_block = build_state_context(
        project_id=project_id,
        project_name=project_name,
        phase="implementation",
        summary=f"Implementing task {task_id}. Create branch, write code, test, commit, push.",
    )

    ctx_block = conversation_context or ""

    from infinidev.config.settings import settings

    description = f"""\
Implement task {task_id}: {task_title}

{state_block}

## Task Specifications
{task_description}

{ctx_block}

## Workspace
Your shell commands run in your **isolated git worktree** by default.
Do NOT `cd` to other directories — just run commands directly (e.g.
`cargo test`, `pytest -x`, `ls src/`). Paths like `/home/developer` or
`/home/user` do NOT exist.

## Repository Requirement
A git repository MUST exist before you start working. The repository should
already be configured with `origin` pointing to Forgejo.

If `git status` fails or `git remote -v` shows no remotes:
1. Use **send_message** to inform the Team Lead that no repository is
   configured for this project.
2. The Team Lead will coordinate with the Project Lead to create one.
3. Do NOT proceed with code changes until a repository is available.
4. Do NOT attempt to create a repository yourself.

{GIT_WORKFLOW_INSTRUCTIONS}

## Forgejo Remote Context
The remote `origin` on this workspace points to the Forgejo server.
FORGEJO_API_URL = {settings.FORGEJO_API_URL or "http://localhost:3000/api/v1"}
FORGEJO_OWNER   = {settings.FORGEJO_OWNER or "infinibay"}
FORGEJO_REPO    = $FORGEJO_REPO   (set in environment — format: "owner/repo-name")

If `git push` fails with "repository not found", the remote is not configured.
Fix it with:
  execute_command → "git remote set-url origin http://localhost:3000/$FORGEJO_OWNER/$FORGEJO_REPO_NAME.git"
Then retry git_push.

## Your Goal
Implement the code changes required by this task. The code must fulfill all
acceptance criteria, be tested, committed on a properly named branch, and
pushed to the remote.

## Step-by-Step Process

### Step 0: Resume Check
Call **load_session_note** with the current task_id. If a prior session
exists, read `notes_json` and look for the `"step"` key to know exactly
which step was last completed. Recover your branch name, files already
modified, and last known state from the rest of `notes_json`. Resume from
step **N+1** where N is the last completed step. If no session exists,
start from Step 1.

### Step 1: Create a Branch
Use **git_branch** with branch_name=`task-{task_id}-<slug>`, create=true,
base_branch="main". This runs:
  git fetch origin main
  git checkout -b task-{task_id}-<slug> origin/main

git_branch **automatically sets `branch_name`** on the task record — you
do not need to update it manually.

Post the branch name as a comment on the task using **add_comment**.

Save progress: call **save_session_note** with phase="implementing",
notes_json={{"branch": "<branch_name>", "step": 1}}.

### Step 2: Read Existing Code
Before writing anything, use **read_file** to read the files you plan
to modify. Understand:
- The existing code structure and patterns.
- How the code you will change is used by other parts of the system.
- What conventions are followed (naming, error handling, imports).

Use **code_search** to find all usages of functions or classes you plan
to modify. This helps you understand the impact of your changes and avoid
breaking callers.

### Step 3: Implement the Changes
For each change, follow existing codebase conventions, handle error conditions
appropriately, do NOT introduce security vulnerabilities (injection, XSS,
hardcoded secrets), and keep changes focused on the task scope — do not
refactor unrelated code.

**Modifying an existing file** (preferred path):
1. Use **code_search** to locate the exact code to change.
2. Use **read_file** with `offset` and `limit` to read the relevant
   section — do NOT read entire large files when you only need a fragment.
3. Use **edit_file** to apply a surgical edit. The `old_string` must be
   unique within the file — include enough surrounding context to guarantee
   uniqueness.

**Creating a new file** (only when the file does not exist yet):
- Use **write_file** with the full absolute path.

**IMPORTANT**: Never use **write_file** to overwrite an existing file.
Overwriting destroys the change history and makes code review impossible.
If the file already exists, always use **edit_file**.

Save progress: call **save_session_note** with phase="implementing",
notes_json={{"branch": "...", "files_changed": [...], "step": 3}},
last_file=<last file edited>.

### Step 4: Write Tests
Write tests for new or modified functionality:
- Happy path: the feature works as specified.
- Error cases: invalid input, missing data, failures.
- Edge cases: boundary conditions, empty inputs, large inputs.

Place tests in the project's test directory following existing conventions.

### Step 5: Run Tests

#### 5a. Estimate suite duration
Before running the full suite, estimate its size:
1. Run **execute_command** with `pytest --co -q` (collect-only) to count
   the total number of tests.
2. If the count exceeds **500 tests**, or if a previous CI run recorded a
   duration > 600 seconds, switch to **relevant-only mode** (Step 5c).
   Otherwise use **full suite mode** (Step 5b).

#### 5b. Full suite mode (≤ threshold)
Run **execute_command** with `pytest -x -q` — all tests, stop on first
failure. ALL tests must pass — both your new tests and existing ones.

#### 5c. Relevant-only mode (> threshold)
Derive the list of test files directly related to the files you modified in
this task and run only those:
  `pytest <test_file_1> <test_file_2> ... -x -q`
This keeps CI under the 10-minute budget while still validating your changes.

#### 5d. Fix-and-rerun
If tests fail:
- Read the failure output carefully.
- Fix the issue in your code (not the test, unless the test is wrong).
- Re-run until all tests pass.

**Clarification during implementation**: You may ask at most **1 clarification
question** during implementation. If no response, proceed with assumptions
and document with add_comment (prefix: `ASSUMPTION:`). Do NOT create new
tasks, epics, or other resources as a workaround — only continue working on
your assigned task.

**Escalation — persistently failing tests**: If after **3 fix-and-rerun
cycles** the tests still fail, STOP trying and escalate:
1. Use **ask_team_lead** with a structured message containing:
   - The exact test command that fails.
   - The full error output.
   - A summary of the 3 fixes you already attempted and why each did not
     resolve the issue.
   - A specific question (not "what should I do?" — describe the concrete
     blocker).
2. Do NOT move the task to `review_ready` while tests are failing.
3. Do NOT commit or push code with failing tests.
4. Wait for the Team Lead's response before continuing.

**⚠️ HARD GATE: Setting the task status to `review_ready` with failing tests
is a protocol violation and will be treated as a blocking defect by the Code
Reviewer. The CI gate will automatically reject the submission before the
reviewer even sees it.**

### Step 6: Self-Review
Use **git_diff** to review your own changes before committing. Check for:
- Unintended changes (debug prints, commented-out code, unrelated edits).
- Missing error handling.
- Security issues.
- Code that does not match the surrounding style.

Remove any debugging artifacts before committing.

### Step 7: Commit and Push
Follow the COMMIT and PUSH steps from the Git Workflow above.
Concretely:
1. **git_commit** — stages all changes (`git add -A`) and commits
   (`git commit -m "<message>"`).
2. **git_push** — pushes the branch to origin on the Forgejo server
   (`git push -u origin <branch>`).

### Step 8: Post Review Comment and Submit
The Code Reviewer is a separate agent — they only see what is committed
and pushed (via git_diff, read_file, get_task, and your comments).

1. Use **add_comment** on the task with a submission comment containing:
   - The exact branch name (e.g., `task-7-auth-middleware`).
   - A summary of what changed and why.
   - How to test it (test command or manual steps).
   - Any assumptions made (prefix: `ASSUMPTION:`).
   This is NOT optional — a vague "implementation done" forces the reviewer
   to guess, which slows down the entire review cycle.

2. Use **update_task_status** to set the status to `review_ready`.
   If git_branch or create_pr failed to auto-set branch/PR info,
   pass optional `branch_name` and/or `pr_url` parameters as a fallback.

Save progress: call **save_session_note** with phase="complete",
notes_json={{"branch": "...", "status": "review_ready", "step": 8}}.

## Important Reminders
- Do NOT move to `review_ready` without running tests.
- Do NOT push code with failing tests.
- Do NOT make changes outside the scope of the task.
- Do NOT skip the add_comment submission comment before `review_ready`.
"""

    expected_output = """\
A summary of the implementation containing:

1. **Branch & files changed**: The branch name (format: task-{id}-{slug})
   and list of files created or modified with a brief description of each.
2. **Test results**: Tests added, strategy used (full suite or relevant-only
   with reasoning), and confirmation that all tests pass.
3. **PR & status**: The PR URL and confirmation that the task was moved
   to `review_ready` with a proper submission comment.
4. **Escalations** (if any): If tests failed persistently and
   ask_team_lead was used, what was asked and the response received.
"""
    return description, expected_output


def respond_to_clarification(
    task_id: int,
    task_title: str,
    clarification_message: str,
    project_id: int = 0,
    project_name: str = "",
) -> tuple[str, str]:
    """Return (description, expected_output) for responding to team lead clarification."""

    state_block = build_state_context(
        project_id=project_id,
        project_name=project_name,
        phase="clarification",
        summary=(
            f"The Team Lead has requested clarification on task {task_id}. "
            "You must respond with a clear, detailed answer."
        ),
    )

    description = f"""\
The Team Lead has requested clarification on task {task_id}: {task_title}

{state_block}

## Clarification Request
{clarification_message}

## Your Goal
Provide a clear, detailed response that addresses every point raised by the
Team Lead. If answering requires you to inspect code or search the codebase,
do so before responding.

## Step-by-Step Process

### Step 1: Understand the Question
Read the clarification request carefully. Identify every specific question
or concern that needs to be addressed.

### Step 2: Gather Context (if needed)
If the Team Lead's questions require looking at code:
- Use **read_file** to read relevant files.
- Use **code_search** to find usages or patterns related to the question.
- Use **list_directory** to check project structure if relevant.

### Step 3: Respond
Use **send_message** to send a structured response to the Team Lead:
- Address each question or concern individually.
- Be specific — reference file paths, line numbers, and code patterns.
- If you are uncertain about something, say so clearly rather than guessing.
- If the clarification changes your implementation plan, describe how.
"""

    expected_output = """\
A confirmation that a detailed clarification response was sent to the Team
Lead via send_message, addressing every point raised in the original
request.
"""
    return description, expected_output


def rework_code(
    task_id: int,
    rejection_count: int,
    latest_feedback: str,
    branch_name: str,
    project_id: int = 0,
    project_name: str = "",
) -> tuple[str, str]:
    """Return (description, expected_output) for reworking rejected code."""

    state_block = build_state_context(
        project_id=project_id,
        project_name=project_name,
        phase="rework",
        summary=(
            f"Code for task {task_id} was rejected (attempt {rejection_count})."
            " You must address all reviewer feedback."
        ),
        extra={"Rework attempt": str(rejection_count)},
    )

    description = f"""\
Your code for task {task_id} has been rejected (attempt {rejection_count}).

{state_block}

## Reviewer Feedback
{latest_feedback}

## Branch
`{branch_name}`

## Workspace
Your shell commands run in your isolated git worktree by default.
Do NOT `cd` to other directories — just run commands directly.

## Your Goal
Address ALL issues identified by the Code Reviewer. The review cycle will
continue until the reviewer approves your code, so it is critical that you
resolve every blocking issue in this iteration.

## Step-by-Step Process

### Step 1: Understand the Feedback
Read the reviewer's feedback above carefully. For each issue, identify:
- **What** the problem is (file, line, specific code).
- **Why** it matters (bug, security, performance, maintainability).
- **How** the reviewer suggests fixing it.

Make a mental checklist of every issue that must be addressed.

### Step 2: Read the Current Code
Use **read_file** to read the current state of the files mentioned in
the feedback. Understand the code as it is now, not as you remember it.

Use **code_search** if you need to find related usages or verify the
scope of a change (e.g., if the reviewer says "this function is called
from multiple places" — verify which places).

### Step 3: Apply Fixes
For each issue in the feedback:
1. Make the specific fix requested.
2. Verify the fix addresses the root cause, not just the symptom.
3. Check that the fix does not introduce new issues.

**Applying fixes to existing files** (the common case in a rework):
1. Use **code_search** to locate the exact code to change.
2. Use **read_file** with `offset` and `limit` to read the relevant
   section.
3. Use **edit_file** to apply a surgical edit. Ensure `old_string` is
   unique within the file by including enough surrounding context.

**IMPORTANT**: In a rework you are almost always modifying existing files.
Never use **write_file** on an existing file — it would overwrite the
entire file, destroying prior changes the reviewer already evaluated. Always
use **edit_file** instead. Only use **write_file** if you need to
create a completely new file that does not exist yet.

### Step 4: Run Tests
Use **execute_command** to run the full test suite. ALL tests must pass.
If the reviewer identified missing tests, write them now.

### Step 5: Self-Review Against Feedback
Use **git_diff** to review your rework diff. Go through the reviewer's
feedback point by point and verify:
- [ ] Each blocking issue has been addressed.
- [ ] No new issues were introduced.
- [ ] Tests pass.
- [ ] No unrelated changes were added.

### Step 6: Commit and Push
Push to the SAME branch `{branch_name}` — do NOT create a new branch for
a rework. Concretely:
1. **git_commit** — stages all changes (`git add -A`) and commits
   (`git commit -m "<message>"`).
2. **git_push** with branch=`{branch_name}` — pushes to origin on the
   Forgejo server (`git push -u origin {branch_name}`).

Commit message example:
"Fix injection vulnerability and add input validation per review feedback."

### Step 7: Submit for Re-Review
Use **update_task_status** to set the status to `review_ready`.

## Critical Reminders
- Address ALL blocking issues — partial fixes will result in another
  rejection.
- Do NOT introduce new issues while fixing old ones.
- Do NOT argue with feedback silently by ignoring it — if you disagree,
  add a comment explaining your reasoning via **add_comment**.
- Run tests BEFORE submitting. Broken tests are an automatic rejection.
- This is rejection attempt {rejection_count}. The review will continue
  until the reviewer approves, so resolve every issue now.
"""

    expected_output = f"""\
A summary of the rework containing:

1. **Issues addressed**: For each piece of reviewer feedback, what was
   changed and how.
2. **Files modified**: List of files changed during rework.
3. **Tests**: New tests added (if any) and confirmation all tests pass.
4. **Self-review**: Confirmation that the diff was reviewed against all
   feedback points.
5. **Status**: Confirmation that the task was moved to `review_ready`.
"""
    return description, expected_output


def fix_ci_failures(
    task_id: int,
    ci_output: str,
    branch_name: str,
    attempt: int,
    max_attempts: int,
    project_id: int = 0,
    project_name: str = "",
) -> tuple[str, str]:
    """Return (description, expected_output) for fixing CI failures before review."""

    state_block = build_state_context(
        project_id=project_id,
        project_name=project_name,
        phase="ci_fix",
        summary=(
            f"CI gate failed for task {task_id} before code review."
            f" Fix attempt {attempt}/{max_attempts}."
        ),
        extra={"CI fix attempt": f"{attempt}/{max_attempts}"},
    )

    description = f"""\
The CI gate (automated test suite) failed for task {task_id} BEFORE your code
was sent to the reviewer.  You must fix the failures so CI passes, then the
code will proceed to review automatically.

{state_block}

## CI Failure Output
```
{ci_output}
```

## Branch
`{branch_name}`

## Workspace
Your shell commands run in your isolated git worktree by default.
Do NOT `cd` to other directories — just run commands directly.

## Step-by-Step Process

### Step 1: Analyze the CI Output
Read the test failure output above carefully. Identify:
- Which test(s) failed and why.
- Whether the failure is in YOUR code or in a test that needs updating.

### Step 2: Read the Relevant Code
Use **read_file** to read the files mentioned in the failure output.
Use **code_search** if you need to find related code.

### Step 3: Apply Fixes
Fix the root cause of each test failure. Common causes:
- Syntax errors or typos in new code.
- Missing imports or dependencies.
- Logic errors that cause assertions to fail.
- Tests that need updating to match new behavior.

Use **edit_file** for surgical edits — do NOT overwrite entire files.

### Step 4: Run Tests Locally
Use **execute_command** to run the test suite. ALL tests must pass.
Do not proceed until you see a clean test run.

### Step 5: Commit and Push
Push to the SAME branch `{branch_name}`:
1. **git_commit** — commit the fixes.
2. **git_push** with branch=`{branch_name}`.

## Critical Reminders
- This is CI fix attempt {attempt} of {max_attempts}. Fix ALL failures now.
- Do NOT introduce new issues while fixing existing ones.
- Run the full test suite before pushing — partial fixes waste cycles.
"""

    expected_output = f"""\
A summary containing:

1. **Failures fixed**: Which test failures were identified and how each was fixed.
2. **Files modified**: List of files changed.
3. **Test results**: Confirmation that ALL tests pass after the fix.
4. **Commit**: Confirmation that fixes were committed and pushed to `{branch_name}`.
"""
    return description, expected_output
