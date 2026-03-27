"""Shared task prompts used across multiple roles."""

GIT_WORKFLOW_INSTRUCTIONS = """
## Git + Forgejo Workflow — MANDATORY RULES

Remote `origin` points to Forgejo (e.g. http://localhost:3000/{owner}/{repo}.git).
If not configured, report to the Team Lead — do NOT configure it yourself.
NEVER change the remote URL.

### CREATE BRANCH
**git_branch**(branch_name=`task-{task_id}-<slug>`, create=true, base_branch="main")
→ `git fetch origin main && git checkout -b task-{task_id}-<slug> origin/main`
- Slug: lowercase, letters/digits/hyphens only. Example: `task-42-add-auth-endpoint`
- Do NOT skip this step. Do NOT commit directly to main.

### WRITE CODE
**edit_symbol** or **replace_lines** for existing files, **create_file** for new files only.
create_file fails if the file already exists — use replace_lines or edit_symbol to modify.

### COMMIT
**git_commit**(message="<imperative verb> <what changed> — task {task_id}")
→ `git add -A && git commit -m "..."`
Example: "Add JWT validation middleware — task 42"
- Run all tests BEFORE committing. Do NOT commit if any test fails.

### PUSH
**git_push**(branch="<branch name>", force=false) → `git push -u origin <branch>`
- If rejected ("remote has new commits"): run `git pull origin main --rebase`, then retry.
- Do NOT use force=true unless the Team Lead explicitly instructs it.


"""


def brainstorm_round(
    round_count: int,
    project_name: str,
    project_description: str,
    project_type: str,
    existing_ideas: str = "",
    user_feedback: str = "",
) -> tuple[str, str]:
    """Return (description, expected_output) for a brainstorming round.

    Used by team_lead, developer, and researcher agents.
    """
    sections = [
        f"Brainstorming session (round {round_count}).\n",
        "## Project Context\n"
        f"- **Name:** {project_name}\n"
        f"- **Description:** {project_description}\n"
        f"- **Type:** {project_type}\n",
    ]

    if user_feedback:
        sections.append(
            "## User Feedback from Previous Round\n"
            "The user rejected the previous set of ideas with this feedback:\n"
            f"> {user_feedback}\n\n"
            "You MUST address every point raised in the feedback above.\n"
        )

    if existing_ideas:
        sections.append(
            "## Previously Proposed Ideas (do NOT repeat these)\n"
            f"{existing_ideas}\n"
        )

    sections.append(
        "## Your Task\n"
        "1. Read the project context carefully.\n"
        "2. If user feedback is present above, list the specific concerns you will address.\n"
        "3. Propose exactly 3 new ideas that are distinct from existing ones.\n"
        "4. For each idea, output the block below **exactly** — no extra text before or after the block.\n"
        "\n"
        "## Mandatory Output Format\n"
        "\n"
        "```\n"
        "## Idea 1\n"
        "**Title:** <one-line title>\n"
        "**Description:** <2-3 sentences explaining the idea>\n"
        "**Impact:** <one sentence on expected benefit>\n"
        "**Feasibility:** <one sentence: High / Medium / Low and why>\n"
        "\n"
        "## Idea 2\n"
        "...\n"
        "\n"
        "## Idea 3\n"
        "...\n"
        "```\n"
        "\n"
        "## Constraints\n"
        "- Do not add any text outside the `## Idea N` blocks.\n"
        "- Do not number fields differently.\n"
        "- Do not merge fields.\n"
        "- If you cannot think of 3 distinct ideas, output fewer blocks rather than repeating existing ones.\n"
    )

    description = "\n".join(sections)
    expected_output = (
        "Exactly 3 Markdown blocks, each starting with `## Idea N`, "
        "containing **Title**, **Description**, **Impact**, and **Feasibility** "
        "fields in that order. No other text."
    )
    return description, expected_output
