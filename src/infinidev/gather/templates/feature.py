"""Fixed questions for feature tickets."""

from infinidev.gather.models import Question

QUESTIONS = [
    Question(
        id="existing_patterns",
        question="What existing features in the codebase are most similar to this new feature?",
        context_prompt=(
            "Search the codebase for features or implementations that are analogous "
            "to the one being requested. Look for:\n"
            "- Similar functionality already implemented\n"
            "- Patterns used for comparable features (how are they structured?)\n"
            "- Code that could be reused or extended\n\n"
            "Feature request:\n{ticket_description}"
        ),
    ),
    Question(
        id="integration_points",
        question="Where should this feature be integrated? What entry points, routers, or registries need modification?",
        context_prompt=(
            "Identify the integration points for this feature:\n"
            "- Entry points (main, CLI commands, API routes)\n"
            "- Registries, plugin systems, or config files that register new functionality\n"
            "- Initialization code that would need to know about the new feature\n"
            "- Import chains that need updating\n\n"
            "Feature request:\n{ticket_description}"
        ),
    ),
    Question(
        id="related_files",
        question="What existing files, classes, and functions will need to be modified or extended?",
        context_prompt=(
            "List ALL files that will likely need changes for this feature:\n"
            "- Files to modify (add new methods, extend classes, add parameters)\n"
            "- Files to create (new modules, new test files)\n"
            "- Configuration files that need updating\n"
            "For each file, read it and note what specifically needs to change.\n\n"
            "Feature request:\n{ticket_description}"
        ),
    ),
    Question(
        id="dependencies",
        question="What internal modules and external libraries does this feature depend on?",
        context_prompt=(
            "Check what dependencies are needed:\n"
            "- Internal modules that the feature will import/use\n"
            "- External libraries already in requirements (can be reused?)\n"
            "- New external libraries that might be needed\n"
            "- Check requirements.txt, pyproject.toml, or package.json\n\n"
            "Feature request:\n{ticket_description}"
        ),
    ),
    Question(
        id="test_patterns",
        question="How are similar features tested in this project? What test patterns should be followed?",
        context_prompt=(
            "Find tests for features similar to the one being requested:\n"
            "- What test framework is used?\n"
            "- What fixtures or helpers exist?\n"
            "- What's the naming convention for test files and functions?\n"
            "- How is test data set up and torn down?\n\n"
            "Feature request:\n{ticket_description}"
        ),
    ),
]
