"""Risk classification for the ``auto`` permission mode.

``auto`` is the default permission mode for shell commands and the code
interpreter. It auto-approves operations that can be *positively recognised*
as read-only and escalates everything else to the existing ``ask`` prompt.

The security posture is **allow-listed, not deny-listed**: a shell command is
"safe" only when every segment of it is a known read-only command. Anything
unknown, obfuscated (command substitution, redirects), or mutating falls
through to a confirmation prompt. Being too conservative only costs an extra
prompt; being too permissive could silently run something destructive — so
when in doubt we classify as risky.

The functions here are pure and side-effect free so they're cheap to unit
test. They return ``(safe: bool, reason: str)`` — ``reason`` is shown to the
user in the prompt when ``safe`` is False.
"""

from __future__ import annotations

import ast
import os
import re
import shlex

# ── shell command classification ─────────────────────────────────────────

# Constructs whose mere presence forces a prompt. We deliberately do NOT try
# to reason about what's inside a command substitution or a redirect target —
# `echo $(rm -rf ~)` and `cat secrets > /tmp/x` must both prompt.
#   $( … )  `( … )`  ${ … }-is-fine-but-$(-)-exec   <( )  >( )   process subst.
_SUBSTITUTION_RE = re.compile(r"\$\(|`|<\(|>\(")
# Output redirection (truncate/append/clobber) and here-strings/here-docs.
# A bare input redirect ``<`` only feeds a file to stdin (read-only) so it is
# allowed; ``>`` in any form mutates the filesystem.
_OUTPUT_REDIRECT_RE = re.compile(r">")

# Operators that separate a command line into independently-runnable segments.
# We split on them and require EVERY segment to be a safe read-only command.
_SEGMENT_SPLIT_RE = re.compile(r"\|\||&&|;|\||&|\n")

# Base commands that are read-only no matter what arguments they get. Listing,
# reading, searching, hashing, text-to-stdout transforms, and system/info
# queries. Notably absent: tee, dd, mount, env, xargs, sudo — these can mutate
# state or run other commands, so they fall through to a prompt.
SAFE_COMMANDS: frozenset[str] = frozenset({
    # listing / reading files
    "ls", "dir", "vdir", "cat", "bat", "head", "tail", "nl", "tac",
    "wc", "file", "stat", "readlink", "realpath", "basename", "dirname",
    "od", "hexdump", "xxd", "strings",
    "cksum", "sum", "md5sum", "sha1sum", "sha224sum", "sha256sum",
    "sha384sum", "sha512sum", "b2sum", "shasum", "md5",
    # search
    "grep", "egrep", "fgrep", "rg", "ag", "ack", "fd", "fdfind", "find",
    "which", "whereis", "type",
    # text transforms to stdout (read-only)
    "echo", "printf", "cut", "sort", "uniq", "comm", "join", "paste",
    "column", "fold", "fmt", "expand", "unexpand", "tr", "rev",
    "diff", "cmp", "colordiff", "jq", "yq",
    # system / env info
    "pwd", "whoami", "id", "groups", "hostname", "uname", "arch",
    "date", "cal", "uptime", "printenv", "locale", "tty", "getconf",
    "nproc", "free", "vmstat", "iostat", "getent",
    # process / disk info
    "ps", "pgrep", "jobs", "df", "du", "lsblk", "lsof", "lscpu",
    "lsusb", "lspci", "top", "htop",
    # navigation / misc harmless
    "true", "false", "test", "seq", "sleep", "tree", "clear", "tput",
    "man", "whatis", "apropos", "info", "tldr", "help",
})

# Flags that turn an otherwise read-only command into a mutating one. Keyed by
# base command. If any of these tokens appear, the command is risky.
_MUTATING_FLAGS: dict[str, frozenset[str]] = {
    # ``find … -delete`` / ``-exec rm`` etc. run arbitrary side effects.
    "find": frozenset({
        "-delete", "-exec", "-execdir", "-ok", "-okdir",
        "-fls", "-fprint", "-fprint0", "-fprintf",
    }),
    # ``sort -o file`` / ``--output`` writes a file without a ``>`` redirect.
    "sort": frozenset({"-o", "--output"}),
    # ``yq -i`` / ``--inplace`` rewrites the target file in place (no ``>``).
    "yq": frozenset({"-i", "--inplace", "--in-place"}),
    # ``tree -o file`` / ``--output file`` writes the listing to a file.
    "tree": frozenset({"-o", "--output"}),
}

# Environment-assignment prefixes that are NOT inert: they alter how the
# following "safe" binary loads or resolves code, turning e.g. ``cat`` into an
# arbitrary-code-execution vector. Any leading ``VAR=value`` using one of these
# (or the ``LD_*`` / ``DYLD_*`` / ``BASH_FUNC_*`` families) forces a prompt.
_FORBIDDEN_ENV_PREFIX: frozenset[str] = frozenset({
    "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "LD_PROFILE", "LD_DEBUG",
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH",
    "BASH_ENV", "ENV", "PATH", "IFS", "PROMPT_COMMAND", "GLOBIGNORE",
    "PYTHONSTARTUP", "PYTHONPATH", "PERL5OPT", "PERL5LIB", "RUBYOPT",
    "NODE_OPTIONS", "SHELLOPTS", "BASH_FUNC",
})

# Subcommand-gated tools: safe only when the subcommand is read-only. Git is
# handled separately (it has flag-sensitive subcommands and global options).
# NOTE: ``config`` is deliberately EXCLUDED for pip/poetry/kubectl — it is a
# namespace, not a read-only leaf (``pip config set`` / ``kubectl config
# use-context`` / ``poetry config k v`` all mutate persistent state). ``go env``
# is allowed but flag-gated below (``go env -w`` writes). ``npm repo``/``docs``
# are excluded — they launch the system browser/opener (an external program).
_SUBCOMMAND_SAFE: dict[str, frozenset[str]] = {
    "pip": frozenset({"list", "show", "freeze", "check", "debug", "inspect"}),
    "pip3": frozenset({"list", "show", "freeze", "check", "debug", "inspect"}),
    "npm": frozenset({
        "ls", "list", "view", "info", "outdated", "ping",
        "root", "prefix", "bin", "why", "explain", "help",
    }),
    "pnpm": frozenset({"ls", "list", "view", "info", "outdated", "why", "root", "bin"}),
    "yarn": frozenset({"list", "info", "why", "versions"}),
    "cargo": frozenset({"tree", "metadata", "search", "verify-project"}),
    "go": frozenset({"list", "version", "env", "doc", "vet"}),
    "docker": frozenset({
        "ps", "images", "version", "info", "inspect", "logs",
        "port", "top", "history", "stats", "events", "diff", "search",
    }),
    "kubectl": frozenset({
        "get", "describe", "logs", "version", "top", "explain",
        "api-resources", "api-versions", "cluster-info",
    }),
    "brew": frozenset({"list", "info", "search", "outdated", "deps", "config", "--version"}),
    "systemctl": frozenset({"status", "show", "list-units", "list-unit-files", "is-active", "is-enabled", "cat"}),
    "poetry": frozenset({"show", "check", "version", "env", "about"}),
}
# Per-(tool, subcommand) write flags: a read-ish subcommand that a flag turns
# into a write. ``go env -w/-u`` rewrites the persistent go environment.
_SUBCOMMAND_WRITE_FLAGS: dict[tuple[str, str], frozenset[str]] = {
    ("go", "env"): frozenset({"-w", "-u"}),
}

# Flags that, when they are the ONLY arguments to any command, are read-only
# version/help queries — e.g. ``docker --version``, ``tsc --version``.
_VERSION_HELP_FLAGS: frozenset[str] = frozenset({"--version", "--help"})

# Git subcommands that only read repository state.
_GIT_READONLY: frozenset[str] = frozenset({
    "status", "diff", "log", "show", "blame", "annotate",
    "ls-files", "ls-tree", "ls-remote", "cat-file", "rev-parse", "rev-list",
    "describe", "shortlog", "reflog", "whatchanged", "name-rev",
    "show-ref", "symbolic-ref", "merge-base", "for-each-ref", "grep",
    "count-objects", "var", "cherry", "range-diff", "diff-tree",
    "diff-index", "diff-files", "verify-commit", "verify-tag", "version",
})
# Flags that turn ``git branch``/``git tag`` from a list into a mutation.
_GIT_BRANCH_WRITE_FLAGS: frozenset[str] = frozenset({
    "-d", "-D", "-m", "-M", "-c", "-C", "--delete", "--move", "--copy",
    "--edit-description", "--set-upstream-to", "-u", "--unset-upstream",
    "-f", "--force",
})
_GIT_TAG_WRITE_FLAGS: frozenset[str] = frozenset({
    "-d", "--delete", "-a", "-s", "-m", "-f", "--force", "--annotate", "--sign",
})
# Flags that turn ``git config`` from a read into a write.
_GIT_CONFIG_WRITE_FLAGS: frozenset[str] = frozenset({
    "--unset", "--unset-all", "--add", "--replace-all",
    "--remove-section", "--rename-section", "-e", "--edit",
})
# Subcommands that share a name between read and write forms but whose write
# form needs a mutating sub-action. Map subcommand → read-only sub-actions; the
# first positional after the subcommand must be in this set.
_GIT_SUBACTION_READ: dict[str, frozenset[str]] = {
    "stash": frozenset({"list", "show"}),       # bare ``git stash`` == push (mutates)
    "submodule": frozenset({"status", "summary"}),
    "notes": frozenset({"list", "show"}),
    "worktree": frozenset({"list"}),
    "remote": frozenset({"show", "get-url"}),
}
# Subcommands whose bare (no-positional) form merely lists state.
_GIT_SUBACTION_BARE_OK: frozenset[str] = frozenset({"submodule", "notes", "remote"})
# Git global options that take a value (so we skip the following token when
# locating the subcommand). ``git -C /path status`` → subcommand is ``status``.
_GIT_GLOBAL_VALUE_OPTS: frozenset[str] = frozenset({
    "-C", "-c", "--git-dir", "--work-tree", "--namespace",
    "--super-prefix", "--exec-path", "--config-env",
})


def _base_name(token: str) -> str:
    """Strip a leading path from an executable token: ``/bin/ls`` → ``ls``.

    A relative path like ``./build.sh`` becomes ``build.sh`` which won't match
    the safe set, so script execution still prompts.
    """
    return os.path.basename(token)


def _consume_env_assignments(tokens: list[str]) -> tuple[list[str], str | None]:
    """Drop leading ``VAR=value`` assignment tokens, flagging dangerous ones.

    ``FOO=bar grep x f`` runs grep with FOO set — a benign assignment is inert,
    so the real base command is the first non-assignment token. But assignments
    to loader/shell-control variables (``LD_PRELOAD``, ``DYLD_*``, ``BASH_ENV``,
    ``PATH``, ``IFS``, ``BASH_FUNC_*`` …) inject code into the "safe" binary, so
    they must escalate. Returns ``(remaining_tokens, forbidden_var | None)``.
    """
    i = 0
    while i < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[i]):
        var = tokens[i].split("=", 1)[0]
        if (var in _FORBIDDEN_ENV_PREFIX
                or var.startswith(("LD_", "DYLD_", "BASH_FUNC_"))):
            return tokens[i:], var
        i += 1
    return tokens[i:], None


def _git_subcommand(tokens: list[str]) -> str | None:
    """Return the git subcommand, skipping global options and their values."""
    i = 1  # tokens[0] is "git"
    while i < len(tokens):
        tok = tokens[i]
        if tok in _GIT_GLOBAL_VALUE_OPTS:
            i += 2  # skip the option and its value
            continue
        if tok.startswith("-"):
            i += 1  # self-contained flag (e.g. --no-pager, --bare, -c key=val=…)
            continue
        return tok
    return None


def _classify_git(tokens: list[str]) -> tuple[bool, str]:
    # Reject inline-config / exec-path global options BEFORE locating the
    # subcommand: ``git -c core.pager='sh -c id'`` / ``-c core.fsmonitor=…`` /
    # ``--config-env`` / ``--exec-path`` inject execution-bearing config that
    # turns an otherwise read-only git command into arbitrary code execution.
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("-c", "--config-env", "--exec-path") or \
                tok.startswith(("--config-env=", "--exec-path=")):
            return False, "git -c/--config-env/--exec-path can inject executable config"
        if tok in _GIT_GLOBAL_VALUE_OPTS:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        break  # reached the subcommand

    sub = _git_subcommand(tokens)
    if sub is None:
        # bare ``git`` (prints help) — harmless.
        return True, ""
    if sub in _GIT_READONLY:
        return True, ""

    after = tokens[tokens.index(sub) + 1:]
    flags = [t for t in after if t.startswith("-")]
    positionals = [t for t in after if not t.startswith("-")]

    if sub == "branch":
        if any(f in _GIT_BRANCH_WRITE_FLAGS for f in flags):
            return False, "git branch with a write flag modifies refs"
        if positionals:
            return False, "git branch with an argument creates/modifies a ref"
        return True, ""
    if sub == "tag":
        if any(f in _GIT_TAG_WRITE_FLAGS for f in flags):
            return False, "git tag with a write flag modifies refs"
        if positionals:
            return False, "git tag with an argument creates/deletes a tag"
        return True, ""
    if sub == "config":
        if any(f in _GIT_CONFIG_WRITE_FLAGS for f in flags):
            return False, "git config with a write flag changes config"
        # ``git config key value`` (two positionals) writes; one positional is
        # a read (``git config key`` / ``git config --get key``).
        if len(positionals) >= 2:
            return False, "git config with a value writes config"
        return True, ""
    if sub in _GIT_SUBACTION_READ:
        read_actions = _GIT_SUBACTION_READ[sub]
        if not positionals:
            if sub in _GIT_SUBACTION_BARE_OK:
                return True, ""
            return False, f"git {sub} with no sub-action may modify state"
        if positionals[0] in read_actions:
            return True, ""
        return False, f"git {sub} {positionals[0]} is not a read-only action"

    return False, f"git {sub} is not a known read-only subcommand"


def _classify_segment(segment: str) -> tuple[bool, str]:
    segment = segment.strip()
    if not segment:
        return True, ""  # empty segment (e.g. trailing operator) is inert
    if _OUTPUT_REDIRECT_RE.search(segment):
        return False, "writes to a file via output redirection"
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return False, "could not be parsed as a simple command"
    tokens, forbidden = _consume_env_assignments(tokens)
    if forbidden:
        return False, f"sets {forbidden} (can inject code into the command)"
    if not tokens:
        # only env assignments, no command — inert
        return True, ""
    base = _base_name(tokens[0])

    if base == "command":
        # POSIX ``command`` is a transparent exec wrapper (``command rm -rf ~``
        # runs rm, bypassing functions/aliases). Only the read-only lookup form
        # (``command -v/-V NAME``) is safe; otherwise classify the wrapped
        # command itself so it inherits the same allow-list rules.
        rest = tokens[1:]
        if any(t in ("-v", "-V") for t in rest):
            return True, ""  # lookup, runs nothing
        j = 0
        while j < len(rest) and rest[j] == "-p":
            j += 1
        inner = rest[j:]
        if not inner:
            return True, ""  # bare ``command`` / flags only
        return _classify_segment(shlex.join(inner))

    if base == "git":
        return _classify_git(tokens)

    if base in SAFE_COMMANDS:
        bad = _MUTATING_FLAGS.get(base)
        if bad:
            for tok in tokens[1:]:
                # match ``-o`` and ``-o=file``/``--output=...`` forms
                flag = tok.split("=", 1)[0]
                if flag in bad:
                    return False, f"{base} {flag} can write/execute"
        return True, ""

    safe_subs = _SUBCOMMAND_SAFE.get(base)
    if safe_subs is not None:
        sub = next((t for t in tokens[1:] if not t.startswith("-")), None)
        if sub is None:
            # only flags: allow version/help queries, else prompt
            if all(t in _VERSION_HELP_FLAGS for t in tokens[1:]):
                return True, ""
            return False, f"{base} with no subcommand"
        if sub in safe_subs:
            write_flags = _SUBCOMMAND_WRITE_FLAGS.get((base, sub))
            if write_flags:
                for tok in tokens[1:]:
                    if tok.split("=", 1)[0] in write_flags:
                        return False, f"{base} {sub} {tok} writes persistent config"
            return True, ""
        return False, f"{base} {sub} is not a known read-only subcommand"

    # Any command, when invoked only with --version/--help, is a read-only query.
    if tokens[1:] and all(t in _VERSION_HELP_FLAGS for t in tokens[1:]):
        return True, ""

    return False, f"'{base}' is not a recognised read-only command"


def classify_command(command: str) -> tuple[bool, str]:
    """Classify a shell command for ``auto`` mode.

    Returns ``(True, "")`` when the command is provably read-only and may run
    without a prompt, or ``(False, reason)`` when it should be escalated to the
    user. The classifier is intentionally conservative: unknown commands,
    command substitution, output redirection, and any mutating flag all yield
    ``(False, …)``.
    """
    if command is None:
        return False, "empty command"
    command = command.strip()
    if not command:
        return False, "empty command"

    # Command/process substitution anywhere hides arbitrary execution.
    if _SUBSTITUTION_RE.search(command):
        return False, "contains command/process substitution"

    for segment in _SEGMENT_SPLIT_RE.split(command):
        safe, reason = _classify_segment(segment)
        if not safe:
            return False, reason or "contains a non-read-only command"
    return True, ""


# ── python code classification (code interpreter) ─────────────────────────

# Modules whose import signals the code may write files, spawn processes, or
# reach the network. ``os`` is intentionally absent — it's pervasive in
# read-only analysis (os.path, os.getcwd, os.listdir); specific dangerous
# ``os.*`` calls are flagged below instead.
_PY_RISKY_MODULES: frozenset[str] = frozenset({
    "subprocess", "socket", "shutil", "ctypes", "cffi", "pty",
    "multiprocessing", "urllib", "requests", "httpx", "aiohttp",
    "ftplib", "smtplib", "telnetlib", "poplib", "imaplib", "paramiko",
    "http", "pickle", "dill", "shelve", "marshal", "webbrowser",
    # dynamic import machinery can resolve to any of the above by name
    "importlib", "imp", "runpy",
    # builtins.exec/eval reach arbitrary code; tempfile.* writes files to disk
    "builtins", "tempfile",
})
# Builtins that enable dynamic dispatch / namespace access, which can reach
# os.system/exec/etc. without those names ever appearing literally.
_PY_RISKY_DISPATCH: frozenset[str] = frozenset({
    "getattr", "setattr", "delattr", "globals", "vars", "locals", "__import__",
})
# Dangerous ``os.<attr>`` calls (filesystem mutation / process control).
_PY_RISKY_OS: frozenset[str] = frozenset({
    "system", "popen", "remove", "unlink", "rmdir", "removedirs",
    "rename", "renames", "replace", "mkdir", "makedirs", "chmod", "chown",
    "lchown", "truncate", "ftruncate", "link", "symlink", "mkfifo", "mknod",
    "kill", "killpg", "fork", "forkpty", "setuid", "setgid",
    "execv", "execve", "execl", "execle", "execlp", "execvp", "execvpe",
    "spawnl", "spawnv", "spawnve", "spawnvp", "open", "write",
})
# Dangerous ``shutil.<attr>`` calls.
_PY_RISKY_SHUTIL: frozenset[str] = frozenset({
    "rmtree", "move", "copy", "copy2", "copyfile", "copytree",
    "make_archive", "unpack_archive", "chown",
})
# Distinctive method names that essentially always mean a filesystem mutation,
# regardless of the receiver (pathlib / os-like APIs). Generic names such as
# ``write``/``run``/``get``/``rename`` are excluded — they collide with
# StringIO, pandas, dict, etc. and would false-positive constantly.
_PY_RISKY_METHODS: frozenset[str] = frozenset({
    "write_text", "write_bytes", "unlink", "rmtree", "rmdir",
    "symlink_to", "hardlink_to", "makedirs",
})
# Builtins that execute arbitrary code.
_PY_RISKY_BUILTINS: frozenset[str] = frozenset({"eval", "exec", "compile", "__import__"})


def _open_mode_is_write(node: ast.Call, mode_index: int) -> bool:
    """Decide whether an open-style call opens for writing.

    ``mode_index`` is the positional index of the mode argument: ``1`` for the
    builtin / module form ``open(file, mode)`` (and ``io.open``/``codecs.open``),
    ``0`` for the method form ``path.open(mode)``. Returns True for explicit
    write/append/exclusive/update modes and — to stay conservative — for any
    non-literal mode we cannot inspect.
    """
    mode_node = None
    if len(node.args) > mode_index:
        mode_node = node.args[mode_index]
    else:
        for kw in node.keywords:
            if kw.arg == "mode":
                mode_node = kw.value
                break
    if mode_node is None:
        return False  # defaults to 'r'
    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
        return any(c in mode_node.value for c in ("w", "a", "x", "+"))
    return True  # mode is dynamic — cannot prove it's read-only


def classify_python(code: str) -> tuple[bool, str]:
    """Classify code-interpreter Python for ``auto`` mode.

    Walks the AST and flags filesystem mutation, process spawning, network
    access, and arbitrary-code execution. Pure analysis/read code (json, ast,
    pandas, the pre-imported read-only code-intel helpers, ``open(...)`` for
    reading) is classified safe. On a syntax error the code cannot be proven
    read-only, so it is classified risky (it would fail to run anyway, but a
    prompt is the safe default).

    The check resolves module aliases (``import os as o``), attribute rebinds
    (``s = os.system``), and dangerous from-imports (``from os import system``)
    so the qualified-call detection can't be sidestepped by rebinding names, and
    treats dynamic dispatch (``getattr``/``globals``/…) as not-provably-read-only.

    Known limitations (accepted): a builtin laundered through a function's
    return value — ``f()('x', 'w')`` where ``f`` returns ``open`` — or reached
    via a deep attribute chain — ``import os.path as p; p.os.system(...)`` — is
    not tracked, since the call target is not a statically-resolvable name.
    Flagging every call-of-a-call would false-positive on ordinary higher-order
    code; the real isolation boundary for adversarial code is ``SANDBOX_ENABLED``,
    not this convenience gate.
    """
    if not code or not code.strip():
        return True, ""  # nothing to run
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, "code could not be parsed; cannot verify it is read-only"

    # ── pass 1a: imports & alias resolution ──
    os_names = {"os"}            # local names bound to the os module
    shutil_names = {"shutil"}    # local names bound to shutil
    iocodecs_names = {"io", "codecs"}  # local names bound to io / codecs
    open_names = {"open"}        # local names bound to a write-capable open()
    dispatch_aliases: set[str] = set()  # names rebound to a risky builtin/dispatch
    risky_bare: set[str] = set()  # names bound to a dangerous module attr (s=os.system)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _PY_RISKY_MODULES:
                    return False, f"imports '{alias.name}'"
                # Only a top-level ``import os``/``import os as o`` binds the
                # name to the module itself; ``import os.path as p`` binds p to
                # the submodule, so don't register it (p.os.* re-entry aside).
                if "." in alias.name:
                    continue
                bound = alias.asname or root
                if root == "os":
                    os_names.add(bound)
                elif root == "shutil":
                    shutil_names.add(bound)
                elif root in ("io", "codecs"):
                    iocodecs_names.add(bound)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _PY_RISKY_MODULES:
                return False, f"imports from '{node.module}'"
            # ``from os import *`` brings in system/popen/etc. under bare names
            # we can't enumerate — refuse it.
            if node.module in ("os", "shutil") and any(a.name == "*" for a in node.names):
                return False, f"star-imports from '{node.module}'"
            # ``from os import system`` / ``from shutil import rmtree`` bind a
            # dangerous callable to a bare name the call-site checks would miss
            # (they only know os.<attr>/shutil.<attr>). Flag at the import.
            if node.module == "os":
                for alias in node.names:
                    if alias.name in _PY_RISKY_OS:
                        return False, f"imports os.{alias.name}"
            elif node.module == "shutil":
                for alias in node.names:
                    if alias.name in _PY_RISKY_SHUTIL:
                        return False, f"imports shutil.{alias.name}"
            elif root in ("io", "codecs"):
                for alias in node.names:
                    if alias.name == "open":
                        open_names.add(alias.asname or "open")

    # ── pass 1b: assignment rebinds (run after imports so aliases are known) ──
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        val = node.value
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not targets:
            continue
        # ``g = getattr`` / ``e = exec`` / ``o = open`` — bare-name rebind.
        if isinstance(val, ast.Name):
            if val.id in open_names:
                open_names.update(targets)
            elif val.id in (_PY_RISKY_DISPATCH | _PY_RISKY_BUILTINS):
                dispatch_aliases.update(targets)
            elif val.id in risky_bare:
                risky_bare.update(targets)
        # ``s = os.system`` / ``rt = shutil.rmtree`` — attribute rebind.
        elif isinstance(val, ast.Attribute) and isinstance(val.value, ast.Name):
            mod, attr = val.value.id, val.attr
            if (mod in os_names and attr in _PY_RISKY_OS) \
                    or (mod in shutil_names and attr in _PY_RISKY_SHUTIL):
                risky_bare.update(targets)

    # ── pass 2: calls ──
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in _PY_RISKY_BUILTINS:
                return False, f"calls {func.id}()"
            if func.id in _PY_RISKY_DISPATCH or func.id in dispatch_aliases:
                return False, f"calls {func.id}() (dynamic dispatch can reach arbitrary code)"
            if func.id in risky_bare:
                return False, f"calls {func.id}() (rebound dangerous callable)"
            if func.id in open_names and _open_mode_is_write(node, 1):
                return False, "opens a file for writing"
        elif isinstance(func, ast.Attribute):
            attr = func.attr
            if attr in _PY_RISKY_METHODS:
                return False, f"calls .{attr}() (filesystem mutation)"
            if attr == "open":
                # io.open / codecs.open use the builtin signature (mode at
                # arg 1); the method form ``path.open(mode)`` has mode at arg 0.
                if isinstance(func.value, ast.Name) and func.value.id in iocodecs_names:
                    if _open_mode_is_write(node, 1):
                        return False, "opens a file for writing"
                elif _open_mode_is_write(node, 0):
                    return False, "opens a file for writing via .open()"
            if isinstance(func.value, ast.Name):
                mod = func.value.id
                if mod in os_names and attr in _PY_RISKY_OS:
                    return False, f"calls os.{attr}()"
                if mod in shutil_names and attr in _PY_RISKY_SHUTIL:
                    return False, f"calls shutil.{attr}()"

    return True, ""
