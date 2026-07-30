"""The composer: the framed input at the bottom of the transcript.

Two pieces the old layout lacked:

* a **rounded frame** drawn around the input, so the place you type is a
  distinct object rather than a coloured strip glued to the transcript;
* a **ghost placeholder** rendered on the cursor's own line instead of on
  a separate line above it — the previous approach made the caret sit one
  row below the hint and jump upward on the first keystroke.

Both are pure prompt_toolkit primitives, so everything else (autocomplete
float, shell mode, paste, attachments) keeps working untouched.
"""

from __future__ import annotations

from collections.abc import Callable

from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.layout.processors import Processor, Transformation

from infinidev.ui.theme import (
    CHAT_INPUT_HEIGHT,
    COMPOSER_BORDER,
    COMPOSER_BORDER_FOCUS,
    COMPOSER_PLACEHOLDER,
    COMPOSER_PROMPT,
    SHELL_BORDER_COLOR,
    SHELL_INPUT_FG,
    SHELL_LABEL_FG,
)

PROMPT_MARK = "› "
SHELL_MARK = "! "


class PlaceholderProcessor(Processor):
    """Render ghost text on line 0 while the buffer is empty.

    prompt_toolkit exposes placeholders only through ``PromptSession``; a
    hand-built ``BufferControl`` needs this. Applying it as a processor —
    rather than swapping in another control — keeps the caret where it
    belongs: at the start of the hint, on the same row.
    """

    def __init__(self, text: str, style: str) -> None:
        self.text = text
        self.style = style

    def apply_transformation(self, transformation_input) -> Transformation:
        document = transformation_input.document
        if transformation_input.lineno == 0 and not document.text:
            return Transformation([(self.style, self.text)])
        return Transformation(transformation_input.fragments)


def build_composer(
    input_control,
    *,
    is_shell_mode: Callable[[], bool],
    placeholder: str = "Ask anything, or / for commands",
    height: int = CHAT_INPUT_HEIGHT,
) -> HSplit:
    """Wrap *input_control* in a rounded, mode-aware frame."""

    if not any(
        isinstance(processor, PlaceholderProcessor)
        for processor in (getattr(input_control, "input_processors", None) or [])
    ):
        processors = list(getattr(input_control, "input_processors", None) or [])
        processors.append(
            PlaceholderProcessor(placeholder, COMPOSER_PLACEHOLDER)
        )
        input_control.input_processors = processors

    def _border_style() -> str:
        if is_shell_mode():
            return SHELL_BORDER_COLOR
        return COMPOSER_BORDER_FOCUS if _has_focus(input_control) else COMPOSER_BORDER

    def _edge(left: str, right: str, *, label: bool = False):
        """A horizontal rule with rounded corners, fitted to the container.

        Built from three windows rather than one formatted string: a
        ``Window`` with ``char="─"`` expands to whatever width the layout
        hands it, so the frame stays flush when the sidebar or explorer
        opens. Measuring the *terminal* instead would draw an 80-column
        rule inside a 60-column column.
        """
        cells = [
            Window(width=1, char=left, style=_border_style),
            Window(char="─", style=_border_style),
            Window(width=1, char=right, style=_border_style),
        ]
        if label:
            cells.insert(
                1,
                ConditionalContainer(
                    content=Window(
                        content=FormattedTextControl(
                            lambda: [(f"{SHELL_LABEL_FG} bold", " shell ")]
                        ),
                        width=7,
                    ),
                    filter=Condition(is_shell_mode),
                ),
            )
        return VSplit(cells, height=1)

    def _mark_fragments():
        style = SHELL_LABEL_FG if is_shell_mode() else COMPOSER_PROMPT
        mark = SHELL_MARK if is_shell_mode() else PROMPT_MARK
        # Leading space: the mark must not touch the frame's left edge.
        return [("", " "), (f"{style} bold", mark)]

    def _input_style() -> str:
        return SHELL_INPUT_FG if is_shell_mode() else ""

    body = VSplit(
        [
            Window(width=1, char="│", style=_border_style),
            Window(content=FormattedTextControl(_mark_fragments), width=3),
            # Grows with what is typed instead of reserving rows nobody is
            # using: one line at rest, up to `height` for a long prompt.
            # ``dont_extend_height`` is the part that matters — without it
            # the enclosing HSplit hands the composer all the slack it can
            # take and it sits at full height with empty rows under the
            # caret.
            Window(
                content=input_control,
                height=D(min=1, max=height),
                dont_extend_height=True,
                style=_input_style,
                wrap_lines=True,
            ),
            Window(width=1, char="│", style=_border_style),
        ]
    )

    return HSplit([_edge("╭", "╮", label=True), body, _edge("╰", "╯")])


def _has_focus(control) -> bool:
    try:
        from prompt_toolkit.application.current import get_app

        return get_app().layout.current_control is control
    except Exception:
        return False


def composer_container(app_state, input_control) -> ConditionalContainer:
    """The composer, visible only on the chat tab."""

    def _is_shell_mode() -> bool:
        return app_state._chat_buffer.text.startswith("!")

    return ConditionalContainer(
        content=build_composer(input_control, is_shell_mode=_is_shell_mode),
        filter=Condition(lambda: app_state.active_tab == "chat"),
    )
