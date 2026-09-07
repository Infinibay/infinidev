# Background tasks

Use `/ps`, `/bg`, `/tasks`, or **Ctrl+B** to list commands started by the agent's
`run_in_background` tool. The selector shows their task IDs, descriptions, status,
and recent output. Choose a task with **Up/Down**, then **Enter**, or click it to open
a live output tab. `/ps bg-1` opens a known task directly; the aliases accept IDs too.

Output tabs combine captured stdout and stderr and refresh while the engine is idle.
Opening the same task again focuses its existing tab. The status includes the exit code
when the command finishes.

| Key | Action |
| --- | --- |
| Up/Down, PageUp/PageDown, mouse wheel | Scroll the output |
| Home | Go to the oldest retained output |
| End | Resume following new output |
| F2 | Return to chat, keeping the tab open |
| Ctrl+W | Close the tab, leaving the process running |

Each task retains up to 256 KiB of combined recent output; the tab indicates when older
output has been discarded. The streams appear in capture order. A child process that
buffers its output must flush it before Infinidev can display it. The viewer is not an
interactive terminal and does not send input to the process.

Tasks belong to the current Infinidev process. This command does not list arbitrary
system processes or research-team workers. In the classic text interface, the aliases
print a task list or a snapshot for the requested ID; live tabs require the TUI.
