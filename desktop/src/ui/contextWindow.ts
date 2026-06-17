/**
 * Best-effort context-window sizes for common models, so the UI can show how
 * full the context is. Mirrors the Python TUI's context_calculator table; when
 * a model isn't recognized we fall back to a conservative default rather than
 * pretend to know. Matching is by case-insensitive substring on the model id.
 */
const WINDOWS: [match: string, tokens: number][] = [
  // Anthropic
  ["claude", 200_000],
  // OpenAI
  ["gpt-4o", 128_000],
  ["gpt-4.1", 1_000_000],
  ["o1", 200_000],
  ["o3", 200_000],
  ["gpt-4-turbo", 128_000],
  ["gpt-4", 8_192],
  ["gpt-3.5", 16_385],
  // Google
  ["gemini-1.5", 1_000_000],
  ["gemini", 32_768],
  // Local / open-weight
  ["qwen2.5-coder", 32_768],
  ["qwen2.5", 32_768],
  ["qwen", 32_768],
  ["deepseek", 64_000],
  ["llama3.1", 128_000],
  ["llama3", 8_192],
  ["llama", 8_192],
  ["gemma", 8_192],
  ["mistral", 32_768],
  ["mixtral", 32_768],
  ["phi", 16_384],
  ["codellama", 16_384],
];

const DEFAULT_WINDOW = 32_768;

/** The context window (in tokens) for `model`, or a conservative default. */
export function contextWindow(model: string): number {
  const m = model.toLowerCase();
  for (const [match, tokens] of WINDOWS) {
    if (m.includes(match)) return tokens;
  }
  return DEFAULT_WINDOW;
}
