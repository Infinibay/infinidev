import { test, expect } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.goto("/#token=browser-test-token");
  await expect(page.getByText("Connecting to the workspace…")).toHaveCount(0);
  await expect(
    page.getByText("Investigate memory routing", { exact: true }).first(),
  ).toBeVisible();
});

test("workspace, team, notes, tools, logs and source are real API views", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await expect(
    page.getByText("Investigate memory routing", { exact: true }).first(),
  ).toBeVisible();
  await expect(
    page.getByText("I’ve organized the investigation", { exact: false }),
  ).toBeVisible();
  await expect(page.getByText("Mara", { exact: true }).first()).toBeVisible();
  await page.screenshot({ path: "test-results/workspace.png", fullPage: true });
  await page
    .getByRole("navigation", { name: "Main navigation" })
    .getByRole("button", { name: "Team" })
    .click();
  await expect(
    page.getByRole("heading", { name: "Mara", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Establish the baseline", { exact: true }),
  ).toBeVisible();
  await page.screenshot({ path: "test-results/team.png", fullPage: true });
  await page.getByRole("tab", { name: "Shared notes" }).click();
  await expect(
    page.getByText("Baseline located.", { exact: false }).first(),
  ).toBeVisible();
  await page.getByRole("button", { name: "Add a note" }).click();
  await page
    .getByRole("textbox", { name: "Shared note" })
    .fill("Use identical seeds for the next comparison.");
  await page.getByRole("button", { name: "Save note" }).click();
  await expect(
    page.getByText("Use identical seeds for the next comparison.", {
      exact: true,
    }),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Communication" }).click();
  await expect(
    page.getByText("Can you confirm whether", { exact: false }),
  ).toBeVisible();
  await expect(page.locator(".conversation-thread")).toHaveCount(1);
  await expect(page.getByText("answered", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Open full thread" }).click();
  await expect(page.getByRole("dialog")).toContainText(
    "held-out set includes unseen keys",
  );
  await page.keyboard.press("Escape");
  await page.screenshot({
    path: "test-results/conversations.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Processes", exact: true }).click();
  await expect(page.locator(".terminal-output pre")).toContainText(
    "Starting baseline evaluation",
  );
  await page.screenshot({ path: "test-results/processes.png", fullPage: true });
  await page.getByRole("button", { name: "Changes", exact: true }).click();
  await page.getByRole("button", { name: "model.py M" }).click();
  await expect(
    page.getByText("+    return x * 2", { exact: true }),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Source", exact: true }).click();
  await page
    .getByRole("textbox", { name: "Edit model.py" })
    .fill("def forward(x):\n    return x * 3\n");
  await page.getByRole("button", { name: "Save changes" }).click();
  await expect(
    page.getByText("Matches the file on disk", { exact: false }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Knowledge", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Memory routing baseline" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  for (const effort of ["low", "medium", "high", "xhigh", "max"]) {
    await expect(
      page.getByRole("button", { name: effort, exact: true }),
    ).toBeVisible();
  }
  await page.screenshot({ path: "test-results/settings.png", fullPage: true });
  await page
    .getByRole("combobox", { name: "Provider", exact: true })
    .selectOption("anthropic");
  await page
    .getByRole("combobox", { name: "Model", exact: true })
    .fill("claude-sonnet-4-6");
  await expect(
    page.getByRole("button", { name: "off", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "xhigh", exact: true }),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "max", exact: true }).click();
  await page.getByRole("button", { name: "Save changes", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Saved", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "claude-sonnet-4-6", exact: true }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test("messages stream once, permissions survive reload, and session names persist", async ({
  page,
}) => {
  await page.getByRole("button", { name: "New session", exact: true }).click();
  await page
    .getByRole("textbox", { name: "Message Infinidev" })
    .fill("Continue the research.");
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(
    page.getByText(
      "The research task completed with evidence and validation.",
      { exact: true },
    ),
  ).toHaveCount(1);
  await page
    .getByRole("textbox", { name: "Message Infinidev" })
    .fill("Check permission before the test.");
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Allow this action" }),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Allow this action" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Deny", exact: true }).click();
  await expect(
    page.getByText("Permission denied.", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Options for Continue the research." })
    .click();
  await page.getByRole("menuitem", { name: "Rename session" }).click();
  await page
    .getByRole("textbox", { name: "Session name" })
    .fill("Research follow-up");
  await page.getByRole("button", { name: "Save name" }).click();
  await expect(
    page.getByText("Research follow-up", { exact: true }).first(),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByText("Research follow-up", { exact: true }).first(),
  ).toBeVisible();
});

test("an idle agent survives reload and wakes when a shared note is published", async ({
  page,
}) => {
  await page.getByRole("button", { name: "New session", exact: true }).click();
  await page
    .getByRole("textbox", { name: "Message Infinidev" })
    .fill("idle fixture");
  await page.getByRole("button", { name: "Send message", exact: true }).click();
  await page.getByRole("button", { name: "Team", exact: true }).click();
  await expect(
    page.getByText("Idle · Waiting for your research note"),
  ).toBeVisible();
  await page.locator(".agent-card").click();
  await expect(page.getByRole("dialog")).toContainText(
    "Until an event arrives",
  );
  await page.screenshot({
    path: "test-results/idle-agent.png",
    fullPage: true,
  });
  await page.keyboard.press("Escape");
  await page.reload();
  await page.getByRole("button", { name: "Team", exact: true }).click();
  await expect(
    page.getByText("Idle · Waiting for your research note"),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Shared notes" }).click();
  await page.getByRole("button", { name: "Add a note" }).click();
  await page
    .getByRole("textbox", { name: "Shared note" })
    .fill("New evidence is available.");
  await page.getByRole("button", { name: "Save note" }).click();
  await page.getByRole("button", { name: "Workspace", exact: true }).click();
  await expect(
    page.getByText("Woke after shared note.", { exact: true }),
  ).toBeVisible();
});

test("mobile navigation, command palette and dark theme remain usable", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Open navigation" }).click();
  await page.getByRole("button", { name: "Toggle theme" }).click();
  await page.getByRole("button", { name: "Workspace", exact: true }).click();
  await expect(
    page.getByRole("textbox", { name: "Message Infinidev" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await expect(page.locator(".harness")).not.toHaveClass(/nav-open/);
  await page.screenshot({
    path: "test-results/mobile.png",
    fullPage: true,
    animations: "disabled",
  });
  await page.keyboard.press("Control+k");
  await page
    .getByRole("textbox", { name: "Search commands and sessions" })
    .fill("knowledge");
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Knowledge" })
    .click();
  await expect(
    page.getByRole("heading", { name: "Good work builds on what’s known." }),
  ).toBeVisible();
});
