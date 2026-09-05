import { expect, test, type Page } from "@playwright/test";

const RUN_VIEWS = [
  ["overview", "Overview"],
  ["training", "Trainer"],
  ["rollouts", "Generation"],
  ["inspector", "Inspector"],
  ["evals", "Evaluation"],
  ["pipeline", "Step lifecycle"],
  ["infra", "Trainer heartbeat"],
  ["config", "Resolved config"],
] as const;
const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])';

function collectBrowserErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  return errors;
}

test("every run view renders without browser errors", async ({ page }) => {
  const errors = collectBrowserErrors(page);
  for (const [view] of RUN_VIEWS) {
    await page.goto(`/#/run/current/${view}`);
    await expect(page.locator(`[data-view="${view}"]`)).toBeVisible();
  }
  await page.goto("/#/run/synthetic-a/config");
  const filter = page.getByRole("textbox", { name: "Filter configuration" });
  await filter.fill("seq_len");
  await expect(page.getByText("seq_len", { exact: true }).first()).toBeVisible();
  expect(errors).toEqual([]);
});

test("run selection opens comparison", async ({ page }) => {
  await page.goto("/#/runs");
  const selectors = page.getByRole("checkbox", { name: /^Select / });
  await expect(selectors.nth(1)).toBeVisible();
  await selectors.nth(0).check();
  await selectors.nth(1).check();
  await page.getByRole("button", { name: /^Compare \(2\)$/ }).click();
  await expect(page).toHaveURL(/#\/compare\?runs=/);
  await expect(
    page.getByRole("heading", { name: /Compare \d+ runs/ }),
  ).toBeVisible();
});

test("theme and chart controls are usable", async ({ page }, testInfo) => {
  await page.goto("/#/run/current/overview");
  const themeButton = testInfo.project.name === "mobile"
    ? page.getByRole("button", { name: /Use (light|dark) theme/ })
    : page.getByRole("button", { name: "Toggle theme" });
  await themeButton.click();
  await expect(page.locator("html")).toHaveClass(/dark/);
  await page.getByRole("button", { name: "Chart settings" }).first().click();
  await expect(page.getByText("Smoothing", { exact: true })).toBeVisible();
});

test("charts and rollout rows open accessible detail views", async ({ page }) => {
  await page.goto("/#/run/current/overview");
  await page.getByRole("button", { name: "Expand Reward chart" }).click();
  const chartDialog = page.getByRole("dialog", { name: "Reward" });
  await expect(chartDialog).toBeVisible();
  await chartDialog.locator(FOCUSABLE).last().focus();
  await page.keyboard.press("Tab");
  expect(await chartDialog.evaluate((element) => element.contains(document.activeElement))).toBe(true);
  await page.keyboard.press("Escape");
  await expect(chartDialog).toBeHidden();

  await page.goto("/#/run/current/inspector");
  const composition = page.getByRole("status", { name: /Batch composition:/ });
  await expect(composition).toContainText(/\d+ prompt groups.*\d+ samples per group.*\d+ total rollouts/);
  const grouped = page.getByRole("region", { name: /Rollout groups in batch/ });
  await expect(grouped).toBeVisible();
  await expect(page.getByRole("region", { name: /Rollouts in group/ }).first()).toBeVisible();
  await page.getByRole("button", { name: /^Samples \(/ }).click();
  const table = page.getByRole("region", { name: /Rollouts in batch/ });
  await expect(table).toBeVisible();
  let delaySort = false;
  await page.route("**/rollouts/rows?**", async (route) => {
    if (delaySort) await new Promise((resolve) => setTimeout(resolve, 800));
    await route.continue();
  });
  delaySort = true;
  const sortedResponse = page.waitForResponse((response) =>
    response.url().includes("/rollouts/rows?") && response.url().includes("order=asc"),
  );
  await table.getByRole("button", { name: "Reward" }).click();
  await expect(table).toBeVisible({ timeout: 200 });
  await sortedResponse;
  delaySort = false;
  const firstRow = table.locator("tbody tr").first();
  await firstRow.focus();
  await page.keyboard.press("Enter");
  const rolloutDialog = page.getByRole("dialog", { name: /Rollout \d+ · batch/ });
  await expect(rolloutDialog).toBeVisible();
  await expect(rolloutDialog.getByText("Prompt", { exact: true })).toBeVisible();
  await page.route("**/rollouts/*/rows/*", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 800));
    await route.continue();
  });
  const sibling = rolloutDialog.locator('button[aria-pressed="false"]').first();
  const siblingIndex = (await sibling.textContent())?.replace("#", "") ?? "";
  const siblingResponse = page.waitForResponse((response) =>
    response.url().endsWith(`/rows/${siblingIndex}`),
  );
  await sibling.click();
  await expect(rolloutDialog.getByText("Prompt", { exact: true })).toBeVisible({ timeout: 200 });
  await expect(rolloutDialog.getByText(`loading #${siblingIndex}`)).toBeVisible();
  await siblingResponse;
  await expect(rolloutDialog).toHaveAccessibleName(new RegExp(`Rollout ${siblingIndex} · batch`));
  await rolloutDialog.locator(FOCUSABLE).last().focus();
  await page.keyboard.press("Tab");
  expect(await rolloutDialog.evaluate((element) => element.contains(document.activeElement))).toBe(true);
  await page.getByRole("button", { name: "Close drawer" }).click();
  await expect(firstRow).toBeFocused();
});

test("desktop navigation and run rows work from the keyboard", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name === "mobile");
  await page.goto("/#/runs");
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Skip to dashboard content" })).toBeFocused();

  const firstRow = page.getByRole("region", { name: "Runs table" }).locator("tbody tr").first();
  await firstRow.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator('[data-page="run"][data-view="overview"]')).toBeVisible();
});

test("batch timeline switches steps without a picker", async ({ page }) => {
  await page.goto("/#/run/current/inspector");
  const keepLatest = page.getByRole("button", { name: "Keep viewing latest" });
  await expect(keepLatest).toHaveAttribute("aria-pressed", "true");
  const rail = page.getByRole("navigation", { name: "Rollout batches" });
  await expect(rail).toBeVisible();
  const steps = rail.getByRole("button", { name: /^Batch step / });
  const count = await steps.count();
  expect(count).toBeGreaterThan(1);

  const previous = steps.nth(count - 2);
  await previous.click();
  await expect(previous).toHaveAttribute("aria-current", "step");
  await expect(keepLatest).toHaveAttribute("aria-pressed", "false");
  await expect.poll(() => new URL(page.url()).hash).toContain("step=");

  await keepLatest.click();
  await expect(steps.last()).toHaveAttribute("aria-current", "step");
  await expect(keepLatest).toHaveAttribute("aria-pressed", "true");
  await expect.poll(() => new URL(page.url()).hash).not.toContain("step=");
});

test("empty and failed API states explain how to recover", async ({ page }) => {
  await page.route("**/api/runs", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  await page.goto("/#/runs");
  await expect(page.getByText("No runs found", { exact: true })).toBeVisible();

  await page.unroute("**/api/runs");
  await page.route("**/api/runs", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ detail: "dashboard unavailable" }),
    }),
  );
  await page.reload();
  await expect(page.getByText("Cannot reach the API", { exact: false })).toBeVisible();
  await expect(page.getByText("dashboard unavailable", { exact: false }).first()).toBeVisible();
});

test("mobile navigation reaches all run views without page overflow", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "mobile");
  await page.goto("/#/run/current/overview");
  const nav = page.getByLabel("Run view", { exact: true });
  await expect(nav).toBeVisible();
  for (const [view] of RUN_VIEWS) {
    await nav.selectOption(view);
    await expect(page).toHaveURL(new RegExp(`#\\/run\\/current\\/${view}`));
  }
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);

  await page.setViewportSize({ width: 320, height: 720 });
  await expect(page.getByRole("link", { name: /(Current|Recent) run/ })).toBeVisible();
  await expect(page.getByRole("link", { name: "All runs" })).toBeVisible();
  const narrowOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(narrowOverflow).toBeLessThanOrEqual(1);

  await page.goto("/#/run/current/overview");
  await page.getByRole("button", { name: "Expand Reward chart" }).click();
  const modalOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(modalOverflow).toBeLessThanOrEqual(1);
});
