import { expect, test, type Page } from "@playwright/test";

function collectBrowserErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  return errors;
}

test("the compact dashboard exposes every primary surface", async ({ page }, testInfo) => {
  const errors = collectBrowserErrors(page);
  await page.goto("/");
  await expect(page.getByRole("navigation", { name: "Dashboard sections" })).toBeVisible();
  await expect(page.locator('[data-view="metrics"]')).toBeVisible();
  await expect(page.locator(".chart-card").first()).toBeVisible();

  await page.getByRole("button", { name: /^all$/i }).click();
  await page.getByRole("searchbox", { name: "Filter metrics" }).fill("reward");
  await expect(page.locator(".metric-keys button").first()).toBeVisible();

  await page.getByRole("button", { name: "Traces", exact: true }).click();
  await expect(page.locator('[data-view="traces"] tbody tr').first()).toBeVisible();
  await page.locator('[data-view="traces"] tbody tr').first().click();
  const rollout = page.getByRole("dialog", { name: /Trace Viewer · sequence/ });
  await expect(rollout.getByText("Trace Viewer", { exact: true })).toBeVisible();
  if (testInfo.project.name === "desktop") {
    await expect(rollout.locator(".trace-list-item").first()).toBeVisible();
    await expect(rollout.locator(".trace-overview-pane")).toBeVisible();
  }
  await page.getByRole("button", { name: "Close detail" }).click();

  await page.getByRole("button", { name: "Report", exact: true }).click();
  await expect(page.locator('[data-view="report"]')).toBeVisible();
  await expect(page.getByText("Evaluation history", { exact: false })).toBeVisible();

  await page.getByRole("button", { name: "Logs", exact: true }).click();
  await expect(page.locator(".log-output")).not.toBeEmpty();
  await page.getByRole("button", { name: "split", exact: true }).click();
  await expect(page.locator(".log-pane").first()).toBeVisible();

  await page.getByRole("button", { name: "Config", exact: true }).click();
  await page.getByRole("searchbox", { name: "Filter configuration" }).fill("seq_len");
  await expect(page.locator(".config-output")).toContainText("seq_len");
  expect(errors).toEqual([]);
});

test("run selection is reflected in the URL and header", async ({ page }) => {
  await page.goto("/");
  const picker = page.getByRole("combobox", { name: "Run", exact: true });
  await picker.selectOption("synthetic-a");
  await expect.poll(() => new URL(page.url()).searchParams.get("run")).toBe("synthetic-a");
  await expect(page.getByRole("region", { name: "Run overview" })).toContainText("Qwen/Qwen3-0.6B");
});

test("the first reward observation has a visible chart marker", async ({ page }) => {
  await page.route("**/series?*", (route) => {
    const keys = new URL(route.request().url()).searchParams.get("keys")?.split(",") ?? [];
    return route.fulfill({
      json: {
        steps: [1],
        timestamps: ["2026-09-19T00:00:00Z"],
        series: Object.fromEntries(keys.map((key) => [key, [0.3125]])),
      },
    });
  });
  await page.goto("/");
  await page.locator('[data-metric="reward/all/mean"]').first().scrollIntoViewIfNeeded();
  const reward = page.getByRole("img", { name: "reward/all/mean chart", exact: true }).first();
  await expect(reward.locator("circle")).toBeVisible();
  await expect(reward.locator("circle title")).toHaveText("step 1: 0.3125");
});

test("empty and failed API states are explicit", async ({ page }) => {
  await page.route("**/api/runs?compact=true", (route) => route.fulfill({ status: 200, contentType: "application/json", body: "[]" }));
  await page.goto("/");
  await expect(page.getByText("No runs found", { exact: true })).toBeVisible();

  await page.unroute("**/api/runs?compact=true");
  await page.route("**/api/runs?compact=true", (route) => route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "dashboard unavailable" }) }));
  await page.reload();
  await expect(page.getByText("Cannot reach the API", { exact: true })).toBeVisible();
  await expect(page.getByText("dashboard unavailable", { exact: false })).toBeVisible();
});

test("mobile layout does not overflow", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "mobile");
  await page.goto("/");
  await expect(page.getByRole("navigation", { name: "Dashboard sections" })).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
});

test("charts support regex selection, inspection, and zoom", async ({ page }, testInfo) => {
  await page.goto("/");
  await page.getByRole("searchbox", { name: "Filter metrics" }).fill("^reward/|^train/loss$");
  const chart = page.getByRole("img", { name: "reward/all/mean chart", exact: true }).first();
  await expect(chart).toBeVisible();
  await chart.focus();
  await page.keyboard.press("ArrowRight");
  await expect(chart.locator("..").locator(".chart-readout")).toContainText("raw");
  if (testInfo.project.name === "desktop") {
    const bounds = (await chart.boundingBox())!;
    await page.mouse.move(bounds.x + bounds.width * .3, bounds.y + bounds.height / 2);
    await page.mouse.down();
    await page.mouse.move(bounds.x + bounds.width * .8, bounds.y + bounds.height / 2);
    await page.mouse.up();
    await expect(page.getByRole("button", { name: "Reset zoom" })).toBeVisible();
    await page.getByRole("button", { name: "Reset zoom" }).click();
  }
  await page.getByRole("searchbox", { name: "Filter metrics" }).fill("[");
  await expect(page.getByText("Invalid regular expression.")).toBeVisible();
});

test("trace pages are bounded, details are on demand, and tool calls survive", async ({ page }) => {
  let detailsRequested = 0;
  const offsets: number[] = [];
  await page.route("**/rollouts/rows?*", async route => {
    const params = new URL(route.request().url()).searchParams;
    expect(params.get("limit")).toBe("50");
    const offset = Number(params.get("offset")); offsets.push(offset);
    await route.fulfill({ json: { available: true, total: 120, filtered: 120, groups: [], rows: Array.from({length: Math.min(50, 120-offset)}, (_,i)=>({row_index:offset+i,reward:1,env:"test",input_token_count:1,completion_token_count:1})) }});
  });
  await page.route(/\/rollouts\/\d+\/rows\/\d+$/, async route => {
    detailsRequested++;
    await route.fulfill({json:{prompt:[], completion:Array.from({length:100},(_,i)=>({role:"assistant",content:null,tool_calls:[{function:{name:"bash",arguments:`echo lazy-message-${i}`}}]}))}});
  });
  await page.goto("/?run=synthetic-a&tab=traces");
  await expect(page.locator('[data-view="traces"] tbody tr')).toHaveCount(50);
  expect(detailsRequested).toBe(0);
  await page.getByRole("button", {name:"Next page"}).click();
  await expect.poll(()=>offsets.includes(50)).toBe(true);
  await page.locator('[data-view="traces"] tbody tr').first().click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByRole("dialog")).toContainText("echo lazy-message-0");
  await expect(page.getByRole("dialog").locator(".trace-entry")).toHaveCount(12);
  expect(detailsRequested).toBe(1);
  expect(new URL(page.url()).searchParams.get("row")).toBe("50");
  await page.getByRole("button", {name:"Close detail"}).click();
  await page.getByRole("combobox", {name:"Run",exact:true}).selectOption("synthetic-b");
  await expect(page.getByRole("dialog")).toHaveCount(0);
});


test("comparison overlays use a second run without replacing the selected run", async ({ page }) => {
  await page.goto("/?run=synthetic-a");
  await page.getByRole("combobox", { name: "Compare run" }).selectOption("synthetic-b");
  await expect(page.locator(".comparison-line").first()).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Run", exact: true })).toHaveValue("synthetic-a");
  await expect(page.locator(".chart-legend").first()).toContainText("synthetic-b");
});

test("downsampled graphs retain spike envelopes and label bucket means", async ({ page }) => {
  await page.route("**/series?*", route => {
    const keys = new URL(route.request().url()).searchParams.get("keys")!.split(",");
    return route.fulfill({json:{steps:[1,10],timestamps:[null,null],downsampled:true,series:Object.fromEntries(keys.map(k=>[k,[2,3]])),envelope:Object.fromEntries(keys.map(k=>[k,{min:[0,1],max:[90,100]}]))}});
  });
  await page.goto("/");
  await page.locator('[data-metric="reward/all/mean"]').first().scrollIntoViewIfNeeded();
  const chart = page.getByRole("img", {name:"reward/all/mean chart",exact:true}).first();
  await expect(chart.locator(".metric-envelope")).toBeVisible();
  await chart.focus();await page.keyboard.press("ArrowRight");
  await expect(chart.locator("..").locator(".chart-readout")).toContainText("bucket mean");
  await expect(chart.locator("..").locator(".chart-readout")).toContainText("max 90");
});

test("hidden browser tabs stop polling and refresh when visible", async ({ page }) => {
  let summaries = 0;
  page.on("request", request => { if (new URL(request.url()).pathname.endsWith("/summary")) summaries++; });
  await page.goto("/");
  await expect(page.locator(".chart-card").first()).toBeVisible();
  await page.evaluate(() => { Object.defineProperty(document, "hidden", {configurable:true,get:()=>true}); document.dispatchEvent(new Event("visibilitychange")); });
  const before = summaries;
  await page.waitForTimeout(3500);
  expect(summaries).toBe(before);
  await page.evaluate(() => { Object.defineProperty(document, "hidden", {configurable:true,get:()=>false}); document.dispatchEvent(new Event("visibilitychange")); });
  await expect.poll(()=>summaries).toBeGreaterThan(before);
});

test("reward cohorts show denominators and queue-step semantics", async ({ page }) => {
  const values: Record<string, number> = {
    "reward/episodes/all/mean": 0.15234375,
    "reward/episodes/all/count": 256,
    "reward/episodes/effective/mean": 0.3482142857,
    "reward/episodes/effective/count": 112,
    "reward/all/mean": 0.15234375,
  };
  await page.route("**/metrics/keys", route => route.fulfill({
    json: { trainer: [], orchestrator: Object.keys(values).map(key => ({ key })), eval: [] },
  }));
  await page.route("**/series?*", route => {
    const keys = new URL(route.request().url()).searchParams.get("keys")?.split(",") ?? [];
    return route.fulfill({ json: { steps: [0], timestamps: ["2026-09-20T00:00:00Z"], series: Object.fromEntries(keys.map(key => [key, [values[key]]])) } });
  });
  await page.goto("/");
  const all = page.locator('[data-metric="reward/episodes/all/mean"]');
  await expect(all).toContainText("Reward · all episodes");
  await expect(all).toContainText("including filtered episodes");
  await all.scrollIntoViewIfNeeded();
  await expect(all).toContainText("Rollout queue step");
  const trainable = page.locator('[data-metric="reward/episodes/effective/mean"]');
  await trainable.scrollIntoViewIfNeeded();
  await expect(trainable).toContainText("not an overall solve rate");
  await expect(page.locator('[data-metric="reward/episodes/all/count"] .chart-latest')).toHaveText("256");
  await expect(page.locator('[data-metric="reward/episodes/effective/count"] .chart-latest')).toHaveText("112");
  await expect(page.locator('[data-metric="reward/all/mean"]')).toHaveCount(0);
});
