import { expect, test } from "@playwright/test";

test("live episodes show the pending model prompt and updated reply", async ({ page }) => {
  const id = "125d6cfa-48c7-4f9e-b21a-76003f1d45fd";
  let replied = false;
  await page.route("**/episodes", (route) => route.fulfill({ json: {
    total: 1,
    episodes: [{ id, env: "live-demo", kind: "eval", status: replied ? "completed" : "running", phase: "get_model_response.started", policy_step: 3, revision: replied ? "2" : "1" }],
  } }));
  await page.route(`**/episodes/${id}?*`, (route) => route.fulfill({ json: {
    id, env: "live-demo", status: replied ? "completed" : "running",
    events: [{ at: "2026-09-17T00:00:00Z", phase: "get_model_response.started", prompt: [{ role: "user", content: "reverse hello" }] },
      ...(replied ? [{ at: "2026-09-17T00:00:01Z", phase: "get_model_response.finished", response: { role: "assistant", content: "olleh" } }] : [])],
  } }));
  await page.goto("/?tab=traces");
  await page.getByRole("button", { name: /live-demo/ }).click();
  await expect(page.getByLabel("Live transcript")).toContainText("reverse hello");
  replied = true;
  await expect(page.getByLabel("Live transcript")).toContainText("olleh");
});
