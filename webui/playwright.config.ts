import { defineConfig, devices } from "@playwright/test";

const port = Number(process.env.WAVELET_E2E_PORT ?? "8767");

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  retries: 0,
  reporter: "line",
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: {
    command:
      "cd .. && uv run python webui/e2e/serve.py",
    url: `http://127.0.0.1:${port}/api/health`,
    env: { WAVELET_E2E_PORT: String(port) },
    reuseExistingServer: true,
    timeout: 30_000,
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    {
      name: "mobile",
      use: { ...devices["iPhone 13"], browserName: "chromium" },
    },
  ],
});
