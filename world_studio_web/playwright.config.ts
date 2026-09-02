import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 180_000,
  expect: { timeout: 15_000 },
  reporter: [["line"]],
  use: {
    baseURL: "http://127.0.0.1:8766",
    browserName: "chromium",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "off",
  },
});
