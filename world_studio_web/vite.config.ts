import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "/studio/",
  build: {
    target: "es2022",
    sourcemap: true,
  },
  test: {
    environment: "jsdom",
    clearMocks: true,
    restoreMocks: true,
    include: ["src/**/*.test.ts"],
  },
});
