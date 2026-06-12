/**
 * Unit tests for the pure config helpers (the validation footer label).
 *
 * Run with Node's built-in test runner (native TS type-stripping):
 *   node --test src/config.test.ts
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { footerLabel } from "./config.ts";

test("footerLabel: valid shows steps + journey", () => {
  assert.equal(
    footerLabel({ valid: true, errors: [], steps: 3, journey: "main" }),
    "Valid · 3 steps · journey: main",
  );
});

test("footerLabel: invalid shows first error", () => {
  assert.equal(
    footerLabel({ valid: false, errors: ["boom", "x"], steps: 0, journey: "" }),
    "Invalid · boom",
  );
});

test("footerLabel: invalid with no error message is graceful", () => {
  assert.equal(
    footerLabel({ valid: false, errors: [], steps: 0, journey: "" }),
    "Invalid · see editor",
  );
});
