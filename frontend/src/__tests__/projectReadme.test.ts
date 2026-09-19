// Mirrors the inline README helpers in dashboard/page.tsx. They can't be
// imported directly: Next.js's app-router typegen rejects any export from
// page.tsx other than its known route exports (default, metadata, ...), so
// they're kept unexported there and this test re-asserts the same
// one-line contracts. Keep README_MAX_CHARS in sync with _README_MAX_CHARS
// in backend/app/api/v1/endpoints/projects.py.
const README_MAX_CHARS = 32_000;

function isReadmeEmpty(description?: string | null): boolean {
  return !description || !description.trim();
}

function exceedsReadmeMax(description: string): boolean {
  return description.length > README_MAX_CHARS;
}

describe("isReadmeEmpty", () => {
  test.each([undefined, null, "", "   ", "\n\t "] as const)(
    "is true for %p",
    (value) => {
      expect(isReadmeEmpty(value)).toBe(true);
    },
  );

  test.each(["hello", "  hello  ", "# Title\n\nBody"])(
    "is false for %p",
    (value) => {
      expect(isReadmeEmpty(value)).toBe(false);
    },
  );
});

describe("exceedsReadmeMax", () => {
  test("is false at exactly the cap", () => {
    expect(exceedsReadmeMax("a".repeat(README_MAX_CHARS))).toBe(false);
  });

  test("is true one character over the cap", () => {
    expect(exceedsReadmeMax("a".repeat(README_MAX_CHARS + 1))).toBe(true);
  });

  test("is false for a short string", () => {
    expect(exceedsReadmeMax("short readme")).toBe(false);
  });

  test("is false for an empty string", () => {
    expect(exceedsReadmeMax("")).toBe(false);
  });
});
