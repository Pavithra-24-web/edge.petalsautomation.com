import { isRealSuccess } from "@/components/dashboard/synthetic/SyntheticData";

// Regression coverage for the reported bug: the UI must never announce
// success on a bare `status === "completed"` — it also requires a real
// generated count and at least one recorded sample id, so a status-only
// regression (backend or otherwise) can't resurrect a false "success".

describe("isRealSuccess", () => {
  it("is true for a completed job with generated samples", () => {
    expect(
      isRealSuccess({ status: "completed", generated_count: 2, sample_ids: ["s1", "s2"] })
    ).toBe(true);
  });

  it("is false when status is not completed", () => {
    expect(
      isRealSuccess({ status: "running", generated_count: 2, sample_ids: ["s1", "s2"] })
    ).toBe(false);
    expect(
      isRealSuccess({ status: "failed", generated_count: 0, sample_ids: [] })
    ).toBe(false);
  });

  it("is false when completed but generated_count is 0", () => {
    expect(
      isRealSuccess({ status: "completed", generated_count: 0, sample_ids: [] })
    ).toBe(false);
  });

  it("is false when completed but sample_ids is empty despite a nonzero count", () => {
    expect(
      isRealSuccess({ status: "completed", generated_count: 3, sample_ids: [] })
    ).toBe(false);
  });

  it("is false when sample_ids is missing or not an array", () => {
    expect(isRealSuccess({ status: "completed", generated_count: 1 })).toBe(false);
    expect(
      isRealSuccess({ status: "completed", generated_count: 1, sample_ids: null })
    ).toBe(false);
  });

  it("is false for null/undefined job", () => {
    expect(isRealSuccess(null)).toBe(false);
    expect(isRealSuccess(undefined)).toBe(false);
  });
});
