import fs from "fs";
import path from "path";
import {
  isAnnotatable,
  isSampleLabeled,
  isSampleUnlabeled,
} from "@/components/dashboard/labeling/labelingStatus";

// Shared with backend/tests/test_labeling_status.py so both implementations
// of the labeling-status predicate are provably kept in agreement.
const FIXTURES_PATH = path.join(
  __dirname,
  "..",
  "..",
  "..",
  "backend",
  "tests",
  "fixtures",
  "labeling_status_fixtures.json"
);

type FixtureCase = {
  name: string;
  filename: string;
  bounding_boxes: any[];
  label_id: string | null;
  is_background: boolean;
  is_disabled: boolean;
  expected_unlabeled: boolean;
};

const cases: FixtureCase[] = JSON.parse(fs.readFileSync(FIXTURES_PATH, "utf-8"));

// Builds the already-flattened `_sample_to_dict` API response shape: unlike
// the backend's ORM fixture, is_background/is_disabled are top-level here.
function sampleFromCase(c: FixtureCase) {
  return {
    filename: c.filename,
    label_id: c.label_id,
    is_background: c.is_background,
    is_disabled: c.is_disabled,
    extra_metadata: {
      boundingBoxes: c.bounding_boxes,
      is_background: c.is_background,
      is_disabled: c.is_disabled,
    },
  };
}

describe("isSampleUnlabeled (shared fixtures)", () => {
  test.each(cases.map((c) => [c.name, c] as const))("%s", (_name, c) => {
    expect(isSampleUnlabeled(sampleFromCase(c))).toBe(c.expected_unlabeled);
  });
});

describe("isSampleLabeled (shared fixtures)", () => {
  test.each(cases.map((c) => [c.name, c] as const))("%s", (_name, c) => {
    const sample = sampleFromCase(c);
    if (isAnnotatable(sample)) {
      expect(isSampleLabeled(sample)).toBe(!c.expected_unlabeled);
    } else {
      expect(isSampleLabeled(sample)).toBe(false);
    }
  });
});

describe("isAnnotatable", () => {
  test("disabled sample is not annotatable", () => {
    expect(
      isAnnotatable({ filename: "x.jpg", is_disabled: true, extra_metadata: {} })
    ).toBe(false);
  });

  test("video sample is not annotatable", () => {
    expect(
      isAnnotatable({ filename: "x.mp4", is_disabled: false, extra_metadata: {} })
    ).toBe(false);
  });

  test("png image is annotatable", () => {
    expect(
      isAnnotatable({ filename: "x.png", is_disabled: false, extra_metadata: {} })
    ).toBe(true);
  });
});
