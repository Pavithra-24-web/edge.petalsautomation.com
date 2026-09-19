import { computeOutputFeatures, computeFeatureChain, resolveSelectedAxes } from "@/components/dashboard/impulse/featurePipeline";

describe("computeOutputFeatures", () => {
  it("raw passes axis names through unchanged", () => {
    expect(computeOutputFeatures("raw", ["accX", "accY", "accZ"])).toEqual(["accX", "accY", "accZ"]);
  });

  it("flatten expands each axis by its selected stats, defaulting to mean/std/rms", () => {
    expect(computeOutputFeatures("flatten", ["accX", "accY"])).toEqual([
      "accX Mean", "accX Std", "accX RMS",
      "accY Mean", "accY Std", "accY RMS",
    ]);
  });

  it("flatten honors a custom stat selection", () => {
    expect(computeOutputFeatures("flatten", ["accX"], { features: ["max", "min"] })).toEqual([
      "accX Max", "accX Min",
    ]);
  });

  it("spectral_analysis produces one named feature per axis", () => {
    expect(computeOutputFeatures("spectral_analysis", ["accX", "accY", "accZ"])).toEqual([
      "accX RMS", "accY RMS", "accZ RMS",
    ]);
  });
});

describe("resolveSelectedAxes", () => {
  it("defaults to everything available when nothing is persisted", () => {
    expect(resolveSelectedAxes(["accX", "accY", "accZ"], undefined)).toEqual(["accX", "accY", "accZ"]);
  });

  it("keeps a valid persisted subset", () => {
    expect(resolveSelectedAxes(["accX", "accY", "accZ"], ["accX", "accZ"])).toEqual(["accX", "accZ"]);
  });

  it("drops persisted axes no longer available and falls back to everything if nothing survives", () => {
    expect(resolveSelectedAxes(["accX", "accY"], ["gyroX", "gyroY"])).toEqual(["accX", "accY"]);
  });

  it("prunes only the stale entries when some persisted axes still exist", () => {
    expect(resolveSelectedAxes(["accX", "accY", "accZ"], ["accX", "gyroX"])).toEqual(["accX"]);
  });
});

describe("computeFeatureChain", () => {
  it("feeds dataset features into block 0 and each block's output into the next (3 axes)", () => {
    const dataset = ["accX", "accY", "accZ"];
    const blocks = [
      { type: "flatten", params: { features: ["rms"] } },
      { type: "raw" },
    ];
    const chain = computeFeatureChain(dataset, blocks);
    expect(chain.inputs[0]).toEqual(dataset);
    expect(chain.outputs[0]).toEqual(["accX RMS", "accY RMS", "accZ RMS"]);
    // Block 2's input is exactly block 1's output.
    expect(chain.inputs[1]).toEqual(chain.outputs[0]);
    expect(chain.outputs[1]).toEqual(chain.outputs[0]); // raw passes through
  });

  it("respects a block's own persisted axis subset instead of always using everything", () => {
    const dataset = ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"];
    const blocks = [{ type: "spectral_analysis", input_axes: ["accX", "accY"] }];
    const chain = computeFeatureChain(dataset, blocks);
    expect(chain.inputs[0]).toEqual(dataset);
    expect(chain.outputs[0]).toEqual(["accX RMS", "accY RMS"]);
  });

  it("scales to 6 and 9 dataset features with no code change", () => {
    const six = ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"];
    const nine = [...six, "magX", "magY", "magZ"];
    expect(computeFeatureChain(six, [{ type: "raw" }]).outputs[0]).toHaveLength(6);
    expect(computeFeatureChain(nine, [{ type: "raw" }]).outputs[0]).toHaveLength(9);
  });
});
