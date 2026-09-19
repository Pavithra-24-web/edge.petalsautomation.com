import {
  buildBlockOutputRows,
  resolveBlockOutputFilename,
  TrainedModelArtifact,
} from "@/lib/block-output-helpers";

function rowsFor(
  featuresReady: boolean,
  activeModelRunId: string | null,
  models: TrainedModelArtifact[] | null,
) {
  return buildBlockOutputRows({ featuresReady, activeModelRunId, models });
}

function byKey(rows: ReturnType<typeof rowsFor>) {
  return Object.fromEntries(rows.map((r) => [r.key, r]));
}

describe("buildBlockOutputRows", () => {
  test("all four rows present, in a fixed order", () => {
    const rows = rowsFor(true, "run-1", []);
    expect(rows.map((r) => r.key)).toEqual([
      "features",
      "keras",
      "litert_float32",
      "litert_int8",
    ]);
  });

  test("features row enabled only when featuresReady is true", () => {
    expect(byKey(rowsFor(true, null, null)).features.enabled).toBe(true);
    expect(byKey(rowsFor(true, null, null)).features.reason).toBeNull();

    const disabled = byKey(rowsFor(false, null, null)).features;
    expect(disabled.enabled).toBe(false);
    expect(disabled.reason).toBe("No features generated yet");
  });

  test("model rows disabled with 'no active model' when there is no active run", () => {
    const rows = byKey(rowsFor(true, null, null));
    expect(rows.keras.enabled).toBe(false);
    expect(rows.keras.reason).toBe("No active model for this impulse");
    expect(rows.litert_float32.enabled).toBe(false);
    expect(rows.litert_int8.enabled).toBe(false);
  });

  test("maps artifacts by format/variant, not by array position", () => {
    // int8 listed first, keras last — position must not matter.
    const models: TrainedModelArtifact[] = [
      { id: "m-int8", format: "tflite", variant: "int8" },
      { id: "m-float32", format: "tflite", variant: "float32" },
      { id: "m-keras", format: "keras" },
    ];
    const rows = byKey(rowsFor(true, "run-1", models));

    expect(rows.keras.enabled).toBe(true);
    expect(rows.keras.modelId).toBe("m-keras");
    expect(rows.litert_float32.enabled).toBe(true);
    expect(rows.litert_float32.modelId).toBe("m-float32");
    expect(rows.litert_int8.enabled).toBe(true);
    expect(rows.litert_int8.modelId).toBe("m-int8");
  });

  test("tflite artifact with no variant field is treated as float32", () => {
    const models: TrainedModelArtifact[] = [{ id: "m-1", format: "tflite" }];
    const rows = byKey(rowsFor(true, "run-1", models));
    expect(rows.litert_float32.enabled).toBe(true);
    expect(rows.litert_float32.modelId).toBe("m-1");
    expect(rows.litert_int8.enabled).toBe(false);
  });

  test("missing int8 artifact reads as quantization failure, not 'no active model'", () => {
    const models: TrainedModelArtifact[] = [
      { id: "m-keras", format: "keras" },
      { id: "m-float32", format: "tflite", variant: "float32" },
    ];
    const rows = byKey(rowsFor(true, "run-1", models));
    expect(rows.litert_int8.enabled).toBe(false);
    expect(rows.litert_int8.reason).toBe(
      "Quantized model unavailable — int8 conversion may have failed",
    );
  });

  test("missing keras artifact for an otherwise-active run has its own reason", () => {
    const models: TrainedModelArtifact[] = [
      { id: "m-int8", format: "tflite", variant: "int8" },
    ];
    const rows = byKey(rowsFor(true, "run-1", models));
    expect(rows.keras.enabled).toBe(false);
    expect(rows.keras.reason).toBe("Keras model unavailable for this run");
  });

  test("null/undefined models list behaves like an empty list", () => {
    const rows = byKey(rowsFor(true, "run-1", null));
    expect(rows.keras.enabled).toBe(false);
    expect(rows.litert_float32.enabled).toBe(false);
    expect(rows.litert_int8.enabled).toBe(false);
  });
});

describe("resolveBlockOutputFilename", () => {
  // Content-Disposition is not on the browser's CORS-safelisted header set,
  // so this must degrade to a *per-row* fallback when the header can't be
  // read cross-origin — never one name shared by every model row (the bug
  // this test guards: all three model downloads landing as "model.bin").

  test("prefers the backend filename when the header is readable", () => {
    expect(
      resolveBlockOutputFilename("keras", 'attachment; filename="model_v7.keras"'),
    ).toBe("model_v7.keras");
    expect(
      resolveBlockOutputFilename(
        "litert_float32",
        'attachment; filename="model_v7_float32.tflite"',
      ),
    ).toBe("model_v7_float32.tflite");
    expect(
      resolveBlockOutputFilename("litert_int8", 'attachment; filename="model_v7_int8.tflite"'),
    ).toBe("model_v7_int8.tflite");
    expect(
      resolveBlockOutputFilename(
        "features",
        'attachment; filename="my-impulse_features.npy"',
        "Some Other Impulse",
      ),
    ).toBe("my-impulse_features.npy");
  });

  test("falls back to a row-specific name — never a shared 'model.bin' — when the header is absent", () => {
    expect(resolveBlockOutputFilename("keras", undefined)).toBe("model.keras");
    expect(resolveBlockOutputFilename("litert_float32", undefined)).toBe("model_float32.tflite");
    expect(resolveBlockOutputFilename("litert_int8", undefined)).toBe("model_int8.tflite");

    const fallbacks = new Set([
      resolveBlockOutputFilename("keras", null),
      resolveBlockOutputFilename("litert_float32", null),
      resolveBlockOutputFilename("litert_int8", null),
    ]);
    expect(fallbacks.size).toBe(3);
    expect(fallbacks.has("model.bin")).toBe(false);
  });

  test("features fallback uses the impulse name as a slug when the header is absent", () => {
    expect(resolveBlockOutputFilename("features", undefined, "My Cool Impulse!!")).toBe(
      "my-cool-impulse_features.npy",
    );
  });

  test("features fallback degrades to 'impulse' when no name is available", () => {
    expect(resolveBlockOutputFilename("features", undefined, undefined)).toBe(
      "impulse_features.npy",
    );
    expect(resolveBlockOutputFilename("features", "", null)).toBe("impulse_features.npy");
  });
});
