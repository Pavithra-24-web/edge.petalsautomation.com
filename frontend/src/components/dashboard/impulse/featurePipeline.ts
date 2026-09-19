// ─── Dataset-driven feature pipeline ───────────────────────────────────────
// The impulse graph (MotionImpulse) owns feature propagation: it reads the
// dataset's axes and each block's own config, and hands every block only the
// feature list it should see. No processing block computes upstream features
// itself, and no block hardcodes an axis list — every axis name here comes
// from the dataset (via samplesApi.featureAxes) or from a prior block's own
// output, never from a literal like ["accX","accY","accZ"].

export const FLATTEN_STAT_LABELS: Record<string, string> = {
  mean: "Mean",
  std: "Std",
  rms: "RMS",
  max: "Max",
  min: "Min",
  skewness: "Skewness",
  kurtosis: "Kurtosis",
};

export const FLATTEN_DEFAULT_STATS = ["mean", "std", "rms"];

/** Feature names a block produces, given the axes it was actually fed. Pure
 *  function of (block type, selected input axes, params) — never reaches
 *  outside its own arguments for axis names. */
export function computeOutputFeatures(blockType: string, inputAxes: string[], params?: any): string[] {
  switch (blockType) {
    case "flatten": {
      const stats: string[] = params?.features?.length ? params.features : FLATTEN_DEFAULT_STATS;
      const out: string[] = [];
      for (const axis of inputAxes) {
        for (const stat of stats) out.push(`${axis} ${FLATTEN_STAT_LABELS[stat] || stat}`);
      }
      return out;
    }
    case "spectral_analysis":
      return inputAxes.map((axis) => `${axis} RMS`);
    case "mfcc": {
      const n = Number(params?.num_coefficients) || 13;
      const out: string[] = [];
      for (const axis of inputAxes) {
        for (let i = 1; i <= n; i++) out.push(`${axis} MFCC ${i}`);
      }
      return out;
    }
    case "raw":
    default:
      return [...inputAxes];
  }
}

export interface FeatureChain {
  /** inputs[i] = features available to block i before its own axis selection is applied */
  inputs: string[][];
  /** outputs[i] = features block i produces (from its selected subset of inputs[i]) */
  outputs: string[][];
}

/** Reconciles each block's persisted `input_axes` against what's actually
 *  available at that position (dataset features for block 0, the previous
 *  block's output for every block after it): stale selections are dropped,
 *  and an empty/unset selection defaults to "everything available". */
export function resolveSelectedAxes(available: string[], persisted: string[] | undefined): string[] {
  const pruned = (persisted || []).filter((a) => available.includes(a));
  return pruned.length ? pruned : available;
}

/** Walks the DSP block list once, threading dataset features through every
 *  block in order. This is the single place that decides what each block
 *  sees — individual block components never compute this themselves. */
export function computeFeatureChain(datasetFeatures: string[], dspBlocks: any[]): FeatureChain {
  const inputs: string[][] = [];
  const outputs: string[][] = [];
  let current = datasetFeatures;
  for (const block of dspBlocks) {
    inputs.push(current);
    const selected = resolveSelectedAxes(current, block?.input_axes);
    const out = computeOutputFeatures(block?.type, selected, block?.params);
    outputs.push(out);
    current = out;
  }
  return { inputs, outputs };
}
