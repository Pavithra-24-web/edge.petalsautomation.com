import { samplesApi } from "@/utils/api";

// ── Decoded-signal fetch: cached, deduped, concurrency-limited ──────────────
// The waveform equivalent of sampleImageCache.ts's fetchSampleUrl: instead of
// a presigned image URL, GET /samples/{id}/signal returns a decoded
// {axes, values, frequency_hz, duration_ms, num_channels, num_samples}
// payload (Motion Phase 2 §5, Batch 3). WaveformThumb mounts one per visible
// list/grid row, so the same concurrency-gate + TTL-cache + in-flight-dedupe
// shape used for image downloads applies here too — a page of thumbnails
// firing at once shouldn't saturate the connection pool, and reselecting a
// sample (e.g. switching list -> grid, or opening the detail panel) should
// resolve from cache instead of re-decoding it.
export interface SampleSignal {
  axes: string[];
  values: number[][];
  frequency_hz: number;
  duration_ms: number | null;
  num_channels: number;
  num_samples: number;
}

const SIGNAL_TTL_MS = 10 * 60 * 1000; // matches sampleImageCache's presign-refresh margin
const SIGNAL_MAX_CONCURRENT = 6;
const _signalCache = new Map<string, { data: SampleSignal; ts: number }>();
const _signalInflight = new Map<string, Promise<SampleSignal>>();
let _signalActive = 0;
const _signalWaiters: Array<() => void> = [];

function _acquireSignalSlot(): Promise<void> {
  if (_signalActive < SIGNAL_MAX_CONCURRENT) {
    _signalActive++;
    return Promise.resolve();
  }
  return new Promise<void>(resolve => _signalWaiters.push(resolve)).then(() => {
    _signalActive++;
  });
}
function _releaseSignalSlot() {
  _signalActive--;
  _signalWaiters.shift()?.();
}

export async function fetchSampleSignal(sampleId: string, force = false): Promise<SampleSignal> {
  if (force) _signalCache.delete(sampleId);
  const hit = _signalCache.get(sampleId);
  if (hit && Date.now() - hit.ts < SIGNAL_TTL_MS) return hit.data;
  if (!force) {
    const inflight = _signalInflight.get(sampleId);
    if (inflight) return inflight;
  }
  const run = (async () => {
    await _acquireSignalSlot();
    try {
      const { data } = await samplesApi.signal(sampleId);
      _signalCache.set(sampleId, { data, ts: Date.now() });
      return data as SampleSignal;
    } finally {
      _releaseSignalSlot();
      _signalInflight.delete(sampleId);
    }
  })();
  _signalInflight.set(sampleId, run);
  return run;
}
