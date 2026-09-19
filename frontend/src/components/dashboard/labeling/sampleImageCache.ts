import { samplesApi } from "@/utils/api";

// ── Presigned image-URL fetch: cached, deduped, concurrency-limited ──────────
// Both the grid cards and the right-hand preview resolve a sample's image via
// samplesApi.download(id) → a presigned URL (backend get_presigned_url,
// expires_in=3600). The preview fires a single request on selection and loads
// reliably; the grid mounts one card per visible sample, firing a burst of
// concurrent requests. The previous per-card `.catch(() => {})` swallowed any
// download() call that failed under that burst (browser per-host socket cap /
// backend throttling), leaving those cards stuck on alt text while the very
// same image loads fine in the single-request preview.
//
// Routing both paths through this helper makes them share:
//   • a per-id URL cache (TTL well under the 1h presign expiry),
//   • in-flight de-duplication (concurrent callers for one id share a request),
//   • a small concurrency gate so a full page of cards can't saturate the
//     connection pool / trip backend throttling all at once,
//   • one transparent retry so a transient failure doesn't blank a card.
const DOWNLOAD_URL_TTL_MS = 10 * 60 * 1000; // refresh well before the 1h presign expiry
const DOWNLOAD_MAX_CONCURRENT = 6;
const _urlCache = new Map<string, { url: string; ts: number }>();
const _urlInflight = new Map<string, Promise<string>>();
let _downloadActive = 0;
const _downloadWaiters: Array<() => void> = [];

function _acquireDownloadSlot(): Promise<void> {
  if (_downloadActive < DOWNLOAD_MAX_CONCURRENT) {
    _downloadActive++;
    return Promise.resolve();
  }
  return new Promise<void>(resolve => _downloadWaiters.push(resolve)).then(() => {
    _downloadActive++;
  });
}
function _releaseDownloadSlot() {
  _downloadActive--;
  _downloadWaiters.shift()?.();
}

export async function fetchSampleUrl(sampleId: string, force = false): Promise<string> {
  if (force) _urlCache.delete(sampleId);
  const hit = _urlCache.get(sampleId);
  if (hit && Date.now() - hit.ts < DOWNLOAD_URL_TTL_MS) return hit.url;
  if (!force) {
    const inflight = _urlInflight.get(sampleId);
    if (inflight) return inflight;
  }
  const run = (async () => {
    await _acquireDownloadSlot();
    try {
      let lastErr: any;
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          const { data } = await samplesApi.download(sampleId);
          _urlCache.set(sampleId, { url: data.url, ts: Date.now() });
          return data.url as string;
        } catch (err) {
          lastErr = err;
          if (attempt === 0) await new Promise(r => setTimeout(r, 300));
        }
      }
      throw lastErr;
    } finally {
      _releaseDownloadSlot();
      _urlInflight.delete(sampleId);
    }
  })();
  _urlInflight.set(sampleId, run);
  return run;
}

// Decoded-image prefetch. The right-hand preview loads each image through
// `useImage(url, 'anonymous')`, so warming the browser's HTTP + decode cache
// with a matching cross-origin request means a later selection resolves from
// cache instead of paying another download + decode. `_imgCache` keeps a strong
// ref to each preloaded HTMLImageElement so it isn't GC'd before it's reused,
// and doubles as a dedupe set so neighbours aren't fetched twice.
const _imgCache = new Map<string, HTMLImageElement>();
export function preloadImage(url: string): void {
  if (!url || _imgCache.has(url)) return;
  const img = new Image();
  img.crossOrigin = "anonymous";
  _imgCache.set(url, img);
  img.src = url;
  // decode() forces the (expensive) decode off the navigation critical path;
  // ignore failures — they just mean the real load will decode as usual.
  img.decode?.().catch(() => { });
}

// Warm the presigned URL and decoded bitmap for a sample id ahead of selection.
export async function prefetchSample(sampleId: string): Promise<void> {
  try {
    const url = await fetchSampleUrl(sampleId);
    preloadImage(url);
  } catch {
    /* best-effort: a failed prefetch just falls back to on-demand load */
  }
}
