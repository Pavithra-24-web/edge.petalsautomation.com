"use client";
import { useEffect, useMemo, useState } from "react";
import { deviceCatalogApi } from "@/utils/api";
import {
  Search, ShieldAlert, ShieldCheck, ShieldQuestion, AlertTriangle, Loader2,
  RotateCw, CircuitBoard, X, Cpu, MemoryStick, HardDrive, Timer, Building2,
} from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────

type DeployTarget = "tflite" | "arduino" | "esp32" | "raspberry_pi" | "unoq" | "cpp" | "pxe";
type DeviceClass = "microcontroller" | "linux_sbc" | "accelerator" | "desktop" | "mobile";

interface CatalogEntry {
  slug: string;
  display_name: string;
  family: string;
  deploy_target: DeployTarget;
  accelerator_note: string | null;
}

// Every field is `T | null` on purpose — NULL means "unknown," not zero or
// blank, and the UI has to keep that distinction visible (Phase 2, §7).
interface DeviceSpecification {
  vendor: string | null;
  device_class: DeviceClass;
  processor_family: string | null;
  processor: string | null;
  cpu_architecture: string | null;
  clock_rate_mhz: number | null;
  ram_kb: number | null;
  rom_kb: number | null;
  runtime_environment: string | null;
  has_ai_accelerator: boolean | null;
  ai_accelerator: string | null;
  latency_budget_ms: number | null;
  latency_budget_basis: string | null;
  supported_precisions: string[] | null;
}

interface CatalogEntryDetail extends CatalogEntry {
  specification: DeviceSpecification | null;
}

// ─── Deploy target legend ───────────────────────────────────────────────────────
// Every board in the catalog resolves to one of PetalEdge's seven existing build
// formats (docs/target_device_architecture.md §2: "many devices to one deploy
// target"). Colour-coding by target, rather than by family, is what actually
// shows that relationship — dozens of families share a colour on purpose.
const TARGET_META: Record<DeployTarget, { label: string; hex: string }> = {
  tflite:       { label: "TFLite",       hex: "#38bdf8" },
  arduino:      { label: "Arduino",      hex: "#34d399" },
  esp32:        { label: "ESP-IDF",      hex: "#22d3ee" },
  raspberry_pi: { label: "Raspberry Pi", hex: "#c084fc" },
  unoq:         { label: "UNO Q",        hex: "#818cf8" },
  cpp:          { label: "C++",          hex: "#a78bfa" },
  pxe:          { label: "PXE",          hex: "#fb7185" },
};
const TARGET_ORDER: DeployTarget[] = ["tflite", "arduino", "esp32", "raspberry_pi", "unoq", "cpp", "pxe"];

const DEVICE_CLASS_LABELS: Record<DeviceClass, string> = {
  microcontroller: "Microcontroller",
  linux_sbc: "Linux SBC",
  accelerator: "Accelerator",
  desktop: "Desktop",
  mobile: "Mobile",
};
const DEVICE_CLASS_ORDER: DeviceClass[] = ["microcontroller", "linux_sbc", "accelerator", "desktop", "mobile"];

function TargetBadge({ target }: { target: DeployTarget }) {
  const meta = TARGET_META[target];
  return (
    <span
      className="pe-cat-badge"
      style={{ color: meta.hex, background: `${meta.hex}1a`, borderColor: `${meta.hex}40` }}
    >
      <span className="pe-cat-dot" style={{ background: meta.hex }} />
      {meta.label}
    </span>
  );
}

// Accelerator-limited is a first-class property of a row at this scale — about
// half the catalog carries one — so it gets its own always-visible tag next to
// the target badge, not just a footnote under the card.
function LimitedTag() {
  return (
    <span className="pe-cat-limited">
      <ShieldAlert size={11} /> Accelerator limited
    </span>
  );
}

// ─── Display formatting — canonical units are stored (MHz / KB); formatting
// for a human is the UI's job, per the DeviceSpecification model docstring. ──

function fmtClock(mhz: number | null): string {
  if (mhz == null) return "Unknown";
  if (mhz >= 1000) {
    const ghz = mhz / 1000;
    return `${Number.isInteger(ghz) ? ghz : ghz.toFixed(2)} GHz`;
  }
  return `${mhz} MHz`;
}

function fmtKB(kb: number | null): string {
  if (kb == null) return "Unknown";
  if (kb >= 1024 * 1024) {
    const gb = kb / (1024 * 1024);
    return `${Number.isInteger(gb) ? gb : gb.toFixed(1)} GB`;
  }
  if (kb >= 1024) {
    const mb = kb / 1024;
    return `${Number.isInteger(mb) ? mb : mb.toFixed(1)} MB`;
  }
  return `${kb} KB`;
}

// A value cell that renders "Unknown" — visibly, not a blank or a dash —
// whenever the underlying figure is genuinely unsourced (Phase 2, §7).
function Fact({ icon, label, value }: { icon: React.ReactNode; label: string; value: string | null }) {
  const unknown = value == null;
  return (
    <div className="pe-spec-fact">
      <span className="pe-spec-fact-icon">{icon}</span>
      <div>
        <div className="pe-spec-fact-label">{label}</div>
        <div className={unknown ? "pe-spec-fact-value pe-spec-unknown" : "pe-spec-fact-value"}>
          {value ?? "Unknown"}
        </div>
      </div>
    </div>
  );
}

// The three-state accelerator fact gets its own renderer: "has none" and
// "nobody's checked" must never look the same (Phase 2, §7 — this is the
// whole reason the column is a nullable boolean rather than a plain flag).
function AcceleratorFact({ has, name }: { has: boolean | null; name: string | null }) {
  let icon: React.ReactNode;
  let text: string;
  let cls: string;
  if (has === true) {
    icon = <ShieldCheck size={14} />;
    text = name ? name : "Yes";
    cls = "pe-spec-accel-yes";
  } else if (has === false) {
    icon = <ShieldAlert size={14} style={{ opacity: 0.5 }} />;
    text = "No AI accelerator";
    cls = "pe-spec-accel-no";
  } else {
    icon = <ShieldQuestion size={14} />;
    text = "Unknown — not yet researched";
    cls = "pe-spec-accel-unknown";
  }
  return (
    <div className="pe-spec-fact">
      <span className="pe-spec-fact-icon">{icon}</span>
      <div>
        <div className="pe-spec-fact-label">AI accelerator</div>
        <div className={`pe-spec-fact-value ${cls}`}>{text}</div>
      </div>
    </div>
  );
}

function SpecPanel({ entry, onClose }: { entry: CatalogEntryDetail; onClose: () => void }) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === "Escape") onClose(); }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const spec = entry.specification;

  return (
    <div className="pe-spec-scrim" onClick={onClose}>
      <div
        className="pe-spec-panel"
        role="dialog"
        aria-modal="true"
        aria-label={`${entry.display_name} specification`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="pe-spec-head">
          <div>
            <div className="pe-spec-family">{entry.family}</div>
            <h2>{entry.display_name}</h2>
          </div>
          <button type="button" className="pe-spec-close" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>

        <div className="pe-spec-badges">
          <TargetBadge target={entry.deploy_target} />
          {entry.accelerator_note && <LimitedTag />}
        </div>
        {entry.accelerator_note && (
          <div className="pe-cat-note pe-spec-top-note">
            <AlertTriangle size={13} />
            <span>{entry.accelerator_note}</span>
          </div>
        )}

        {!spec ? (
          <div className="pe-spec-fact-value pe-spec-unknown" style={{ marginTop: 16 }}>
            No specification on file.
          </div>
        ) : (
          <>
            <div className="pe-spec-grid">
              <Fact icon={<Building2 size={14} />} label="Vendor" value={spec.vendor} />
              <Fact icon={<CircuitBoard size={14} />} label="Device class" value={DEVICE_CLASS_LABELS[spec.device_class]} />
              <Fact icon={<Cpu size={14} />} label="Processor family" value={spec.processor_family} />
              <Fact icon={<Cpu size={14} />} label="Processor" value={spec.processor} />
              <Fact icon={<Cpu size={14} />} label="CPU architecture" value={spec.cpu_architecture} />
              <Fact icon={<Timer size={14} />} label="Clock rate" value={fmtClock(spec.clock_rate_mhz)} />
              <Fact icon={<MemoryStick size={14} />} label="RAM" value={fmtKB(spec.ram_kb)} />
              <Fact icon={<HardDrive size={14} />} label="ROM / Flash" value={fmtKB(spec.rom_kb)} />
              <Fact icon={<CircuitBoard size={14} />} label="Runtime environment" value={spec.runtime_environment} />
              <AcceleratorFact has={spec.has_ai_accelerator} name={spec.ai_accelerator} />
            </div>

            <div className="pe-spec-divider" />

            <div className="pe-spec-latency">
              <div className="pe-spec-fact-label">Default latency budget</div>
              <div className="pe-spec-fact-value">
                {spec.latency_budget_ms == null ? (
                  <span className="pe-spec-unknown">Unknown</span>
                ) : (
                  <>
                    {spec.latency_budget_ms} ms
                    {spec.latency_budget_basis && (
                      <span className="pe-spec-basis"> — {spec.latency_budget_basis}</span>
                    )}
                  </>
                )}
              </div>
            </div>

            <div className="pe-spec-latency">
              <div className="pe-spec-fact-label">Supported precisions</div>
              {spec.supported_precisions == null ? (
                <div className="pe-spec-fact-value pe-spec-unknown">Unknown</div>
              ) : (
                <div className="pe-cat-card-badges" style={{ marginTop: 4 }}>
                  {spec.supported_precisions.map((p) => (
                    <span key={p} className="pe-spec-precision">{p}</span>
                  ))}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ─── Page ───────────────────────────────────────────────────────────────────────

const DEBOUNCE_MS = 220;

export default function DeviceCatalogPage() {
  const [allFamilies, setAllFamilies] = useState<{ family: string; count: number }[] | null>(null);
  const [entries, setEntries] = useState<CatalogEntry[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [family, setFamily] = useState<string>("");
  const [deviceClass, setDeviceClass] = useState<string>("");
  const [limitedOnly, setLimitedOnly] = useState(false);

  const [detailSlug, setDetailSlug] = useState<string | null>(null);
  const [detail, setDetail] = useState<CatalogEntryDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // Debounce the search box so every keystroke doesn't round-trip to the API.
  useEffect(() => {
    const t = setTimeout(() => setSearch(searchInput.trim()), DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [searchInput]);

  // Family list for the filter is fetched once and stays stable — it should
  // never shrink just because a search narrowed the visible results.
  useEffect(() => {
    deviceCatalogApi.families()
      .then(({ data }) => setAllFamilies(data.map((g: any) => ({ family: g.family, count: g.entries.length }))))
      .catch(() => {});
  }, []);

  async function load() {
    setLoading(true);
    setError(false);
    try {
      const { data } = await deviceCatalogApi.list({
        family: family || undefined,
        q: search || undefined,
        device_class: deviceClass || undefined,
      });
      setEntries(data);
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [family, search, deviceClass]);

  // Board detail — fetched lazily on open. GET /{slug} always embeds the full
  // specification, so the list request itself stays identity-only and lean
  // (the topbar picker calls it on every open and needs none of this).
  useEffect(() => {
    if (!detailSlug) { setDetail(null); return; }
    let cancelled = false;
    setDetailLoading(true);
    deviceCatalogApi.get(detailSlug)
      .then(({ data }) => { if (!cancelled) setDetail(data); })
      .catch(() => { if (!cancelled) setDetail(null); })
      .finally(() => { if (!cancelled) setDetailLoading(false); });
    return () => { cancelled = true; };
  }, [detailSlug]);

  const visible = useMemo(
    () => (entries ?? []).filter((e) => !limitedOnly || !!e.accelerator_note),
    [entries, limitedOnly],
  );

  const grouped = useMemo(() => {
    const map = new Map<string, CatalogEntry[]>();
    for (const e of visible) {
      if (!map.has(e.family)) map.set(e.family, []);
      map.get(e.family)!.push(e);
    }
    return [...map.entries()]
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([fam, items]) => ({ family: fam, entries: items }));
  }, [visible]);

  const totalKnown = useMemo(
    () => (allFamilies ?? []).reduce((sum, f) => sum + f.count, 0),
    [allFamilies],
  );

  const filtersActive = !!search || !!family || !!deviceClass || limitedOnly;

  function clearFilters() {
    setSearchInput("");
    setSearch("");
    setFamily("");
    setDeviceClass("");
    setLimitedOnly(false);
  }

  return (
    <div className="pe-catalog max-w-6xl mx-auto space-y-6">
      <style>{`
        .pe-catalog-header p { color: var(--app-text-muted); }
        .pe-cat-toolbar {
          display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
        }
        .pe-cat-search {
          position: relative; flex: 1 1 240px; min-width: 200px;
        }
        .pe-cat-search svg {
          position: absolute; left: 10px; top: 50%; transform: translateY(-50%);
          color: var(--app-text-soft); pointer-events: none;
        }
        .pe-cat-search input { padding-left: 32px; }
        .pe-cat-select { width: auto; min-width: 150px; flex: 0 0 auto; }
        .pe-cat-toggle {
          display: inline-flex; align-items: center; gap: 6px; flex: 0 0 auto;
          font-size: 12.5px; font-weight: 500; white-space: nowrap;
          padding: 7px 12px; border-radius: 8px; cursor: pointer;
          background: var(--app-surface-2); border: 1px solid var(--app-border);
          color: var(--app-text-muted); transition: all .12s ease;
        }
        .pe-cat-toggle[aria-pressed="true"] {
          color: #fbbf24; border-color: rgba(251,191,36,0.45); background: rgba(251,191,36,0.10);
        }
        .pe-cat-clear {
          font-size: 12px; color: var(--app-text-soft); display: inline-flex;
          align-items: center; gap: 3px; flex: 0 0 auto; cursor: pointer;
        }
        .pe-cat-clear:hover { color: var(--app-text); }
        .pe-cat-status { font-size: 12.5px; color: var(--app-text-soft); }
        .pe-cat-legend {
          display: flex; flex-wrap: wrap; gap: 8px;
          padding: 12px 14px; border-radius: 10px;
          background: var(--app-surface-2); border: 1px solid var(--app-border);
        }
        .pe-cat-legend-label {
          font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
          color: var(--app-text-soft); align-self: center; margin-right: 4px;
        }
        .pe-cat-badge {
          display: inline-flex; align-items: center; gap: 6px;
          font-size: 12px; font-weight: 500; line-height: 1;
          padding: 4px 9px; border-radius: 999px; border: 1px solid transparent;
          white-space: nowrap;
        }
        .pe-cat-dot { width: 6px; height: 6px; border-radius: 999px; flex-shrink: 0; }
        .pe-cat-limited {
          display: inline-flex; align-items: center; gap: 4px;
          font-size: 10.5px; font-weight: 600; line-height: 1; text-transform: uppercase;
          letter-spacing: 0.02em; padding: 3px 7px; border-radius: 999px;
          color: #fbbf24; background: rgba(251,191,36,0.12); border: 1px solid rgba(251,191,36,0.35);
          white-space: nowrap;
        }
        .pe-cat-family { margin-bottom: 26px; }
        .pe-cat-family-head {
          display: flex; align-items: baseline; gap: 10px; margin-bottom: 12px;
          padding-bottom: 8px; border-bottom: 1px solid var(--app-border);
        }
        .pe-cat-family-head h2 {
          font-size: 14.5px; font-weight: 600; color: var(--app-text);
          display: flex; align-items: center; gap: 8px;
        }
        .pe-cat-family-count { font-size: 12px; color: var(--app-text-soft); }
        .pe-cat-grid {
          display: grid; gap: 10px;
          grid-template-columns: repeat(auto-fill, minmax(215px, 1fr));
        }
        .pe-cat-card {
          border-radius: 12px; padding: 12px 14px; text-align: left; width: 100%;
          background: var(--app-surface); border: 1px solid var(--app-border);
          cursor: pointer; transition: border-color .12s ease, transform .12s ease;
        }
        .pe-cat-card:hover { border-color: color-mix(in srgb, var(--app-border) 40%, var(--app-text-soft) 60%); }
        .pe-cat-card:focus-visible { outline: 2px solid var(--brand-500, #8b5cf6); outline-offset: 2px; }
        .pe-cat-card h3 {
          font-size: 13px; font-weight: 600; color: var(--app-text);
          margin-bottom: 8px; line-height: 1.3;
        }
        .pe-cat-card-badges { display: flex; flex-wrap: wrap; gap: 6px; }
        .pe-cat-note {
          display: flex; align-items: flex-start; gap: 6px;
          margin-top: 10px; padding-top: 10px; border-top: 1px dashed var(--app-border);
          font-size: 11.5px; line-height: 1.45; color: var(--app-text-muted);
        }
        .pe-cat-note svg { flex-shrink: 0; margin-top: 1px; color: #fbbf24; }
        .pe-cat-state {
          display: flex; flex-direction: column; align-items: center; gap: 10px;
          padding: 48px 0; color: var(--app-text-muted); text-align: center;
        }

        /* Board detail panel */
        .pe-spec-scrim {
          position: fixed; inset: 0; background: rgba(0,0,0,0.55);
          display: flex; align-items: flex-start; justify-content: center;
          padding: 5vh 16px; z-index: 60; overflow-y: auto;
        }
        .pe-spec-panel {
          width: 100%; max-width: 560px; border-radius: 16px;
          background: var(--app-surface); border: 1px solid var(--app-border);
          box-shadow: var(--app-shadow); padding: 20px 22px 24px;
        }
        .pe-spec-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
        .pe-spec-family { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--app-text-soft); }
        .pe-spec-head h2 { font-size: 18px; font-weight: 600; color: var(--app-text); margin-top: 2px; }
        .pe-spec-close {
          flex-shrink: 0; color: var(--app-text-soft); background: none; border: none;
          cursor: pointer; padding: 4px; border-radius: 6px;
        }
        .pe-spec-close:hover { color: var(--app-text); background: var(--app-surface-2); }
        .pe-spec-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
        .pe-spec-top-note { margin-top: 12px; border-top: none; padding-top: 0; }
        .pe-spec-grid {
          display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
          gap: 14px; margin-top: 18px;
        }
        .pe-spec-fact { display: flex; align-items: flex-start; gap: 8px; }
        .pe-spec-fact-icon { color: var(--app-text-soft); margin-top: 2px; flex-shrink: 0; }
        .pe-spec-fact-label { font-size: 11px; color: var(--app-text-soft); }
        .pe-spec-fact-value { font-size: 13px; color: var(--app-text); font-weight: 500; margin-top: 1px; }
        .pe-spec-unknown { color: var(--app-text-soft); font-style: italic; font-weight: 400; }
        .pe-spec-accel-yes { color: #34d399; }
        .pe-spec-accel-no { color: var(--app-text-muted); }
        .pe-spec-accel-unknown { color: var(--app-text-soft); font-style: italic; }
        .pe-spec-divider { height: 1px; background: var(--app-border); margin: 18px 0; }
        .pe-spec-latency { margin-bottom: 14px; }
        .pe-spec-basis { color: var(--app-text-soft); font-weight: 400; }
        .pe-spec-precision {
          font-size: 11px; font-family: monospace; padding: 3px 8px; border-radius: 6px;
          background: var(--app-surface-2); border: 1px solid var(--app-border); color: var(--app-text);
        }
      `}</style>

      <div className="pe-catalog-header">
        <h1 className="text-xl font-semibold" style={{ color: "var(--app-text)" }}>Device catalog</h1>
        <p className="text-sm mt-1">
          Known hardware PetalEdge can build for
          {totalKnown > 0 && <> — {totalKnown} boards across {(allFamilies ?? []).length} families</>}.
          Click a board for its full specification — selecting one for a project comes in a later phase.
        </p>
      </div>

      <div className="pe-cat-legend">
        <span className="pe-cat-legend-label">Deploy target</span>
        {TARGET_ORDER.map((t) => <TargetBadge key={t} target={t} />)}
      </div>

      <div className="pe-cat-toolbar">
        <div className="pe-cat-search">
          <Search size={14} />
          <input
            className="input"
            placeholder="Search boards or families…"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            aria-label="Search the device catalog"
          />
        </div>
        <select
          className="input pe-cat-select"
          value={family}
          onChange={(e) => setFamily(e.target.value)}
          aria-label="Filter by family"
        >
          <option value="">All families</option>
          {(allFamilies ?? []).map((f) => (
            <option key={f.family} value={f.family}>{f.family} ({f.count})</option>
          ))}
        </select>
        <select
          className="input pe-cat-select"
          value={deviceClass}
          onChange={(e) => setDeviceClass(e.target.value)}
          aria-label="Filter by device class"
        >
          <option value="">All device classes</option>
          {DEVICE_CLASS_ORDER.map((c) => (
            <option key={c} value={c}>{DEVICE_CLASS_LABELS[c]}</option>
          ))}
        </select>
        <button
          type="button"
          className="pe-cat-toggle"
          aria-pressed={limitedOnly}
          onClick={() => setLimitedOnly((v) => !v)}
        >
          <ShieldAlert size={13} /> Accelerator limited only
        </button>
        {filtersActive && (
          <button type="button" className="pe-cat-clear" onClick={clearFilters}>
            <X size={12} /> Clear filters
          </button>
        )}
      </div>

      {!loading && !error && (
        <div className="pe-cat-status">
          Showing {visible.length} of {totalKnown || visible.length} board{visible.length !== 1 ? "s" : ""}
        </div>
      )}

      {loading && (
        <div className="pe-cat-state">
          <Loader2 size={22} className="animate-spin" />
          <span>Loading catalog…</span>
        </div>
      )}

      {!loading && error && (
        <div className="pe-cat-state">
          <AlertTriangle size={22} />
          <span>Couldn't load the device catalog.</span>
          <button onClick={load} className="btn-secondary">
            <RotateCw size={14} /> Retry
          </button>
        </div>
      )}

      {!loading && !error && grouped.length === 0 && (
        <div className="pe-cat-state">
          <CircuitBoard size={22} />
          <span>No boards match {filtersActive ? "these filters" : "the catalog"}.</span>
          {filtersActive && (
            <button onClick={clearFilters} className="btn-secondary">Clear filters</button>
          )}
        </div>
      )}

      {!loading && !error && grouped.map((group) => (
        <div key={group.family} className="pe-cat-family">
          <div className="pe-cat-family-head">
            <h2><CircuitBoard size={14} strokeWidth={1.8} /> {group.family}</h2>
            <span className="pe-cat-family-count">
              {group.entries.length} board{group.entries.length !== 1 ? "s" : ""}
            </span>
          </div>
          <div className="pe-cat-grid">
            {group.entries.map((entry) => (
              <button
                key={entry.slug}
                type="button"
                className="pe-cat-card"
                onClick={() => setDetailSlug(entry.slug)}
              >
                <h3>{entry.display_name}</h3>
                <div className="pe-cat-card-badges">
                  <TargetBadge target={entry.deploy_target} />
                  {entry.accelerator_note && <LimitedTag />}
                </div>
                {entry.accelerator_note && (
                  <div className="pe-cat-note">
                    <AlertTriangle size={13} />
                    <span>{entry.accelerator_note}</span>
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>
      ))}

      {detailSlug && (
        detailLoading || !detail ? (
          <div className="pe-spec-scrim" onClick={() => setDetailSlug(null)}>
            <div className="pe-spec-panel" onClick={(e) => e.stopPropagation()}>
              <div className="pe-cat-state" style={{ padding: 24 }}>
                <Loader2 size={20} className="animate-spin" />
                <span>Loading specification…</span>
              </div>
            </div>
          </div>
        ) : (
          <SpecPanel entry={detail} onClose={() => setDetailSlug(null)} />
        )
      )}
    </div>
  );
}
