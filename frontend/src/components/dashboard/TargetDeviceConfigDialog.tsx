"use client";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Cpu, ChevronDown, X, RotateCcw, Check, Loader2, HelpCircle } from "lucide-react";
import { useAppStore } from "@/store/appStore";
import { deviceCatalogApi, projectsApi } from "@/utils/api";

/**
 * The configuration half of Target Device Phase 3
 * (docs/target_device_phase3.md) — everything that needed a device hardware
 * specification (Phase 2) and so couldn't ship with the topbar chip.
 *
 * Replaces the chip's inline picker as the one surface for changing a
 * project's target device: selecting a board here loads its recommended
 * configuration automatically (steps 3–6 of the spec's workflow); RAM, ROM,
 * latency and a custom name are the only editable values, each shown beside
 * the board's own figure so a customised value reads as one at a glance.
 */

interface DeviceSpec {
  vendor: string | null;
  device_class: string;
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

interface CatalogEntry {
  slug: string;
  display_name: string;
  family: string;
  deploy_target: string;
  accelerator_note: string | null;
  specification: DeviceSpec | null;
}

interface FamilyGroup {
  family: string;
  entries: CatalogEntry[];
}

// Deployment packages are implemented per BOARD, not per deploy_target — a
// board's `deploy_target` says which build format it resolves to, not
// whether Petal Edge has actually shipped a downloadable package for that
// exact hardware. Several catalog entries share `deploy_target: raspberry_pi`
// (Jetson Nano, Coral Dev Board, Renesas RZ/V2H, …) or `unoq` (Arduino
// VENTUNO Q) with a board that does ship a package, without shipping one
// themselves — so membership here is an explicit slug list, never derived
// from `deploy_target`. Must be kept in sync with `SUPPORTED_DEVICE_PROFILES`
// in `deployment/page.tsx`, the other place these same three boards are
// enumerated. Order is Petal Edge's own hardware first.
const DEPLOYMENT_READY_SLUGS: readonly string[] = ["unoq", "raspberry_pi_4", "generic_tflite"];

// The seeded slug for "no specific hardware" (Phase 1, migration 0028). The
// catalog has no field distinguishing production-vs-development — device_class
// (microcontroller / linux_sbc / accelerator / desktop / mobile) describes the
// physical hardware, not that axis, so a Jetson would land in the wrong group
// if we tried to derive it from there. `family === "Generic"` is the one
// identity marker the catalog already carries for this entry; a real
// distinction would need a new catalog field, out of scope here.
const GENERIC_FAMILY = "Generic";

type StorageUnit = "KB" | "MB" | "GB";
const KB_PER_MB = 1024;
const KB_PER_GB = 1024 * 1024;

/** Pick the unit that reads naturally for a figure of this size — a 256 KB
 *  MCU shouldn't render as "0.0002 GB", and a 4 GB SBC shouldn't render as
 *  "4194304 KB". Falls back to MB when there's nothing to size against. */
function storageUnitFor(kb: number | null): StorageUnit {
  if (kb === null) return "MB";
  if (kb >= KB_PER_GB) return "GB";
  if (kb >= KB_PER_MB) return "MB";
  return "KB";
}

function kbToDisplay(kb: number, unit: StorageUnit): number {
  if (unit === "GB") return kb / KB_PER_GB;
  if (unit === "MB") return kb / KB_PER_MB;
  return kb;
}

function displayToKb(value: number, unit: StorageUnit): number {
  if (unit === "GB") return Math.round(value * KB_PER_GB);
  if (unit === "MB") return Math.round(value * KB_PER_MB);
  return Math.round(value);
}

/** Trims float noise (2.0000000004 GB) without hardcoding decimal places —
 *  a round-tripped value must redisplay as exactly what was typed. */
function formatDisplay(n: number): string {
  return parseFloat(n.toFixed(4)).toString();
}

/** Clock rate as a value/unit pair. The unit renders inside the field as a
 *  suffix, so it can't stay glued to the number the way a single string would. */
function splitClock(mhz: number | null): { value: string; unit: string } | null {
  if (mhz === null) return null;
  return mhz >= 1000
    ? { value: formatDisplay(mhz / 1000), unit: "GHz" }
    : { value: String(mhz), unit: "MHz" };
}

/** Generic, class-level fallback figures for the Application Budget section
 *  only — display purposes here, never sent to the backend and never
 *  substituted into `spec` itself. The board's own `board_default` (read
 *  from `DeviceSpecification`, resolved server-side in
 *  `_target_device_config`) stays NULL when unsourced, exactly as documented
 *  there — later phases validate against that value, so a generic estimate
 *  must never masquerade as it. `null` for a class/field means there's no
 *  defensible generic figure (an accelerator chip has no RAM/ROM budget of
 *  its own; it's host-attached), not that research is still pending. */
const CLASS_BUDGET_DEFAULTS: Record<
  string,
  { ram_kb: number | null; rom_kb: number | null; latency_budget_ms: number | null }
> = {
  microcontroller: { ram_kb: 256, rom_kb: 1024, latency_budget_ms: 150 },
  linux_sbc: { ram_kb: 1_048_576, rom_kb: 8_388_608, latency_budget_ms: 100 },
  accelerator: { ram_kb: null, rom_kb: null, latency_budget_ms: 20 },
  desktop: { ram_kb: 8_388_608, rom_kb: 268_435_456, latency_budget_ms: 50 },
  mobile: { ram_kb: 4_194_304, rom_kb: 67_108_864, latency_budget_ms: 50 },
};

interface ResolvedBudgetDefault {
  value: number | null;
  isClassDefault: boolean;
}

/** The board's own figure wins; otherwise falls back to a generic per-class
 *  estimate so the Application Budget section never shows "Unknown" for a
 *  board whose hardware is a known class even if its exact figure isn't. */
function resolveBudgetDefault(
  specValue: number | null,
  deviceClass: string | null,
  field: "ram_kb" | "rom_kb" | "latency_budget_ms",
): ResolvedBudgetDefault {
  if (specValue !== null) return { value: specValue, isClassDefault: false };
  const fallback = deviceClass ? (CLASS_BUDGET_DEFAULTS[deviceClass]?.[field] ?? null) : null;
  return { value: fallback, isClassDefault: fallback !== null };
}

/** Parses a field's text box. Blank means "use the board's value" — the
 *  override column stays NULL. A non-numeric or non-positive entry is
 *  reported as invalid rather than silently coerced. */
function parseOverrideText(text: string): { value: number | null; invalid: boolean } {
  const trimmed = text.trim();
  if (trimmed === "") return { value: null, invalid: false };
  const n = Number(trimmed);
  if (!Number.isFinite(n) || n <= 0) return { value: null, invalid: true };
  return { value: n, invalid: false };
}

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function TargetDeviceConfigDialog({ open, onClose }: Props) {
  const { activeProject, setActiveProject } = useAppStore();

  const [groups, setGroups] = useState<FamilyGroup[]>([]);
  const [loadingCatalog, setLoadingCatalog] = useState(false);

  const [selectedSlug, setSelectedSlug] = useState<string | null>(null);
  const [customName, setCustomName] = useState("");
  const [ramText, setRamText] = useState("");
  const [romText, setRomText] = useState("");
  const [latencyText, setLatencyText] = useState("");

  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  // document.body doesn't exist during the server render, and this app builds
  // with output: "export" — the portal below waits for this before mounting.
  const [mounted, setMounted] = useState(false);
  const modalRef = useRef<HTMLDivElement | null>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Re-seed the form from the project's currently *saved* configuration each
  // time the dialog opens — never from a previous, possibly-abandoned, edit.
  useEffect(() => {
    if (!open || !activeProject) return;
    setSelectedSlug(activeProject.target_device_slug ?? null);
    const config = activeProject.target_device_config;
    setCustomName(config?.custom_name ?? "");
    setSaveError(null);
    // Text fields are pre-filled from the RESOLVED value only when the field
    // is actually overridden — an inherited value belongs in the reference
    // figure beside the input, not typed into it as if the user had entered it.
    setRamText("");
    setRomText("");
    setLatencyText("");
    if (config?.ram_kb.overridden && config.ram_kb.value !== null) {
      const unit = storageUnitFor(config.ram_kb.board_default ?? config.ram_kb.value);
      setRamText(formatDisplay(kbToDisplay(config.ram_kb.value, unit)));
    }
    if (config?.rom_kb.overridden && config.rom_kb.value !== null) {
      const unit = storageUnitFor(config.rom_kb.board_default ?? config.rom_kb.value);
      setRomText(formatDisplay(kbToDisplay(config.rom_kb.value, unit)));
    }
    if (config?.latency_ms.overridden && config.latency_ms.value !== null) {
      setLatencyText(String(config.latency_ms.value));
    }
  }, [open, activeProject]);

  useEffect(() => {
    if (!open || groups.length > 0) return;
    let cancelled = false;
    setLoadingCatalog(true);
    deviceCatalogApi
      .families({ include: "specification" })
      .then(({ data }) => {
        if (!cancelled) setGroups(Array.isArray(data) ? data : []);
      })
      .catch(() => {
        if (!cancelled) setGroups([]);
      })
      .finally(() => {
        if (!cancelled) setLoadingCatalog(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, groups.length]);

  useEffect(() => {
    if (!open) return;
    function focusableElements(): HTMLElement[] {
      const modal = modalRef.current;
      if (!modal) return [];
      return Array.from(
        modal.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((el) => el.offsetParent !== null);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key !== "Tab") return;
      const focusables = focusableElements();
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  // Locks page scroll and moves focus into the dialog while it's open, and
  // restores both on close — via this effect's cleanup, so an unmount without
  // an explicit onClose (e.g. navigating away) still restores focus/scroll.
  useEffect(() => {
    if (!open) return;
    previouslyFocusedRef.current = document.activeElement as HTMLElement | null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const firstFocusable = modalRef.current?.querySelector<HTMLElement>(
      'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    );
    (firstFocusable ?? modalRef.current)?.focus();

    return () => {
      document.body.style.overflow = previousOverflow;
      previouslyFocusedRef.current?.focus();
    };
  }, [open]);

  const allEntries = useMemo(() => groups.flatMap((g) => g.entries), [groups]);
  const selectedEntry = useMemo(
    () => allEntries.find((e) => e.slug === selectedSlug) ?? null,
    [allEntries, selectedSlug],
  );
  // A slug the project already has, but whose catalog row is gone — same
  // "unresolved" tolerance as the backend, so the dialog doesn't blank out.
  const isUnresolved = !!selectedSlug && groups.length > 0 && !selectedEntry;
  const spec = selectedEntry?.specification ?? null;

  function handleSelectBoard(slug: string | null) {
    setSelectedSlug(slug);
    setSaveError(null);
    // Step 6 of the workflow: changing the board reloads its defaults
    // immediately, in the open dialog, before saving — so every override
    // input clears. Nothing from the previous board should read as though it
    // still applies to the new one.
    setCustomName("");
    setRamText("");
    setRomText("");
    setLatencyText("");
  }

  const ramParsed = parseOverrideText(ramText);
  const romParsed = parseOverrideText(romText);
  const latencyParsed = parseOverrideText(latencyText);
  const hasDevice = !!selectedSlug;
  const canSave =
    hasDevice && !ramParsed.invalid && !romParsed.invalid && !latencyParsed.invalid;

  async function persist(payload: {
    target_device_slug?: string | null;
    target_device_custom_name?: string | null;
    target_device_ram_kb?: number | null;
    target_device_rom_kb?: number | null;
    target_device_latency_ms?: number | null;
  }) {
    if (!activeProject) return;
    setSaving(true);
    setSaveError(null);
    try {
      const { data } = await projectsApi.update(activeProject.id, payload);
      setActiveProject(data);
      onClose();
    } catch {
      setSaveError("Couldn't save the target device configuration. Try again.");
    } finally {
      setSaving(false);
    }
  }

  function handleSave() {
    if (!canSave) return;
    // Must match the unit shown next to the input (which falls back to a
    // class-level default when the board's own figure is unsourced) — saving
    // against the raw board figure here would silently misinterpret a value
    // the user typed against a fallback-derived GB/MB label.
    const ramUnit = storageUnitFor(resolveBudgetDefault(spec?.ram_kb ?? null, spec?.device_class ?? null, "ram_kb").value);
    const romUnit = storageUnitFor(resolveBudgetDefault(spec?.rom_kb ?? null, spec?.device_class ?? null, "rom_kb").value);
    persist({
      target_device_slug: selectedSlug,
      target_device_custom_name: customName.trim() || null,
      target_device_ram_kb: ramParsed.value === null ? null : displayToKb(ramParsed.value, ramUnit),
      target_device_rom_kb: romParsed.value === null ? null : displayToKb(romParsed.value, romUnit),
      target_device_latency_ms: latencyParsed.value === null ? null : Math.round(latencyParsed.value),
    });
  }

  function handleReset() {
    if (!selectedSlug) return;
    persist({
      target_device_slug: selectedSlug,
      target_device_custom_name: null,
      target_device_ram_kb: null,
      target_device_rom_kb: null,
      target_device_latency_ms: null,
    });
  }

  if (!mounted || !open || !activeProject) return null;

  const deviceClass = spec?.device_class ?? null;
  const ramDefault = resolveBudgetDefault(spec?.ram_kb ?? null, deviceClass, "ram_kb");
  const romDefault = resolveBudgetDefault(spec?.rom_kb ?? null, deviceClass, "rom_kb");
  const latencyDefault = resolveBudgetDefault(
    spec?.latency_budget_ms ?? null, deviceClass, "latency_budget_ms",
  );
  const ramUnit = storageUnitFor(ramDefault.value);
  const romUnit = storageUnitFor(romDefault.value);
  const clock = splitClock(spec?.clock_rate_mhz ?? null);

  // Portaled to document.body: `.shell-topbar` (this dialog's DOM parent via
  // TargetDeviceChip → TopBar) sets backdrop-filter, which makes it the
  // containing block for `position: fixed` descendants and traps a
  // same-subtree overlay inside the ~64px topbar box instead of the viewport.
  // Rendering inline here reproduces that — don't "simplify" this away.
  return createPortal(
    <div
      className="pe-tdconf-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="pe-tdconf-title"
    >
      <div
        ref={modalRef}
        className="pe-tdconf-modal"
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="pe-tdconf-head">
          <h3 id="pe-tdconf-title" className="pe-tdconf-title">
            <span className="pe-tdconf-title-icon" aria-hidden="true">
              <Cpu size={16} strokeWidth={2.1} />
            </span>
            Configure your target device and application budget
          </h3>
          <button type="button" onClick={onClose} className="pe-tdconf-close" aria-label="Close">
            <X size={16} />
          </button>
        </div>

        <div className="pe-tdconf-body">
          <section className="pe-tdconf-section">
            <h4 className="pe-tdconf-section-title">Target device</h4>
            <p className="pe-tdconf-intro">
              Choose the hardware this project targets. Selecting a board fills in its
              processor specification and sets the defaults for the application budget
              below. You can change it at any time.
            </p>

            <Row label="Board" labelId="pe-tdconf-board-label">
              <BoardPicker
                groups={groups}
                loading={loadingCatalog}
                selectedSlug={selectedSlug}
                unresolvedSlug={isUnresolved ? selectedSlug : null}
                onSelect={handleSelectBoard}
                labelledBy="pe-tdconf-board-label"
              />
            </Row>

            {selectedSlug === null ? (
              <p className="pe-tdconf-hint">Select a board to see its specification.</p>
            ) : loadingCatalog && groups.length === 0 ? (
              <p className="pe-tdconf-hint">Loading specification…</p>
            ) : (
              <>
                <SpecRow label="Processor family" value={spec?.processor_family ?? null} />
                <SpecRow label="Processor" value={spec?.processor ?? null} />
                <SpecRow label="CPU architecture" value={spec?.cpu_architecture ?? null} />
                <SpecRow
                  label="Clock rate"
                  value={clock?.value ?? null}
                  unit={clock?.unit ?? null}
                  help="The board's maximum CPU frequency, read from the device catalog. It isn't editable here."
                />
              </>
            )}

            <Row
              label="Custom device name"
              htmlFor="pe-tdconf-name"
              optional
              help="A name for this particular unit. It replaces the board name wherever the project shows your target device."
            >
              <input
                id="pe-tdconf-name"
                type="text"
                className="pe-tdconf-input"
                value={customName}
                onChange={(e) => setCustomName(e.target.value)}
                placeholder={selectedEntry ? selectedEntry.display_name : "e.g. Line 3 sensor node"}
                disabled={!hasDevice}
                maxLength={120}
              />
            </Row>
          </section>

          <section className="pe-tdconf-section">
            <h4 className="pe-tdconf-section-title">Application budget</h4>
            <p className="pe-tdconf-intro">
              Set the RAM, ROM and latency your model may use on this device. Leave a
              field blank to keep the board's own figure.
            </p>

            {!hasDevice ? (
              <p className="pe-tdconf-hint">
                Select a target device above to configure its application budget.
              </p>
            ) : (
              <>
                <BudgetField
                  label="RAM"
                  unitLabel={ramUnit}
                  text={ramText}
                  onTextChange={setRamText}
                  boardDefaultDisplay={ramDefault.value != null ? kbToDisplay(ramDefault.value, ramUnit) : null}
                  isClassDefault={ramDefault.isClassDefault}
                  invalid={ramParsed.invalid}
                />
                <BudgetField
                  label="ROM / Flash"
                  unitLabel={romUnit}
                  text={romText}
                  onTextChange={setRomText}
                  boardDefaultDisplay={romDefault.value != null ? kbToDisplay(romDefault.value, romUnit) : null}
                  isClassDefault={romDefault.isClassDefault}
                  invalid={romParsed.invalid}
                />
                <BudgetField
                  label="Latency budget"
                  unitLabel="ms"
                  text={latencyText}
                  onTextChange={setLatencyText}
                  boardDefaultDisplay={latencyDefault.value}
                  isClassDefault={latencyDefault.isClassDefault}
                  invalid={latencyParsed.invalid}
                  help="The longest time one inference may take on this device before your application misses its deadline."
                />
              </>
            )}
          </section>

          {saveError && <p className="pe-tdconf-error">{saveError}</p>}
        </div>

        <div className="pe-tdconf-foot">
          <button
            type="button"
            className="pe-tdconf-reset"
            onClick={handleReset}
            disabled={!hasDevice || saving}
          >
            <RotateCcw size={14} aria-hidden="true" />
            Reset to default settings
          </button>
          <button
            type="button"
            className="pe-tdconf-save"
            onClick={handleSave}
            disabled={!canSave || saving}
          >
            {saving ? (
              <Loader2 size={14} className="pe-tdconf-spin" aria-hidden="true" />
            ) : (
              <Check size={14} aria-hidden="true" />
            )}
            Save
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

function fieldId(label: string): string {
  return `pe-tdconf-f-${label.replace(/[^a-z0-9]+/gi, "-").toLowerCase()}`;
}

/** The one row shape every field in the dialog uses: label in the left
 *  column, control block in the right, the label centred against the first
 *  line of the control (the input itself, not its caption). Read-only specs
 *  obey it too — that shared alignment is what holds the dialog together. */
function Row({
  label,
  htmlFor,
  labelId,
  optional,
  help,
  children,
}: {
  label: string;
  /** Set for rows whose control is a real form element. */
  htmlFor?: string;
  /** Set instead for controls labelled by reference (the board picker's
   *  button, which can't be the target of a `for`). */
  labelId?: string;
  optional?: boolean;
  help?: string;
  children: ReactNode;
}) {
  const helpId = `${htmlFor ?? labelId ?? fieldId(label)}-help`;
  return (
    <div className="pe-tdconf-row">
      {/* The help button is interactive, so it sits beside the <label>, never
          inside it — a button nested in a label forwards its click to the
          labelled control. */}
      <div className="pe-tdconf-row-label">
        {htmlFor ? (
          <label className="pe-tdconf-label" htmlFor={htmlFor}>
            {label}
          </label>
        ) : (
          <span className="pe-tdconf-label" id={labelId}>
            {label}
          </span>
        )}
        {optional && <span className="pe-tdconf-optional">(optional)</span>}
        {help && <HelpTip id={helpId} subject={label} text={help} />}
      </div>
      <div className="pe-tdconf-row-control">{children}</div>
    </div>
  );
}

/** A real tooltip, not a decorative glyph: the button carries an accessible
 *  name, the bubble is its description, and it opens on focus as well as
 *  hover so it's reachable from the keyboard inside the dialog's focus trap. */
function HelpTip({ id, subject, text }: { id: string; subject: string; text: string }) {
  return (
    <span className="pe-tdconf-help">
      <button
        type="button"
        className="pe-tdconf-help-btn"
        aria-label={`About ${subject.toLowerCase()}`}
        aria-describedby={id}
      >
        <HelpCircle size={13} aria-hidden="true" />
      </button>
      <span className="pe-tdconf-help-bubble" role="tooltip" id={id}>
        {text}
      </span>
    </span>
  );
}

/** A board specification value. Nothing here is editable — the project has
 *  no storage for these at all — so it renders as an inert filled control:
 *  same shape as the rows around it, visibly not a field you can type in. */
function SpecRow({
  label,
  value,
  unit,
  help,
}: {
  label: string;
  value: string | null;
  unit?: string | null;
  help?: string;
}) {
  const id = fieldId(label);
  const unitId = `${id}-unit`;
  return (
    <Row label={label} htmlFor={id} help={help}>
      <div className="pe-tdconf-field-wrap">
        <input
          id={id}
          type="text"
          readOnly
          aria-disabled="true"
          aria-describedby={unit ? unitId : undefined}
          className={`pe-tdconf-input pe-tdconf-input--inert${value === null ? " is-unknown" : ""}${
            unit ? " pe-tdconf-input--unit" : ""
          }`}
          value={value ?? "Unknown"}
        />
        {unit && (
          <span className="pe-tdconf-unit" id={unitId}>
            {unit}
          </span>
        )}
      </div>
    </Row>
  );
}

function BudgetField({
  label,
  unitLabel,
  text,
  onTextChange,
  boardDefaultDisplay,
  isClassDefault,
  invalid,
  help,
}: {
  label: string;
  unitLabel: string;
  text: string;
  onTextChange: (t: string) => void;
  boardDefaultDisplay: number | null;
  isClassDefault: boolean;
  invalid: boolean;
  help?: string;
}) {
  const overridden = text.trim() !== "" && !invalid;
  // Genuinely no figure applies (e.g. RAM/ROM for a host-attached accelerator
  // chip, which has no on-board memory budget of its own) — distinct from
  // "unsourced", which the class-level fallback in the parent already covers.
  const defaultText =
    boardDefaultDisplay === null
      ? "Not applicable for this device"
      : isClassDefault
        ? `~${formatDisplay(boardDefaultDisplay)} ${unitLabel} (typical for this device class)`
        : `${formatDisplay(boardDefaultDisplay)} ${unitLabel}`;
  const numericDefault = boardDefaultDisplay === null ? null : parseFloat(formatDisplay(boardDefaultDisplay));
  const numericValue = invalid ? null : (text.trim() === "" ? null : Number(text));
  const exceedsDefault =
    overridden && numericDefault !== null && numericValue !== null && numericValue > numericDefault;

  const inputId = fieldId(label);
  const unitId = `${inputId}-unit`;
  const captionId = `${inputId}-caption`;

  return (
    <Row label={label} htmlFor={inputId} help={help}>
      <div
        className={`pe-tdconf-field-wrap${overridden ? " is-overridden" : ""}${invalid ? " is-invalid" : ""}`}
      >
        <input
          id={inputId}
          type="text"
          inputMode="decimal"
          className="pe-tdconf-input pe-tdconf-input--unit"
          value={text}
          onChange={(e) => onTextChange(e.target.value)}
          placeholder={boardDefaultDisplay === null ? "N/A" : formatDisplay(boardDefaultDisplay)}
          aria-invalid={invalid}
          aria-describedby={`${unitId} ${captionId}`}
        />
        {/* Decoration on the field, not part of its value — the unit is fixed
            by the board's own figure and never round-trips through state. */}
        <span className="pe-tdconf-unit" id={unitId}>
          {unitLabel}
        </span>
      </div>
      <p className="pe-tdconf-caption" id={captionId}>
        {overridden && <span className="pe-tdconf-customised-pill">Customised</span>}
        <span>Board default: {defaultText}</span>
        {exceedsDefault && <span className="pe-tdconf-exceeds-note">· exceeds board default</span>}
      </p>
      {invalid && (
        <p className="pe-tdconf-field-error">
          Enter a positive number, or leave blank to use the board's value.
        </p>
      )}
    </Row>
  );
}

/** Whether Petal Edge currently ships a deployment package for this exact
 *  board — a fixed, per-board fact (`DEPLOYMENT_READY_SLUGS`), never derived
 *  from `deploy_target`: several boards share a `deploy_target` with a board
 *  that has a package without shipping one themselves. */
function isDeploymentReady(entry: CatalogEntry): boolean {
  return DEPLOYMENT_READY_SLUGS.includes(entry.slug);
}

function boardRowLabel(entry: CatalogEntry): string {
  return entry.family === GENERIC_FAMILY ? "Custom" : entry.display_name;
}

function BoardPicker({
  groups,
  loading,
  selectedSlug,
  unresolvedSlug,
  onSelect,
  labelledBy,
}: {
  groups: FamilyGroup[];
  loading: boolean;
  selectedSlug: string | null;
  unresolvedSlug: string | null;
  onSelect: (slug: string | null) => void;
  labelledBy: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!open) {
      setQuery("");
      return;
    }
    searchRef.current?.focus();
    function onDocClick(e: MouseEvent) {
      if (wrapRef.current?.contains(e.target as Node)) return;
      setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  const production = groups.find((g) => g.family === GENERIC_FAMILY)?.entries ?? [];
  const development = groups.filter((g) => g.family !== GENERIC_FAMILY);
  const allFlatEntries = [...production, ...development.flatMap((g) => g.entries)];

  // Pinned above "Production device": the exact boards in
  // `DEPLOYMENT_READY_SLUGS` — from either group — in that list's order
  // (Petal Edge's own hardware first).
  const deploymentReady = DEPLOYMENT_READY_SLUGS
    .map((slug) => allFlatEntries.find((e) => e.slug === slug))
    .filter((e): e is CatalogEntry => !!e);

  const q = query.trim().toLowerCase();
  const filteredProduction = q
    ? production.filter((e) => "custom".includes(q) || e.display_name.toLowerCase().includes(q))
    : production;
  const filteredDevelopment = q
    ? development
        .map((g) => ({
          family: g.family,
          entries: g.entries.filter(
            (e) => e.display_name.toLowerCase().includes(q) || g.family.toLowerCase().includes(q),
          ),
        }))
        .filter((g) => g.entries.length > 0)
    : development;
  const filteredDeploymentReady = q
    ? deploymentReady.filter(
        (e) =>
          boardRowLabel(e).toLowerCase().includes(q) ||
          e.family.toLowerCase().includes(q) ||
          (e.family === GENERIC_FAMILY && "custom".includes(q)),
      )
    : deploymentReady;

  const selectedEntry = allFlatEntries.find((e) => e.slug === selectedSlug);
  const triggerLabel =
    selectedSlug === null
      ? "No target device selected"
      : unresolvedSlug
        ? `${selectedSlug} (retired board)`
        : selectedEntry
          ? (selectedEntry.family === GENERIC_FAMILY ? "Custom" : selectedEntry.display_name)
          : "Loading…";

  return (
    <div className="pe-tdconf-board-wrap" ref={wrapRef}>
      <button
        type="button"
        className={`pe-tdconf-board-trigger${selectedSlug === null ? " is-empty" : ""}`}
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-labelledby={labelledBy}
      >
        <span>{triggerLabel}</span>
        <ChevronDown size={15} aria-hidden="true" />
      </button>

      {open && (
        <div className="pe-tdconf-board-panel" role="listbox" aria-label="Target device">
          <input
            ref={searchRef}
            type="text"
            className="pe-tdconf-board-search"
            placeholder="Search boards"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Search boards"
          />
          <div className="pe-tdconf-board-list">
            <button
              type="button"
              role="option"
              aria-selected={selectedSlug === null}
              className="pe-tdconf-board-option pe-tdconf-board-option--clear"
              onClick={() => {
                onSelect(null);
                setOpen(false);
              }}
            >
              No target device selected
            </button>

            {loading && groups.length === 0 ? (
              <p className="pe-tdconf-board-empty">Loading the device catalog…</p>
            ) : filteredProduction.length === 0 &&
              filteredDevelopment.length === 0 &&
              filteredDeploymentReady.length === 0 ? (
              <p className="pe-tdconf-board-empty">No board matches "{query}".</p>
            ) : (
              <>
                {filteredDeploymentReady.length > 0 && (
                  <div role="group" aria-label="Deployment Ready">
                    <div className="pe-tdconf-board-eyebrow">Deployment Ready</div>
                    {filteredDeploymentReady.map((entry) => (
                      <button
                        key={`ready-${entry.slug}`}
                        type="button"
                        role="option"
                        aria-selected={entry.slug === selectedSlug}
                        className="pe-tdconf-board-option"
                        onClick={() => {
                          onSelect(entry.slug);
                          setOpen(false);
                        }}
                      >
                        <span className="pe-tdconf-board-option-label">{boardRowLabel(entry)}</span>
                      </button>
                    ))}
                  </div>
                )}

                {filteredProduction.length > 0 && (
                  <div role="group" aria-label="Production device">
                    <div className="pe-tdconf-board-eyebrow">Production device</div>
                    {filteredProduction.map((entry) => {
                      const ready = isDeploymentReady(entry);
                      return (
                        <button
                          key={entry.slug}
                          type="button"
                          role="option"
                          aria-selected={entry.slug === selectedSlug}
                          className="pe-tdconf-board-option"
                          onClick={() => {
                            onSelect(entry.slug);
                            setOpen(false);
                          }}
                        >
                          <span className="pe-tdconf-board-option-label">Custom</span>
                          {!ready && (
                            <span
                              className="pe-tdconf-board-nopkg"
                              title="The deployment package for this board is not implemented yet."
                            >
                              No deployment package yet
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                )}

                {filteredDevelopment.length > 0 && (
                  <div role="group" aria-label="Development boards">
                    <div className="pe-tdconf-board-eyebrow">Development boards</div>
                    {filteredDevelopment.map((g) => (
                      <div key={g.family} role="group" aria-label={g.family}>
                        <div className="pe-tdconf-board-family">{g.family}</div>
                        {g.entries.map((entry) => {
                          const ready = isDeploymentReady(entry);
                          return (
                            <button
                              key={entry.slug}
                              type="button"
                              role="option"
                              aria-selected={entry.slug === selectedSlug}
                              className="pe-tdconf-board-option"
                              onClick={() => {
                                onSelect(entry.slug);
                                setOpen(false);
                              }}
                            >
                              <span className="pe-tdconf-board-option-label">{entry.display_name}</span>
                              {!ready && (
                                <span
                                  className="pe-tdconf-board-nopkg"
                                  title="The deployment package for this board is not implemented yet."
                                >
                                  No deployment package yet
                                </span>
                              )}
                            </button>
                          );
                        })}
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
