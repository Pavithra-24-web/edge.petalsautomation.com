"use client";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Save, ChevronDown, ArrowRight } from "lucide-react";

/** Primary `Save parameters` action plus a secondary `Save and generate
 *  features` action, presented as a single right-aligned split button
 *  (`[ Save parameters ▾]`) matching the reference layout. The chevron opens
 *  a one-item menu rather than a second full-width button. Both paths only
 *  save; the "and generate" path then navigates to the Generate features
 *  tab — it never dispatches a generation job (out of scope for this batch).
 *  The menu itself is portalled to `document.body`: the surrounding
 *  `.pe-dsp-card` sets `overflow: hidden` for its rounded corners, which
 *  would otherwise clip an absolutely-positioned dropdown. */
export default function SaveParametersButton({
  onSave,
  onSaved,
  onSavedAndGenerate,
  disabled,
}: {
  onSave: () => Promise<void>;
  onSaved: () => void;
  onSavedAndGenerate: () => void;
  disabled?: boolean;
}) {
  const [saving, setSaving] = useState<"save" | "generate" | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [menuPos, setMenuPos] = useState({ top: 0, right: 0 });
  const inflight = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!menuOpen) return;
    const onClick = (e: MouseEvent) => {
      const target = e.target as Node;
      if (containerRef.current?.contains(target)) return;
      if (menuRef.current?.contains(target)) return;
      setMenuOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [menuOpen]);

  const openMenu = () => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (rect) {
      setMenuPos({ top: rect.bottom + 6, right: window.innerWidth - rect.right });
    }
    setMenuOpen((v) => !v);
  };

  const run = async (mode: "save" | "generate") => {
    if (inflight.current || disabled) return;
    inflight.current = true;
    setSaving(mode);
    try {
      await onSave();
      if (mode === "save") onSaved();
      else onSavedAndGenerate();
    } finally {
      setSaving(null);
      inflight.current = false;
    }
  };

  const busy = disabled || saving !== null;

  return (
    <div className="flex justify-end" style={{ marginTop: 12 }}>
      <div className="relative inline-flex" ref={containerRef}>
        <button
          type="button"
          className="pe-save-btn"
          style={{ borderTopRightRadius: 0, borderBottomRightRadius: 0 }}
          disabled={busy}
          aria-busy={saving === "save"}
          onClick={() => run("save")}
        >
          {saving === "save" ? <span className="pe-save-spinner" aria-hidden="true" /> : <Save size={14} aria-hidden="true" />}
          <span>{saving === "save" ? "Saving..." : "Save parameters"}</span>
        </button>
        <button
          type="button"
          className="pe-save-btn"
          style={{ borderTopLeftRadius: 0, borderBottomLeftRadius: 0, borderLeft: "1px solid rgba(255,255,255,0.25)", padding: "0.55rem 0.6rem" }}
          disabled={busy}
          aria-label="More save options"
          aria-expanded={menuOpen}
          onClick={openMenu}
        >
          <ChevronDown size={14} aria-hidden="true" />
        </button>
      </div>

      {menuOpen && typeof document !== "undefined" &&
        createPortal(
          <div
            ref={menuRef}
            className="fixed z-50"
            style={{
              top: menuPos.top,
              right: menuPos.right,
              minWidth: 260,
              background: "var(--app-surface)",
              border: "1px solid var(--app-border)",
              borderRadius: 8,
              boxShadow: "0 10px 30px -10px rgba(0,0,0,0.45)",
              padding: 4,
            }}
          >
            <button
              type="button"
              className="pe-row-menu-item"
              onClick={() => {
                setMenuOpen(false);
                run("generate");
              }}
            >
              <ArrowRight size={14} aria-hidden="true" />
              <span>Save parameters and generate features</span>
            </button>
          </div>,
          document.body
        )}
    </div>
  );
}
