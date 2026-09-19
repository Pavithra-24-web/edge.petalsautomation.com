"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Search } from "lucide-react";

export type DeviceOption = {
  value: string;
  label: string;
  category: string;
  description: string;
  Icon: React.ComponentType<any>;
};

type Props = {
  options: DeviceOption[];
  value: string;
  onChange: (value: string) => void;
  categoryOrder: string[];
};

export default function DeviceSelector({ options, value, onChange, categoryOrder }: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const selected = options.find((o) => o.value === value) ?? options[0];

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return options;
    return options.filter(
      (o) =>
        o.label.toLowerCase().includes(q) ||
        o.category.toLowerCase().includes(q) ||
        o.description.toLowerCase().includes(q),
    );
  }, [options, query]);

  const grouped = useMemo(() => {
    const map = new Map<string, DeviceOption[]>();
    for (const opt of filtered) {
      if (!map.has(opt.category)) map.set(opt.category, []);
      map.get(opt.category)!.push(opt);
    }
    const ordered = categoryOrder
      .filter((c) => map.has(c))
      .map((c) => [c, map.get(c)!] as const);
    for (const [k, v] of map) {
      if (!categoryOrder.includes(k)) ordered.push([k, v] as const);
    }
    return ordered;
  }, [filtered, categoryOrder]);

  const flatOptions = useMemo(() => grouped.flatMap(([, items]) => items), [grouped]);

  useEffect(() => {
    if (open) {
      const i = flatOptions.findIndex((o) => o.value === value);
      setActiveIndex(i >= 0 ? i : 0);
      setQuery("");
      setTimeout(() => searchRef.current?.focus(), 0);
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function onDocClick(e: MouseEvent) {
      if (!containerRef.current?.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const list = listRef.current;
    if (!list) return;
    const activeEl = list.querySelector<HTMLElement>(`[data-flat-index="${activeIndex}"]`);
    activeEl?.scrollIntoView({ block: "nearest" });
  }, [activeIndex, open]);

  function commit(opt: DeviceOption) {
    onChange(opt.value);
    setOpen(false);
    setTimeout(() => triggerRef.current?.focus(), 0);
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Escape") {
      e.preventDefault();
      setOpen(false);
      triggerRef.current?.focus();
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, flatOptions.length - 1));
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
      return;
    }
    if (e.key === "Home") {
      e.preventDefault();
      setActiveIndex(0);
      return;
    }
    if (e.key === "End") {
      e.preventDefault();
      setActiveIndex(flatOptions.length - 1);
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      const opt = flatOptions[activeIndex];
      if (opt) commit(opt);
    }
  }

  function onTriggerKeyDown(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown" || e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setOpen(true);
    }
  }

  const SelectedIcon = selected?.Icon;

  return (
    <div
      ref={containerRef}
      className={`pe-dep2-combo ${open ? "pe-dep2-combo--open" : ""}`}
    >
      <button
        ref={triggerRef}
        type="button"
        className="pe-dep2-combo-trigger"
        onClick={() => setOpen((v) => !v)}
        onKeyDown={onTriggerKeyDown}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label="Select target device"
      >
        {SelectedIcon && (
          <span className="pe-dep2-combo-option-icon" aria-hidden="true">
            <SelectedIcon size={16} />
          </span>
        )}
        <span className="pe-dep2-combo-trigger-body">
          <span className="pe-dep2-combo-trigger-name">
            {selected?.label ?? "Select a device"}
          </span>
          <span className="pe-dep2-combo-trigger-meta">
            {selected ? `${selected.category} · ${selected.description}` : "Choose a target to deploy"}
          </span>
        </span>
        <ChevronDown size={16} className="pe-dep2-combo-chevron" aria-hidden="true" />
      </button>

      {open && (
        <div className="pe-dep2-combo-popover" role="dialog" aria-label="Device options">
          <div className="pe-dep2-combo-search">
            <Search size={14} aria-hidden="true" className="text-current opacity-60" />
            <input
              ref={searchRef}
              type="text"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setActiveIndex(0);
              }}
              onKeyDown={onKeyDown}
              placeholder="Search devices…"
              aria-label="Search devices"
              aria-controls="pe-dep2-combo-listbox"
              aria-activedescendant={
                flatOptions[activeIndex] ? `pe-dep2-opt-${flatOptions[activeIndex].value}` : undefined
              }
            />
          </div>

          {flatOptions.length === 0 ? (
            <div className="pe-dep2-combo-empty">No devices match “{query}”.</div>
          ) : (
            <ul
              ref={listRef}
              id="pe-dep2-combo-listbox"
              role="listbox"
              aria-label="Devices grouped by category"
              className="pe-dep2-combo-list"
            >
              {grouped.map(([category, items]) => (
                <li key={category} role="group" aria-label={category}>
                  <div className="pe-dep2-combo-group-label" role="presentation">
                    {category}
                  </div>
                  <ul role="presentation" className="m-0 p-0 list-none">
                    {items.map((opt) => {
                      const flatIdx = flatOptions.indexOf(opt);
                      const isActive = flatIdx === activeIndex;
                      const isSelected = opt.value === value;
                      const OptIcon = opt.Icon;
                      return (
                        <li
                          key={opt.value}
                          id={`pe-dep2-opt-${opt.value}`}
                          role="option"
                          aria-selected={isSelected}
                          data-active={isActive}
                          data-flat-index={flatIdx}
                          className="pe-dep2-combo-option"
                          onMouseEnter={() => setActiveIndex(flatIdx)}
                          onClick={() => commit(opt)}
                        >
                          <span className="pe-dep2-combo-option-icon" aria-hidden="true">
                            <OptIcon size={16} />
                          </span>
                          <span className="pe-dep2-combo-option-body">
                            <span className="pe-dep2-combo-option-name">{opt.label}</span>
                            <span className="pe-dep2-combo-option-sub">{opt.description}</span>
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
