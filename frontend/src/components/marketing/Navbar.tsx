"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import { Sun, Moon, Menu, X, ChevronDown } from "lucide-react";
import PetalEdgeLogo from "@/components/PetalEdgeLogo";
import type { HomeTheme } from "./primitives";
import { GROUPS, solutionsByGroup, solutionHref } from "@/lib/solutions.config";

type NavLink =
  | { label: string; href: string; kind?: undefined }
  | { label: string; href: string; kind: "solutions" };

/* Solutions sits between Platform and Model — it is the "what can I do with
   this" entry point, so it comes before the model detail pages. */
const NAV_LINKS: NavLink[] = [
  { label: "Platform", href: "/#platform" },
  { label: "Solutions", href: "/solutions/", kind: "solutions" },
  { label: "Model", href: "/model" },
  { label: "Features", href: "/features" },
  { label: "Pricing", href: "/pricing" },
  { label: "Contact us", href: "/contact" },
];

/** The three groups, each with its solutions, built once from the config. */
const SOLUTION_COLUMNS = GROUPS.map((g) => ({
  ...g,
  items: solutionsByGroup(g.id),
}));

export default function Navbar({
  theme,
  onToggleTheme,
}: {
  theme: HomeTheme;
  onToggleTheme: () => void;
}) {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const [solOpen, setSolOpen] = useState(false);
  const solRef = useRef<HTMLLIElement | null>(null);
  const solBtnRef = useRef<HTMLButtonElement | null>(null);
  const pathname = usePathname();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Any route change closes both menus — otherwise the panel survives a
  // client-side navigation and hangs over the new page.
  useEffect(() => {
    setSolOpen(false);
    setOpen(false);
  }, [pathname]);

  // Escape closes the dropdown and returns focus to the trigger; a pointer
  // press anywhere outside dismisses it.
  useEffect(() => {
    if (!solOpen) return;

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setSolOpen(false);
        solBtnRef.current?.focus();
      }
    };
    const onPointerDown = (e: PointerEvent) => {
      if (!solRef.current?.contains(e.target as Node)) setSolOpen(false);
    };

    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onPointerDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onPointerDown);
    };
  }, [solOpen]);

  // Tabbing out of the dropdown closes it, so keyboard users are never left
  // with an invisible open panel behind them.
  const onSolBlur = useCallback((e: React.FocusEvent<HTMLLIElement>) => {
    if (!e.currentTarget.contains(e.relatedTarget as Node)) setSolOpen(false);
  }, []);

  return (
    <>
      <header className={`pe-nav ${scrolled ? "scrolled" : ""}`}>
        <div className="pe-container">
          <nav className="pe-nav-inner" aria-label="Primary">
            <button
              className="pe-icon-btn pe-burger"
              aria-label="Toggle menu"
              aria-expanded={open}
              onClick={() => setOpen((v) => !v)}
            >
              {open ? <X /> : <Menu />}
            </button>

            <a href="#top" className="pe-brand" aria-label="Petal Edge — home">
              <PetalEdgeLogo variant="icon" size={40} label="" />
              <span className="pe-brand-name" aria-hidden="true">
                Petal&nbsp;<span className="pe-grad-text">Edge</span>
              </span>
            </a>

            <ul className="pe-nav-links">
              {NAV_LINKS.map((l) =>
                l.kind === "solutions" ? (
                  <li
                    key={l.label}
                    ref={solRef}
                    className="pe-nav-drop-wrap"
                    onBlur={onSolBlur}
                    onMouseEnter={() => setSolOpen(true)}
                    onMouseLeave={() => setSolOpen(false)}
                  >
                    <button
                      ref={solBtnRef}
                      type="button"
                      className={`pe-nav-link pe-nav-drop-btn ${solOpen ? "open" : ""}`}
                      aria-expanded={solOpen}
                      aria-haspopup="true"
                      onClick={() => setSolOpen((v) => !v)}
                    >
                      {l.label} <ChevronDown aria-hidden="true" />
                    </button>

                    <div
                      className={`pe-nav-drop ${solOpen ? "open" : ""}`}
                      // Hidden from assistive tech and taken out of the tab order
                      // when closed, so a keyboard user never lands inside a
                      // panel they cannot see.
                      hidden={!solOpen}
                    >
                      <div className="pe-nav-drop-grid">
                        {SOLUTION_COLUMNS.map((col) => (
                          <div key={col.id} className="pe-nav-drop-col">
                            <p className="pe-nav-drop-head">
                              <span className="pe-nav-drop-label">{col.label}</span>
                              <span className="pe-nav-drop-kicker">{col.kicker}</span>
                            </p>
                            <ul>
                              {col.items.map((s) => (
                                <li key={s.slug}>
                                  <a href={solutionHref(s.slug)}>{s.name}</a>
                                </li>
                              ))}
                            </ul>
                          </div>
                        ))}
                      </div>
                      <a className="pe-nav-drop-all" href="/solutions/">
                        All solutions
                      </a>
                    </div>
                  </li>
                ) : (
                  <li key={l.label}>
                    <a className="pe-nav-link" href={l.href}>
                      {l.label}
                    </a>
                  </li>
                )
              )}
            </ul>

            <div className="pe-nav-right">
              <button
                className="pe-icon-btn"
                aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
                onClick={onToggleTheme}
              >
                {theme === "dark" ? <Sun /> : <Moon />}
              </button>
              <a href="/login/" className="pe-nav-login pe-desktop-only">
                Login
              </a>
              <a href="/book-demo" className="pe-btn pe-btn-ghost pe-btn-sm pe-desktop-only">
                Book Demo
              </a>
              <a href="/login/" className="pe-btn pe-btn-primary pe-btn-sm">
                Get Started
              </a>
            </div>
          </nav>
        </div>
      </header>

      <div className={`pe-mobile-panel ${open ? "open" : ""}`}>
        {NAV_LINKS.map((l) =>
          l.kind === "solutions" ? (
            // Mobile gets the same three groups as labelled stacked sections
            // rather than a nested dropdown — one scroll, no second interaction.
            <div key={l.label} className="pe-mobile-groups">
              {SOLUTION_COLUMNS.map((col) => (
                <section key={col.id} className="pe-mobile-group" aria-label={col.label}>
                  {/* Not a heading element: the panel renders before <main>, so
                      real headings here would open the document outline at h3
                      ahead of the page's own h1. aria-label on the section
                      carries the grouping for assistive tech instead. */}
                  <p className="pe-mobile-group-head">
                    <span>{col.label}</span>
                    <span className="pe-nav-drop-kicker">{col.kicker}</span>
                  </p>
                  {col.items.map((s) => (
                    <a
                      key={s.slug}
                      href={solutionHref(s.slug)}
                      onClick={() => setOpen(false)}
                    >
                      {s.name}
                    </a>
                  ))}
                </section>
              ))}
            </div>
          ) : (
            <a key={l.label} href={l.href} onClick={() => setOpen(false)}>
              {l.label}
            </a>
          )
        )}
        <div className="pe-mobile-cta">
          <a href="/login/" className="pe-btn pe-btn-ghost">
            Login
          </a>
          <a href="/login/" className="pe-btn pe-btn-primary">
            Get Started
          </a>
        </div>
      </div>
    </>
  );
}
