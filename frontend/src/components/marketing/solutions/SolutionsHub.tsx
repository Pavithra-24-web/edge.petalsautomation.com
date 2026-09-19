"use client";

import { useEffect, useRef } from "react";
import { ArrowRight, Compass } from "lucide-react";
import "../../../app/marketing.css";
import { useHomeTheme, Reveal } from "../primitives";
import HomeBackground from "../HomeBackground";
import Navbar from "../Navbar";
import { Footer } from "../Closing";
import SolutionWhy from "./SolutionWhy";
import { SolIcon } from "./icons";
import {
  GROUPS,
  HUB,
  CTA_ACTIONS,
  solutionsByGroup,
  solutionHref,
} from "@/lib/solutions.config";

/* ---------------------------------------------------------------------------
   /solutions — hub. Lists all nine solutions in their three groups, using the
   same shell and primitives as every other marketing page.
--------------------------------------------------------------------------- */
export default function SolutionsHub() {
  const [theme, toggleTheme] = useHomeTheme();
  const glowRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (window.matchMedia?.("(pointer: coarse)").matches) return;
    let raf = 0;
    const onMove = (e: PointerEvent) => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const el = glowRef.current;
        if (el) el.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
      });
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => {
      window.removeEventListener("pointermove", onMove);
      cancelAnimationFrame(raf);
    };
  }, []);

  const [primary, secondary] = CTA_ACTIONS;

  return (
    <div className="pe-home" data-theme={theme}>
      <HomeBackground />
      <div ref={glowRef} className="pe-cursor-glow" aria-hidden="true" />

      <Navbar theme={theme} onToggleTheme={toggleTheme} />

      <main className="pe-boost">
        <section className="pe-section pe-sol-hero" id="top">
          <div className="pe-container">
            <div className="pe-section-head">
              <Reveal as="span" className="pe-eyebrow"><Compass /> {HUB.eyebrow}</Reveal>
              <Reveal as="h1" className="pe-h1 pe-sol-h1 pe-sol-h1-center" delay={60}>
                {HUB.title}
                <span className="pe-grad-text">{HUB.titleAccent}</span>
              </Reveal>
              {/* Deliberately not a capability list: SolutionWhy renders lower on
                  this same page and already enumerates the pipeline. This line
                  says how the nine pages are organised, nothing else. */}
              <Reveal as="p" className="pe-lede" delay={120}>{HUB.lede}</Reveal>
              <Reveal className="pe-sol-ctas pe-sol-ctas-center" delay={180}>
                <a className="pe-btn pe-btn-primary pe-btn-lg" href={primary.href}>
                  {primary.label} <ArrowRight />
                </a>
                <a className="pe-btn pe-btn-ghost pe-btn-lg" href={secondary.href}>
                  {secondary.label}
                </a>
              </Reveal>
            </div>
          </div>
        </section>

        {GROUPS.map((group) => (
          <section key={group.id} className="pe-section" id={group.id}>
            <div className="pe-container">
              <div className="pe-section-head">
                <Reveal as="span" className="pe-eyebrow">
                  <SolIcon name={group.icon} /> {group.label}
                </Reveal>
                <Reveal as="h2" className="pe-h2" delay={40}>{group.kicker}</Reveal>
                <Reveal as="p" className="pe-lede" delay={60}>{group.desc}</Reveal>
              </div>

              <div className="pe-sol-hub-grid">
                {solutionsByGroup(group.id).map((s, i) => (
                  <Reveal key={s.slug} delay={(i % 3) * 70}>
                    <a
                      className="pe-card pe-card-hover pe-sol-hub-card"
                      href={solutionHref(s.slug)}
                    >
                      <h3>{s.name}</h3>
                      <p>{s.tagline}</p>
                      <span className="pe-sol-hub-more">
                        Explore <ArrowRight />
                      </span>
                    </a>
                  </Reveal>
                ))}
              </div>
            </div>
          </section>
        ))}

        <SolutionWhy />
      </main>

      <Footer />
    </div>
  );
}
