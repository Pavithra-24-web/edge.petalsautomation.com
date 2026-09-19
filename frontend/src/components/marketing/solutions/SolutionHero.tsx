"use client";

import { ArrowRight } from "lucide-react";
import { Reveal } from "../primitives";
import { SolIcon } from "./icons";
import SolutionMedia from "./SolutionMedia";
import { GROUPS, CTA_ACTIONS, type Solution } from "@/lib/solutions.config";

/* ---------------------------------------------------------------------------
   Section 1 — Hero.
   Fixed CTA pair (Start Free + Request Demo); everything else per-solution.

   `panel` is the visual slot on the right. It defaults to the solution's own
   hero asset from the config, and a page may pass a composed component instead
   — see VisionStill on /solutions/computer-vision. Either way the grid only
   splits into two columns when there is something to put in the slot.
--------------------------------------------------------------------------- */
export default function SolutionHero({
  solution,
  panel,
}: {
  solution: Solution;
  panel?: React.ReactNode;
}) {
  const group = GROUPS.find((g) => g.id === solution.group);
  const [primary, secondaryCta] = CTA_ACTIONS;
  const { hero } = solution;
  const media = panel ?? (solution.media?.hero ? <SolutionMedia asset={solution.media.hero} /> : null);

  return (
    <section className="pe-section pe-sol-hero" id="top">
      <div className="pe-container">
        <div className={`pe-sol-hero-grid ${media ? "has-media" : ""}`}>
          <div className="pe-sol-hero-copy">
            {group && (
              <Reveal as="span" className="pe-eyebrow">
                <SolIcon name={group.icon} /> {group.label}
              </Reveal>
            )}

            <Reveal as="h1" className="pe-h1 pe-sol-h1" delay={60}>
              {hero.title}
              {hero.titleAccent && <span className="pe-grad-text">{hero.titleAccent}</span>}
            </Reveal>

            <Reveal as="p" className="pe-hero-sub" delay={120}>
              {hero.subtitle}
            </Reveal>

            <Reveal className="pe-sol-ctas" delay={180}>
              <a className="pe-btn pe-btn-primary pe-btn-lg" href={primary.href}>
                {primary.label} <ArrowRight />
              </a>
              <a className="pe-btn pe-btn-ghost pe-btn-lg" href={secondaryCta.href}>
                {secondaryCta.label}
              </a>
              {hero.secondary && (
                <a className="pe-sol-hero-anchor" href={hero.secondary.href}>
                  {hero.secondary.label} <ArrowRight />
                </a>
              )}
            </Reveal>

            <Reveal as="ul" className="pe-sol-chips" delay={240}>
              {hero.chips.map((c) => (
                <li key={c} className="pe-pill">{c}</li>
              ))}
            </Reveal>
          </div>

          {media && (
            <Reveal className="pe-sol-hero-media" delay={140}>
              {media}
            </Reveal>
          )}
        </div>
      </div>
    </section>
  );
}
