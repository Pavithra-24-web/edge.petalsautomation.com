"use client";

import { Sparkles } from "lucide-react";
import { Reveal } from "../primitives";
import { SolIcon, accent } from "./icons";
import SolutionMedia from "./SolutionMedia";
import { FEATURE_POOL, type Solution } from "@/lib/solutions.config";

/* ---------------------------------------------------------------------------
   Section 4 — Key features.
   Six ids picked per solution from the shared pool; card copy is authored once
   in solutions.config.ts and reused verbatim across pages.

   No evidence chip: the API paths and builder names that used to render here
   are developer-facing and belong nowhere near a customer page. They survive as
   `// traces to:` comments on each FEATURE_POOL entry, so a future edit is still
   checkable against the code.
--------------------------------------------------------------------------- */
export default function SolutionFeatures({
  solution,
  heading,
}: {
  solution: Solution;
  /* Override for pages that already say "what you get" higher up — without one
     the outline would carry the same h2 twice. See SolutionPage. */
  heading?: React.ReactNode;
}) {
  return (
    <section className="pe-section" id="features">
      <div className="pe-container">
        <div className="pe-section-head">
          <Reveal as="span" className="pe-eyebrow"><Sparkles /> Key features</Reveal>
          <Reveal as="h2" className="pe-h2" delay={60}>
            {heading ?? (
              <>What you get in the <span className="pe-grad-text">platform</span></>
            )}
          </Reveal>
        </div>

        <div className="pe-sol-feat-grid">
          {solution.features.map((id, i) => {
            const f = FEATURE_POOL[id];
            return (
              <Reveal
                key={id}
                className="pe-card pe-card-hover pe-sol-feat-card"
                style={accent(f.accent)}
                delay={(i % 3) * 70}
              >
                <span className="pe-ico pe-ico-grad"><SolIcon name={f.icon} /></span>
                <h3>{f.title}</h3>
                <p>{f.desc}</p>
              </Reveal>
            );
          })}
        </div>

        <SolutionMedia asset={solution.media?.features} className="pe-sol-media-wide" />
      </div>
    </section>
  );
}
