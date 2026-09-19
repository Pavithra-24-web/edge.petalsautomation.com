"use client";

import { ArrowRight, Cpu } from "lucide-react";
import { Reveal } from "../primitives";
import { SolIcon, accent } from "./icons";
import SolutionMedia from "./SolutionMedia";
import { HARDWARE_TARGETS, type Solution } from "@/lib/solutions.config";

/* ---------------------------------------------------------------------------
   Section 6 — Deployment Options.

   Two modes, because these pages answer different questions:

   "focused" (Applications, Industries) shows only the targets that make sense
   for that solution and then points at Integrations. These pages are about what
   you build and where you apply it — they are not deployment documentation, and
   repeating the same six cards on all of them taught the reader nothing.

   "full" (Integrations) shows every supported target with its intended usage.
   This is the one place the complete picture lives.

   ESP32 is excluded platform-wide by decision.
--------------------------------------------------------------------------- */
export default function SolutionHardware({ solution }: { solution: Solution }) {
  const isFull = solution.hardwareMode === "full";

  return (
    <section className="pe-section" id="deployment-options">
      <div className="pe-container">
        <div className="pe-section-head">
          <Reveal as="span" className="pe-eyebrow"><Cpu /> Deployment options</Reveal>
          <Reveal as="h2" className="pe-h2" delay={60}>
            {isFull ? (
              <>Every target you can <span className="pe-grad-text">build for</span></>
            ) : (
              <>Where this <span className="pe-grad-text">runs</span></>
            )}
          </Reveal>
          <Reveal as="p" className="pe-lede" delay={120}>
            {isFull
              ? "Pick the one that matches the hardware and the model you trained."
              : "The targets teams reach for on this kind of project."}
          </Reveal>
        </div>

        <div className="pe-sol-hw-grid">
          {solution.hardware.map((id, i) => {
            const t = HARDWARE_TARGETS[id];
            return (
              <Reveal
                key={id}
                className="pe-card pe-card-hover pe-sol-hw-card"
                style={accent(t.accent)}
                delay={(i % 3) * 70}
              >
                <span className="pe-ico pe-ico-grad"><SolIcon name={t.icon} /></span>
                <div className="pe-sol-hw-body">
                  <h3>{t.name}</h3>
                  <p className="pe-sol-hw-pkg">{t.pkg}</p>
                  <p className="pe-sol-hw-runs">{t.runs}</p>
                  {/* Intended usage — and, for the C++ and Arduino libraries, the
                      classification-only limit. Integration pages only. */}
                  {isFull && <p className="pe-sol-hw-best">{t.bestFor}</p>}
                </div>
              </Reveal>
            );
          })}
        </div>

        {!isFull && (
          <Reveal className="pe-sol-hw-more" delay={80}>
            <p>
              Petal Edge builds for six targets in total, each with its own package
              format and constraints.
            </p>
            <a className="pe-sol-related-link" href="/solutions/#integrations">
              See all deployment options <ArrowRight />
            </a>
          </Reveal>
        )}

        <SolutionMedia asset={solution.media?.hardware} className="pe-sol-media-wide" />

        {solution.related.length > 0 && (
          <Reveal className="pe-sol-related" delay={80}>
            <span className="pe-sol-related-label">Related</span>
            {solution.related.map((r) => (
              <a key={r.href} className="pe-sol-related-link" href={r.href}>
                {r.label} <ArrowRight />
              </a>
            ))}
          </Reveal>
        )}
      </div>
    </section>
  );
}
