"use client";

import { ArrowRight } from "lucide-react";
import { Reveal } from "../primitives";
import { CTA_ACTIONS, type Solution } from "@/lib/solutions.config";

/* ---------------------------------------------------------------------------
   Section 8 — Closing CTA. Buttons fixed; headline per solution.
--------------------------------------------------------------------------- */
export default function SolutionCTA({ solution }: { solution: Solution }) {
  return (
    <section className="pe-section pe-sol-cta" id="get-started">
      <div className="pe-container">
        <Reveal className="pe-card pe-sol-cta-card">
          <h2 className="pe-h2">
            {solution.ctaHeadline}
          </h2>
          <p className="pe-lede">
            Create a project, bring your data, and take it through to a deployed model.
          </p>
          <div className="pe-sol-cta-actions">
            {CTA_ACTIONS.map((a) => (
              <a
                key={a.href}
                className={`pe-btn pe-btn-lg ${a.variant === "primary" ? "pe-btn-primary" : "pe-btn-ghost"}`}
                href={a.href}
              >
                {a.label}
                {a.variant === "primary" && <ArrowRight />}
              </a>
            ))}
          </div>
        </Reveal>
      </div>
    </section>
  );
}
