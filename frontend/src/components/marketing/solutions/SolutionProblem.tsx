"use client";

import { Check, Users } from "lucide-react";
import { Reveal } from "../primitives";
import { SolIcon, accent } from "./icons";
import { EDGE_INFERENCE_BULLETS, type Solution } from "@/lib/solutions.config";

/* ---------------------------------------------------------------------------
   Section 2 — Problem.
   Statement + audience are per-solution; the three edge-inference bullets are
   fixed sitewide (solutionspage.md §2.2).
--------------------------------------------------------------------------- */
export default function SolutionProblem({ solution }: { solution: Solution }) {
  return (
    <section className="pe-section" id="problem">
      <div className="pe-container">
        <div className="pe-sol-split">
          <div>
            <Reveal as="span" className="pe-eyebrow">
              <Users /> The problem
            </Reveal>
            {/* The section needs a real h2 or the page outline jumps from the
                hero h1 straight to these h3s. Visually hidden: the statement
                below already reads as the section's opening line. */}
            <h2 className="pe-sr-only">What this solves</h2>
            <Reveal as="p" className="pe-sol-statement" delay={60}>
              {solution.problem.statement}
            </Reveal>

            <Reveal as="h3" className="pe-sol-sub-h" delay={120}>
              Who this is for
            </Reveal>
            <ul className="pe-sol-audience">
              {solution.problem.audience.map((a, i) => (
                <Reveal as="li" key={a} delay={140 + i * 60}>
                  <Check strokeWidth={3} /> {a}
                </Reveal>
              ))}
            </ul>
          </div>

          <div className="pe-sol-edge">
            <Reveal as="h3" className="pe-sol-sub-h" delay={80}>
              Why edge inference fits
            </Reveal>
            {EDGE_INFERENCE_BULLETS.map((b, i) => (
              <Reveal
                key={b.title}
                className="pe-card pe-sol-edge-card"
                style={accent(b.accent)}
                delay={120 + i * 70}
              >
                <span className="pe-ico pe-ico-grad"><SolIcon name={b.icon} /></span>
                <div>
                  <h4>{b.title}</h4>
                  <p>{b.desc}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
