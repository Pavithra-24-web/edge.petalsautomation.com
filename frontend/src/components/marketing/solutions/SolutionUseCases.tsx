"use client";

import { Boxes, Check } from "lucide-react";
import { Reveal } from "../primitives";
import SolutionMedia from "./SolutionMedia";
import { HARDWARE_TARGETS, type HardwareTargetId, type Solution } from "@/lib/solutions.config";

/** Use-case targets are either a hardware id or a short free-text outcome. */
function targetLabel(target: string): string {
  const hw = HARDWARE_TARGETS[target as HardwareTargetId];
  return hw ? hw.name : target;
}

/* ---------------------------------------------------------------------------
   Section 5 — Use cases. Four to six concrete examples per solution.
--------------------------------------------------------------------------- */
export default function SolutionUseCases({ solution }: { solution: Solution }) {
  return (
    <section className="pe-section" id="use-cases">
      <div className="pe-container">
        <div className="pe-section-head">
          <Reveal as="span" className="pe-eyebrow"><Boxes /> Use cases</Reveal>
          <Reveal as="h2" className="pe-h2" delay={60}>
            What teams <span className="pe-grad-text">build with it</span>
          </Reveal>
        </div>

        <div className="pe-sol-uc-grid">
          {solution.useCases.map((u, i) => (
            <Reveal
              key={u.title}
              className="pe-card pe-card-hover pe-sol-uc-card"
              delay={(i % 3) * 70}
            >
              <h3>{u.title}</h3>
              <p>{u.desc}</p>
              <div className="pe-badges">
                <span className="pe-badge ok"><Check strokeWidth={3} /> {u.model}</span>
                <span className="pe-badge">{targetLabel(u.target)}</span>
              </div>
            </Reveal>
          ))}
        </div>

        <SolutionMedia asset={solution.media?.useCases} className="pe-sol-media-wide" />
      </div>
    </section>
  );
}
