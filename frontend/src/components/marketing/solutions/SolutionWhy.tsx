"use client";

import { Check } from "lucide-react";
import { Reveal } from "../primitives";
import { SolIcon, accent } from "./icons";
import { WHY_PETAL_EDGE } from "@/lib/solutions.config";

/* ---------------------------------------------------------------------------
   Section 7 — Why Petal Edge. Fixed sitewide: identical on all nine pages.
--------------------------------------------------------------------------- */
export default function SolutionWhy() {
  const w = WHY_PETAL_EDGE;

  return (
    <section className="pe-section" id="why">
      <div className="pe-container">
        <div className="pe-section-head">
          <Reveal as="span" className="pe-eyebrow"><Check /> {w.eyebrow}</Reveal>
          <Reveal as="h2" className="pe-h2" delay={60}>
            {w.title}<span className="pe-grad-text">{w.titleAccent}</span>
          </Reveal>
          <Reveal as="p" className="pe-lede" delay={120}>{w.lede}</Reveal>
        </div>

        <div className="pe-sol-why-grid">
          {w.points.map((p, i) => (
            <Reveal
              key={p.title}
              className="pe-card pe-card-hover pe-sol-why-card"
              style={accent(p.accent)}
              delay={(i % 4) * 70}
            >
              <span className="pe-ico pe-ico-grad"><SolIcon name={p.icon} /></span>
              <h3>{p.title}</h3>
              <p>{p.desc}</p>
            </Reveal>
          ))}
        </div>

        <Reveal className="pe-card pe-sol-included" delay={100}>
          <h3>What&rsquo;s included</h3>
          <ul>
            {w.included.map((item) => (
              <li key={item}><Check strokeWidth={3} /> {item}</li>
            ))}
          </ul>
        </Reveal>
      </div>
    </section>
  );
}
