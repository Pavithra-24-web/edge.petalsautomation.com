"use client";

import { Cpu, Check, TrendingUp, ShieldCheck, Zap, Headphones } from "lucide-react";
import { Reveal } from "./primitives";
import { PLANS } from "./Closing";

/** Inline CSS custom property for the per-icon accent (matches Showcase.tsx). */
const accent = (c: string) => ({ ["--c"]: c }) as React.CSSProperties;

/* Bottom feature bar — reassurance strip beneath the plan grid. */
const HIGHLIGHTS = [
  { icon: TrendingUp, c: "#8b5cf6", title: "Start Free", desc: "Explore edge AI with no commitment." },
  { icon: ShieldCheck, c: "#3b82f6", title: "No Credit Card", desc: "Get started instantly without any payment." },
  { icon: Zap, c: "#22c55e", title: "Scale Anytime", desc: "Upgrade or downgrade as your needs evolve." },
  { icon: Headphones, c: "#8b5cf6", title: "Expert Support", desc: "Get help from our team whenever you need it." },
];

export function PricingSection() {
  return (
    <section className="pe-section" id="pricing">
      <div className="pe-container">
        <div className="pe-section-head">
          <Reveal as="span" className="pe-eyebrow"><Cpu /> Pricing</Reveal>
          <Reveal as="h2" className="pe-h2" delay={60}>
            Simple, <span className="pe-grad-text">scalable</span> pricing
          </Reveal>
          <Reveal as="p" className="pe-lede" delay={120}>
            Start free and grow into production. No credit card required to begin.
          </Reveal>
        </div>

        <div className="pe-price-grid">
          {PLANS.map((p, i) => (
            <Reveal key={p.name} className={`pe-card pe-price-card ${p.highlight ? "pe-hi" : ""}`} delay={i * 80}>
              {p.highlight && <span className="pe-price-tag">Most Popular</span>}
              <div className="pe-price-name pe-grad-text">{p.name}</div>
              <p className="pe-price-desc">{p.desc}</p>
              <div className="pe-price-amt">
                <span className="n">{p.price}</span>
                <span className="p">{p.period}</span>
              </div>
              <ul className="pe-price-feats">
                {p.feats.map((f) => (
                  <li key={f}><Check strokeWidth={3} /> {f}</li>
                ))}
              </ul>
              <a href="/contact/" className={`pe-btn ${p.ctaClass}`}>{p.cta}</a>
            </Reveal>
          ))}
        </div>

        <div className="pe-card pe-price-bar">
          {HIGHLIGHTS.map((h, i) => {
            const Icon = h.icon;
            return (
              <Reveal key={h.title} className="pe-price-bar-item" delay={i * 80} style={accent(h.c)}>
                <span className="pe-ico"><Icon /></span>
                <div>
                  <h4>{h.title}</h4>
                  <p>{h.desc}</p>
                </div>
              </Reveal>
            );
          })}
        </div>
      </div>
    </section>
  );
}
