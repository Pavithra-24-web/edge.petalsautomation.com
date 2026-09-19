"use client";

import { ArrowRight, Sparkles } from "lucide-react";
import { Reveal } from "../../primitives";
import { SolIcon, accent } from "../icons";
import type { IconName } from "@/lib/solutions.config";

/* ---------------------------------------------------------------------------
   AI-assisted labeling — the section that replaced the shared problem block on
   /solutions/computer-vision.

   Copy left, product shot right: the screen being described is the AI Labeling
   page itself, so the still is the argument and the copy is the caption.

   The four points are a loop, not a bag of features — suggest, annotate,
   review, ship — so their accents walk the brand gradient in that order
   (indigo → violet → cyan → green), the same stops the workflow rail uses.
   Nothing here animates beyond the shared reveal and the button's hover.
--------------------------------------------------------------------------- */
const POINTS: { icon: IconName; label: string; accent: string }[] = [
  { icon: "Wand2", label: "Automatic object suggestions", accent: "#6366f1" },
  { icon: "Gauge", label: "Faster annotation workflow", accent: "#8b5cf6" },
  { icon: "MousePointerClick", label: "Human review & correction", accent: "#06b6d4" },
  { icon: "Database", label: "High-quality training datasets", accent: "#22c55e" },
];

export default function VisionLabeling() {
  return (
    <section className="pe-section pe-cv-label" id="ai-labeling" aria-labelledby="ai-labeling-h">
      <div className="pe-container">
        <div className="pe-cvl-split">
          <div className="pe-cvl-copy">
            <Reveal as="span" className="pe-eyebrow">
              <Sparkles /> Labeling
            </Reveal>

            <Reveal as="h2" className="pe-h2" id="ai-labeling-h" delay={60}>
              AI-assisted <span className="pe-grad-text">Labeling</span>
            </Reveal>

            <Reveal as="p" className="pe-lede" delay={120}>
              Accelerate dataset creation with AI-assisted image labeling. Automatically
              detect and suggest bounding boxes for objects, significantly reducing manual
              annotation time while improving consistency across large datasets. Review,
              adjust, and approve predictions in seconds to build high-quality training
              data faster.
            </Reveal>

            <ul className="pe-cvl-points">
              {POINTS.map((p, i) => (
                <Reveal
                  as="li"
                  key={p.label}
                  className="pe-cvl-point"
                  style={accent(p.accent)}
                  delay={180 + i * 70}
                >
                  <span className="pe-ico pe-ico-grad" aria-hidden="true">
                    <SolIcon name={p.icon} />
                  </span>
                  {p.label}
                </Reveal>
              ))}
            </ul>

            <Reveal className="pe-cvl-actions" delay={460}>
              <a className="pe-btn pe-btn-primary pe-btn-lg" href="/login/">
                Start Labeling <ArrowRight />
              </a>
            </Reveal>
          </div>

          <Reveal className="pe-cvl-media" delay={140}>
            <div className="pe-cvl-frame">
              <img
                className="pe-cvl-img"
                src="/00cf9046-29fb-4c5e-b48d-6ab393eee454.png"
                /* Intrinsic size, so the slot holds its height while the shot
                   is still loading — it sits below the fold and is lazy. */
                width={1634}
                height={963}
                alt="The AI Labeling screen: a plain-text action reading “Detect bike and car”, and a preview grid of four images with suggested bounding boxes labelled bike or car, each with its confidence and accept or reject controls."
                loading="lazy"
                decoding="async"
              />
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
