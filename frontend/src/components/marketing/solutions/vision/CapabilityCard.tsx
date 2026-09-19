"use client";

import { accent } from "../icons";
import { stageOf, type Capability } from "./marquee.data";

/* ---------------------------------------------------------------------------
   CapabilityCard — one thing the platform holds, as it travels past.

   Shape follows the reading speed. A moving card is read in about a second, so
   this one is a horizontal bar: glyph, name, and the pipeline stage it belongs
   to. The stage is not decoration — it is why the card is that colour, and the
   legend above the belt is what makes the colour legible.
--------------------------------------------------------------------------- */
export default function CapabilityCard({ item }: { item: Capability }) {
  const stage = stageOf(item.stage);
  const Icon = item.icon;

  return (
    <div className="pe-mq-card pe-cap" style={accent(stage.accent)}>
      <span className="pe-mq-glyph pe-cap-glyph" aria-hidden="true"><Icon /></span>
      <span className="pe-cap-text">
        <span className="pe-cap-title">{item.title}</span>
        <span className="pe-cap-stage">
          <span className="pe-cap-dot" aria-hidden="true" />
          {stage.label}
        </span>
      </span>
    </div>
  );
}
