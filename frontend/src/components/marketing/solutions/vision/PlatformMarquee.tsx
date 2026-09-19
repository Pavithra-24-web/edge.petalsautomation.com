"use client";

import { Layers } from "lucide-react";
import { Reveal } from "../../primitives";
import MarqueeRow from "./MarqueeRow";
import CapabilityCard from "./CapabilityCard";
import {
  CAPABILITIES_DATA,
  CAPABILITIES_MODEL,
  CAPABILITIES_SHIP,
  STAGES,
  type Capability,
} from "./marquee.data";

/* ---------------------------------------------------------------------------
   What you get in the platform — three belts, one per third of the pipeline.

   The rows are not a shuffled bag of features. Row one is everything that
   happens before a model exists, row two is building it and finding out
   whether it works, row three is getting it onto hardware and keeping it
   there. Reading top to bottom is reading the workflow in order; the belts
   move in alternating directions so the three never lock into one block.
--------------------------------------------------------------------------- */
const card = (item: Capability) => <CapabilityCard item={item} />;

export default function PlatformMarquee() {
  return (
    <section className="pe-section pe-cv-platform" id="platform" aria-labelledby="platform-h">
      <div className="pe-container">
        <div className="pe-section-head pe-cv-head">
          <Reveal as="span" className="pe-eyebrow">
            <Layers /> What you get
          </Reveal>
          <Reveal as="h2" className="pe-h2" id="platform-h" delay={60}>
            What you get in <span className="pe-grad-text">the platform</span>
          </Reveal>
          <Reveal as="p" className="pe-lede" delay={120}>
            Every stage of the workflow lives in one place — collecting and labelling
            images, designing the impulse, training, testing what came out, and shipping
            it to the device that has to run it.
          </Reveal>
        </div>

        {/* The legend is what turns the colour coding into information: each
            card is tinted by the stage it belongs to. */}
        <Reveal as="ul" className="pe-mq-legend" delay={160}>
          {STAGES.map((s) => (
            <li key={s.id} style={{ ["--c" as string]: s.accent }}>
              <span className="pe-mq-legend-dot" aria-hidden="true" />
              {s.label}
            </li>
          ))}
        </Reveal>
      </div>

      {/* Full-bleed: the belts run edge to edge, so they read as continuous
          rather than as three boxes sitting inside the container.

          Durations are set per row from its measured width so all three travel
          at the same ~40px/s — rows of different lengths at the same duration
          would run at visibly different speeds. Short labels can be read at
          that pace; the applications belt below, which carries sentences,
          deliberately runs slower. */}
      <Reveal className="pe-mq" delay={120}>
        <MarqueeRow
          items={CAPABILITIES_DATA}
          render={card}
          duration={56}
          direction="left"
          label="Data and labelling capabilities"
        />
        <MarqueeRow
          items={CAPABILITIES_MODEL}
          render={card}
          duration={52}
          direction="right"
          label="Training and evaluation capabilities"
        />
        <MarqueeRow
          items={CAPABILITIES_SHIP}
          render={card}
          duration={62}
          direction="left"
          label="Deployment and operations capabilities"
        />
      </Reveal>
    </section>
  );
}
