"use client";

import { Boxes } from "lucide-react";
import { Reveal } from "../../primitives";
import MarqueeRow from "./MarqueeRow";
import ApplicationCard from "./ApplicationCard";
import { APPLICATIONS_A, APPLICATIONS_B, type Application } from "./marquee.data";

/* ---------------------------------------------------------------------------
   What teams build with it — two belts, opposite directions, slower than the
   capability rows above because each card carries a sentence to read.

   Every line here is a model you train on your own images. Petal Edge ships no
   pre-trained detector for any of them, and the copy is written so nothing on
   this belt reads as one.
--------------------------------------------------------------------------- */
const card = (item: Application) => <ApplicationCard item={item} />;

export default function ApplicationsMarquee() {
  return (
    <section className="pe-section" id="use-cases" aria-labelledby="use-cases-h">
      <div className="pe-container">
        <div className="pe-section-head pe-cv-head">
          <Reveal as="span" className="pe-eyebrow"><Boxes /> Use cases</Reveal>
          <Reveal as="h2" className="pe-h2" id="use-cases-h" delay={60}>
            What teams <span className="pe-grad-text">build with it</span>
          </Reveal>
          <Reveal as="p" className="pe-lede" delay={120}>
            All of these start the same way: your images, your classes, your device.
            Nothing here is a stock model — it is what the same workflow produces when
            you point it at a different problem.
          </Reveal>
        </div>
      </div>

      {/* ~30px/s against the capability belt's ~40 — a card here has a sentence
          to read, and the two sections should not feel like one long belt. */}
      <Reveal className="pe-mq pe-mq-apps" delay={120}>
        <MarqueeRow
          items={APPLICATIONS_A}
          render={card}
          duration={139}
          direction="left"
          label="Applications, first row"
        />
        <MarqueeRow
          items={APPLICATIONS_B}
          render={card}
          duration={128}
          direction="right"
          label="Applications, second row"
        />
      </Reveal>
    </section>
  );
}
