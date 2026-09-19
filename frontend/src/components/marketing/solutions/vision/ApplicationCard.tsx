"use client";

import { accent } from "../icons";
import type { Application } from "./marquee.data";

/* ---------------------------------------------------------------------------
   ApplicationCard — one thing people build, as it travels past.

   Taller than a capability card and carrying a sentence, because an
   application needs the sentence to mean anything. Same belt, slower row.
--------------------------------------------------------------------------- */
export default function ApplicationCard({ item }: { item: Application }) {
  const Icon = item.icon;

  return (
    <article className="pe-mq-card pe-app" style={accent(item.accent)}>
      <span className="pe-mq-glyph pe-app-glyph" aria-hidden="true"><Icon /></span>
      <h3 className="pe-app-title">{item.title}</h3>
      <p className="pe-app-desc">{item.desc}</p>
    </article>
  );
}
