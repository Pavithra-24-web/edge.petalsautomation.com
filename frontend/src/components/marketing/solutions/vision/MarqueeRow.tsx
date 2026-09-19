"use client";

import InfiniteTicker from "./InfiniteTicker";

/* ---------------------------------------------------------------------------
   MarqueeRow — one belt of one kind of card.

   Splitting this from InfiniteTicker keeps the engine free of content: the row
   decides what a card looks like and how fast it travels, the ticker decides
   nothing but the loop. Rows on a page carry different durations and opposite
   directions on purpose — two belts running in step read as one wide table
   scrolling, which is the thing this is not.
--------------------------------------------------------------------------- */
export default function MarqueeRow<T extends { title: string }>({
  items,
  render,
  duration,
  direction = "left",
  label,
}: {
  items: T[];
  render: (item: T) => React.ReactNode;
  duration: number;
  direction?: "left" | "right";
  label: string;
}) {
  return (
    <InfiniteTicker duration={duration} direction={direction} label={label}>
      {items.map((item) => (
        <div className="pe-mq-item" key={item.title}>
          {render(item)}
        </div>
      ))}
    </InfiniteTicker>
  );
}
