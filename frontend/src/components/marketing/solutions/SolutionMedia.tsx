"use client";

import type { MediaAsset } from "@/lib/solutions.config";

/* ---------------------------------------------------------------------------
   SolutionMedia — renders a section's single image or video slot.
   Assets are optional: when a solution has no asset for a section, nothing is
   rendered at all, so the section degrades to text rather than a broken frame.
--------------------------------------------------------------------------- */
export default function SolutionMedia({
  asset,
  className = "",
}: {
  asset?: MediaAsset;
  className?: string;
}) {
  if (!asset) return null;

  return (
    <figure className={`pe-sol-media ${className}`}>
      {asset.kind === "video" ? (
        <video
          src={asset.src}
          poster={asset.poster}
          muted
          loop
          playsInline
          autoPlay
          preload="metadata"
          aria-label={asset.alt}
        />
      ) : (
        <img src={asset.src} alt={asset.alt} loading="lazy" decoding="async" />
      )}
      {asset.caption && <figcaption>{asset.caption}</figcaption>}
    </figure>
  );
}
