"use client";

/* ---------------------------------------------------------------------------
   HomeBackground — the marketing homepage's ambient backdrop.

   A single fixed, non-interactive layer composed of several purposeful sub-
   layers, ordered back-to-front. Every layer is driven by the .pe-* design
   tokens (brand palette only) and is heavily masked so it reads as a quiet,
   handcrafted enterprise surface rather than a busy gradient wallpaper.

     1. aurora   — five huge corner/center ambient lights (<12% opacity)
     2. mesh     — very-low-opacity 4-tint brand mesh (indigo/cyan/purple/blue)
     3. grid     — two-tier engineering blueprint grid (24px minor + 120px major)
     4. lines    — a few thin geometric guide lines
     5. rings    — decorative blueprint circles + a faint dot-field
     6. orbs     — soft, blurred depth circles that drift slowly
     7. vignette — edge falloff for depth / focus toward the content column
     8. noise    — ~2–3% film grain so flat areas never band

   Kept as one component so the whole system stays consistent and reusable;
   it renders once at the page root behind all content (see page.tsx).
--------------------------------------------------------------------------- */
export default function HomeBackground() {
  return (
    <div className="pe-bg" aria-hidden="true">
      <div className="pe-bg-aurora" />
      <div className="pe-bg-mesh" />
      <div className="pe-bg-grid" />
      <div className="pe-bg-lines" />
      <div className="pe-bg-rings" />
      <div className="pe-bg-dots" />
      <div className="pe-bg-orbs">
        <span className="pe-orb o1" />
        <span className="pe-orb o2" />
        <span className="pe-orb o3" />
        <span className="pe-orb o4" />
        <span className="pe-orb o5" />
        <span className="pe-orb o6" />
        <span className="pe-orb o7" />
      </div>
      <div className="pe-bg-vignette" />
      <div className="pe-bg-noise" />
    </div>
  );
}
