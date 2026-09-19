/* ---------------------------------------------------------------------------
   VisionStill — the hero media slot on /solutions/computer-vision.

   One static product shot, nothing else. It replaced the drawn detection
   console, and it is deliberately inert: no animation, no video, no canvas,
   no motion of any kind. The framing around it is unchanged — the same
   halo, the same 1.5px gutter, radius and shadow the console sat in — so the
   hero's layout and responsive behaviour are exactly as they were.

   The PNG carries its own aspect ratio, so the slot imposes none: the image
   scales with the column and is never letterboxed or stretched.
--------------------------------------------------------------------------- */
export default function VisionStill() {
  return (
    <div className="pe-vc">
      <span className="pe-vc-halo" aria-hidden="true" />

      <div className="pe-vc-frame">
        <img
          className="pe-vc-img"
          src="/089530c6-914c-47b3-a86c-088068e2ad2a.png"
          alt="A computer vision dashboard: a warehouse camera view with bounding boxes over five people and two stacked-case objects, each labelled with its class and confidence, beside a detection summary, a confidence overview and an activity log."
          decoding="async"
        />
      </div>
    </div>
  );
}
