/* ---------------------------------------------------------------------------
   Stage accents.

   A pipeline stage's colour is not a per-card decoration — it encodes how far
   through the pipeline that stage sits. The accent walks the brand gradient
   (the same four stops as --pe-gradient) from the first stage to the last, so
   the rail fill, the slide glow and the node dots all agree about position.

   Works for any stage count: the vision variant has ten, the sensor variant
   eight, the deployment path six.
--------------------------------------------------------------------------- */

/** Brand gradient stops, in pipeline order. Mirrors --pe-gradient. */
const STOPS: [number, number, number][] = [
  [0x63, 0x66, 0xf1], // indigo  — intake
  [0x8b, 0x5c, 0xf6], // violet  — modelling
  [0x06, 0xb6, 0xd4], // cyan    — verification
  [0x22, 0xc5, 0x5e], // green   — shipped
];

const lerp = (a: number, b: number, t: number) => Math.round(a + (b - a) * t);

/**
 * Accent for stage `i` of `total`, as an `rgb()` string.
 * Single-stage pipelines get the first stop rather than dividing by zero.
 */
export function stageAccent(i: number, total: number): string {
  const t = total > 1 ? Math.min(Math.max(i / (total - 1), 0), 1) : 0;
  const span = t * (STOPS.length - 1);
  const from = Math.min(Math.floor(span), STOPS.length - 2);
  const f = span - from;
  const a = STOPS[from];
  const b = STOPS[from + 1];
  return `rgb(${lerp(a[0], b[0], f)} ${lerp(a[1], b[1], f)} ${lerp(a[2], b[2], f)})`;
}
