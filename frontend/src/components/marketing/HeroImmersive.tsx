"use client";

/* ---------------------------------------------------------------------------
   HeroImmersive — the marketing hero, built as one unified product moment.

   Layout is a 40 / 60 split: marketing copy on the left, and on the right a
   large floating "Petal Edge Studio" preview that is the visual centre of the
   page. One continuous radial aura runs behind the headline and on under the
   preview so the two columns read as a single component, not two panels.

   The preview is deliberately software chrome, not media chrome: one thin
   application bar with the Studio title and a LIVE badge, then the detection
   feed filling everything below it edge-to-edge. The <video> itself is inert
   (no controls, no PiP, pointer-events: none) so it can never read as an
   embedded player — the real detections in the footage are the only overlay.

   NOTE: the bounding boxes and class labels are burned into the clip's own
   pixels — they are baked in at export time by
   backend/app/services/post_processing/video_processor.py (cv2.rectangle +
   cv2.putText), not drawn by this component. Restyling them (branded pill
   labels, corner-bracket boxes, different colours) means re-exporting the clip
   through a branded renderer; no CSS here can touch them.

   The served files are derived, never hand-placed: frontend/media-src/ holds
   the encode master and the exact ffmpeg recipe that produces the .web.mp4,
   .web.webm and poster this element points at. Re-run it after any re-export —
   dropping a raw editor output into public/ is what put a 38 MB, 13 Mbps,
   non-faststart file on the homepage and stalled the hero.

   Swapping the clip is otherwise self-contained: the paths are fixed, and the
   stage re-fits itself to the new clip's aspect ratio (see the effect below).

   Everything visual lives in marketing-immersive.css under the `.phi-`
   namespace, so the original Hero / .pe-hero / .pe-orbit rules are untouched.
--------------------------------------------------------------------------- */

import { useEffect, useRef } from "react";
import { ArrowRight, PlayCircle, Check } from "lucide-react";
import { Reveal } from "./primitives";
import HeroTitle from "./HeroTitle";
import "@/app/marketing-immersive.css";

const TRUST = ["No credit card", "Enterprise Ready", "GPU Training", "Open Source Compatible"];

/* Minimal ambient particles: x/y placement, drift duration and delay. Kept to
   eight so the field stays quiet — soft lighting, not a starfield. */
const PARTICLES = [
  { x: "12%", y: "22%", d: 13, delay: 0, s: 3 },
  { x: "27%", y: "72%", d: 17, delay: 2.4, s: 2 },
  { x: "41%", y: "16%", d: 15, delay: 1.1, s: 2 },
  { x: "56%", y: "84%", d: 19, delay: 3.6, s: 3 },
  { x: "68%", y: "12%", d: 16, delay: 0.8, s: 2 },
  { x: "79%", y: "62%", d: 21, delay: 4.2, s: 3 },
  { x: "88%", y: "31%", d: 14, delay: 2.0, s: 2 },
  { x: "95%", y: "77%", d: 18, delay: 5.0, s: 2 },
];

export default function HeroImmersive() {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  /* Match the stage to the clip's real shape, so swapping /Home_edit.mp4 for a
     differently-proportioned video never results in a silent centre-crop.

     This has to cover both orderings. A React `onLoadedMetadata` prop alone is
     not enough: the browser starts fetching as soon as the element exists, so
     with a warm cache `loadedmetadata` routinely fires before hydration
     attaches the handler and the event is simply missed — which is exactly
     what happened here on first test. So check readyState first and only fall
     back to the event when metadata genuinely hasn't arrived yet.

     Written straight to the DOM node rather than held in state: it is a style
     value nothing in render depends on, so this avoids re-rendering the hero. */
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const fit = () => {
      const { videoWidth: w, videoHeight: h } = video;
      if (!(w > 0 && h > 0)) return;
      /* Clamped, not raw. Honouring a square clip's true ratio makes the stage
         as tall as the card is wide — measured at 856px wide that produced a
         910px card against a ~510px copy column, which wrecks the hero's
         balance. Widescreen sources (anything past 4:3) still get their exact
         ratio and so are never cropped; only squarer-than-4:3 clips are held
         back, where a controlled crop is much the lesser evil. */
      stageRef.current?.style.setProperty("--phi-stage-ar", String(Math.max(w / h, 4 / 3)));
    };

    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
      fit();
      return;
    }
    video.addEventListener("loadedmetadata", fit);
    return () => video.removeEventListener("loadedmetadata", fit);
  }, []);

  /* Decode only while the hero is actually on screen. `autoPlay` gets it going,
     but a 23s clip on `loop` keeps a decode + composite running forever once the
     hero has scrolled away — and on this page it leaves the viewport within one
     scroll. Pausing offscreen hands the decoder back while the rest of the page
     animates, and resuming is instant because the frames are already buffered.

     play() is fire-and-forget: it rejects benignly when autoplay policy defers
     it or the element is torn down mid-scroll, and neither case is actionable. */
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) void video.play().catch(() => {});
        else video.pause();
      },
      { threshold: 0.1 },
    );

    io.observe(video);
    return () => io.disconnect();
  }, []);

  return (
    <section className="pe-hero phi-hero" id="top">
      {/* ---- Ambient stack: navy wash → grid → the one shared aura → particles.
           Absolutely positioned behind .pe-container (z-index 2). ---- */}
      <div className="phi-bg" aria-hidden="true">
        <span className="phi-bg-navy" />
        <span className="phi-bg-grid" />
        <span className="phi-aura" />
        <span className="phi-particles">
          {PARTICLES.map((p, i) => (
            <span
              key={i}
              className="phi-particle"
              style={
                {
                  "--px": p.x,
                  "--py": p.y,
                  "--pd": `${p.d}s`,
                  "--pdelay": `${p.delay}s`,
                  "--ps": `${p.s}px`,
                } as React.CSSProperties
              }
            />
          ))}
        </span>
      </div>

      <div className="pe-container">
        <div className="pe-hero-grid">
          {/* ---- Left (40%): marketing copy ---- */}
          <div className="phi-copy">
            <Reveal className="pe-hero-badge phi-badge">
              <span className="pe-pill phi-pill">NEW</span>
              <span>
                UNO Q&nbsp;(.pxe) export is live —&nbsp;<b>deploy in one click</b>
              </span>
            </Reveal>

            <HeroTitle />

            <Reveal as="p" className="pe-lede" delay={120}>
              Collect sensor data, build production-ready AI models, optimize them for
              embedded hardware, deploy to edge devices, and monitor them in real time
              — all from one intelligent platform.
            </Reveal>

            <Reveal className="pe-hero-cta" delay={180}>
              <a href="/login/" className="pe-btn pe-btn-lg phi-btn phi-btn-primary">
                Get Started Free <ArrowRight />
              </a>
              <a href="/book-demo" className="pe-btn pe-btn-lg phi-btn phi-btn-outline">
                <PlayCircle /> Book Live Demo
              </a>
            </Reveal>

            <Reveal as="ul" className="pe-trust" delay={240}>
              {TRUST.map((t) => (
                <li key={t}>
                  <Check strokeWidth={3} /> {t}
                </li>
              ))}
            </Reveal>
          </div>

          {/* ---- Right (60%): the Petal Edge Studio preview ----
               phi-scene is the layout slot, phi-glow the slow cyan pulse that
               seats the card, phi-float the gentle idle drift, and phi-card the
               glass frame. Hover lift lives on the card so it composes with the
               float on the wrapper instead of fighting it for `transform`. ---- */}
          <Reveal className="phi-scene-wrap" delay={200}>
            <div className="phi-scene">
              <span className="phi-glow" aria-hidden="true" />

              <div className="phi-float">
                <div className="phi-card">
                  {/* Title bar — the only chrome. Reads as the app, not a player. */}
                  <div className="phi-bar">
                    <span className="phi-bar-brand">
                      <span className="phi-bar-mark" aria-hidden="true" />
                      Petal Edge Studio
                    </span>
                    <span className="phi-live">
                      <span className="phi-live-dot" aria-hidden="true" />
                      LIVE
                    </span>
                  </div>

                  {/* Stage — the detection feed, edge to edge, zero padding. */}
                  <div className="phi-stage" ref={stageRef}>
                    <video
                      className="phi-video"
                      ref={videoRef}
                      poster="/Home_edit_1_poster.jpg"
                      autoPlay
                      loop
                      muted
                      playsInline
                      preload="metadata"
                      disablePictureInPicture
                      disableRemotePlayback
                      controlsList="nodownload noplaybackrate nofullscreen noremoteplayback"
                      aria-label="Petal Edge Studio running live object detection"
                    >
                      <source src="/Home_edit_1.web.webm" type="video/webm" />
                      <source src="/Home_edit_1.web.mp4" type="video/mp4" />
                    </video>
                  </div>
                </div>
              </div>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
