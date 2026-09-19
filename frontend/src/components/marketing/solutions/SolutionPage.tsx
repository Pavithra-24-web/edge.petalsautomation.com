"use client";

import { useEffect, useRef } from "react";
import "../../../app/marketing.css";
import { useHomeTheme } from "../primitives";
import HomeBackground from "../HomeBackground";
import Navbar from "../Navbar";
import { Footer } from "../Closing";
import SolutionHero from "./SolutionHero";
import SolutionProblem from "./SolutionProblem";
import SolutionWorkflowCarousel from "./workflow/SolutionWorkflowCarousel";
import VisionStill from "./vision/VisionStill";
import VisionLabeling from "./vision/VisionLabeling";
import PlatformMarquee from "./vision/PlatformMarquee";
import ApplicationsMarquee from "./vision/ApplicationsMarquee";
import SolutionFeatures from "./SolutionFeatures";
import SolutionUseCases from "./SolutionUseCases";
import SolutionHardware from "./SolutionHardware";
import SolutionWhy from "./SolutionWhy";
import SolutionCTA from "./SolutionCTA";
import { getSolution } from "@/lib/solutions.config";

/* ---------------------------------------------------------------------------
   SolutionPage — the one template every solution page renders through.
   Shell matches /features and /pricing exactly: .pe-home root, ambient
   background, cursor glow, shared navbar, .pe-boost main, shared footer.
--------------------------------------------------------------------------- */
export default function SolutionPage({ slug }: { slug: string }) {
  const [theme, toggleTheme] = useHomeTheme();
  const glowRef = useRef<HTMLDivElement | null>(null);

  // Cursor glow — follows the pointer on devices that have one.
  useEffect(() => {
    if (window.matchMedia?.("(pointer: coarse)").matches) return;
    let raf = 0;
    const onMove = (e: PointerEvent) => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const el = glowRef.current;
        if (el) el.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
      });
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => {
      window.removeEventListener("pointermove", onMove);
      cancelAnimationFrame(raf);
    };
  }, []);

  const solution = getSolution(slug);
  if (!solution) return null;

  /* Computer Vision is the one page with its own product moment: a static
     detection screenshot in the hero, then two marquees — the platform's
     capabilities, and what people build with them. Everything else is the
     common template, the walkthrough carousel included: the capability belt
     says what the platform covers, the carousel walks the pipeline stage by
     stage, and the two do not restate each other.

     The hero shot is the page's only media file, and it is a still: no clip,
     no autoplay, no canvas. The belts are text and icons, and nothing here is
     borrowed from the homepage. */
  const isVision = solution.slug === "computer-vision";

  return (
    <div className="pe-home" data-theme={theme}>
      <HomeBackground />
      <div ref={glowRef} className="pe-cursor-glow" aria-hidden="true" />

      <Navbar theme={theme} onToggleTheme={toggleTheme} />

      <main className={isVision ? "pe-boost pe-cvp" : "pe-boost"}>
        <SolutionHero solution={solution} panel={isVision ? <VisionStill /> : undefined} />
        {isVision && <PlatformMarquee />}
        {/* Vision trades the shared problem block for the labeling section:
            the page has already said what it is for by this point, and what an
            image team actually wants to see next is the annotation screen. */}
        {isVision ? <VisionLabeling /> : <SolutionProblem solution={solution} />}
        <SolutionWorkflowCarousel solution={solution} />
        {/* The capability belt above already carries the shared "What you get in
            the platform" heading, so the feature grid takes a heading of its
            own here — these six cards are the image-specific detail under it. */}
        <SolutionFeatures
          solution={solution}
          heading={
            isVision ? (
              <>Built for image data, <span className="pe-grad-text">specifically</span></>
            ) : undefined
          }
        />
        {/* Same section, same heading, two treatments: the vision page runs the
            applications belt, the other eight keep the use-case grid. */}
        {isVision ? <ApplicationsMarquee /> : <SolutionUseCases solution={solution} />}
        <SolutionHardware solution={solution} />
        <SolutionWhy />
        <SolutionCTA solution={solution} />
      </main>

      <Footer />
    </div>
  );
}
