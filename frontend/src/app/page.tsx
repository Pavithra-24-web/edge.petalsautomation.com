"use client";

import { useEffect, useRef } from "react";
import "./marketing.css";
import { useHomeTheme } from "@/components/marketing/primitives";
import HomeBackground from "@/components/marketing/HomeBackground";
import Navbar from "@/components/marketing/Navbar";
// import Hero from "@/components/marketing/Hero"; // original hero — uncomment + render <Hero/> below to switch back
import HeroImmersive from "@/components/marketing/HeroImmersive";
import { Workflow } from "@/components/marketing/Sections";
import { DashboardShowcase } from "@/components/marketing/Showcase";
import { Footer } from "@/components/marketing/Closing";

export default function HomePage() {
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

  return (
    <div className="pe-home" data-theme={theme}>
      <HomeBackground />
      <div ref={glowRef} className="pe-cursor-glow" aria-hidden="true" />

      <Navbar theme={theme} onToggleTheme={toggleTheme} />

      <main>
        {/* Swap back to the original hero by rendering <Hero /> here instead. */}
        <HeroImmersive />
        <Workflow />
        <DashboardShowcase />
      </main>

      <Footer />
    </div>
  );
}
