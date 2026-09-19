"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { CalendarCheck, Loader2, CheckCircle2 } from "lucide-react";
import "../marketing.css";
import { useHomeTheme, Reveal } from "@/components/marketing/primitives";
import HomeBackground from "@/components/marketing/HomeBackground";
import Navbar from "@/components/marketing/Navbar";
import { Footer } from "@/components/marketing/Closing";
import { demoApi } from "@/utils/api";

const SOURCES = [
  "LinkedIn",
  "X / Twitter",
  "Instagram",
  "YouTube",
  "Facebook",
  "Reddit",
  "Search engine (Google)",
  "Referral / word of mouth",
  "Other",
];

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
// Basic international-friendly phone check: optional leading +, then 7–15 digits
// once separators (spaces, hyphens, parentheses) are stripped.
function isValidMobile(raw: string): boolean {
  const trimmed = raw.trim();
  if (!/^\+?[\d\s()-]+$/.test(trimmed)) return false;
  const digits = trimmed.replace(/\D/g, "");
  return digits.length >= 7 && digits.length <= 15;
}

type Status = "idle" | "loading" | "success" | "error";

export default function BookDemoPage() {
  const [theme, toggleTheme] = useHomeTheme();
  const glowRef = useRef<HTMLDivElement | null>(null);

  const [name, setName] = useState("");
  const [mobile, setMobile] = useState("");
  const [email, setEmail] = useState("");
  const [company, setCompany] = useState("");
  const [description, setDescription] = useState("");
  const [source, setSource] = useState("");
  const [status, setStatus] = useState<Status>("idle");

  // Cursor glow — follows the pointer on devices that have one. (Matches home.)
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

  const valid = useMemo(
    () =>
      name.trim().length > 0 &&
      isValidMobile(mobile) &&
      EMAIL_RE.test(email.trim()) &&
      description.trim().length > 0 &&
      source.trim().length > 0,
    [name, mobile, email, description, source]
  );

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!valid || status === "loading") return;
    setStatus("loading");
    try {
      await demoApi.create({
        name: name.trim(),
        mobile: mobile.trim(),
        email: email.trim(),
        description: description.trim(),
        source,
        company: company.trim() || undefined,
      });
      setStatus("success");
    } catch {
      // Keep entered values so the user can retry without re-typing.
      setStatus("error");
    }
  }

  return (
    <div className="pe-home" data-theme={theme}>
      <HomeBackground />
      <div ref={glowRef} className="pe-cursor-glow" aria-hidden="true" />

      <Navbar theme={theme} onToggleTheme={toggleTheme} />

      <main>
        <section className="pe-section">
          <div className="pe-container">
            <div className="pe-section-head">
              <Reveal as="span" className="pe-eyebrow" delay={0}>
                <CalendarCheck /> Book a Demo
              </Reveal>
              <Reveal as="h2" className="pe-h2" delay={60}>
                See Petal Edge <span className="pe-grad-text">in action</span>
              </Reveal>
              <Reveal as="p" className="pe-lede" delay={120}>
                Tell us a bit about you and we&apos;ll reach out to schedule a
                personalized walkthrough.
              </Reveal>
            </div>

            <Reveal className="pe-card pe-demo-card" delay={180}>
              {status === "success" ? (
                <div className="pe-demo-success">
                  <span className="pe-demo-success-ico">
                    <CheckCircle2 />
                  </span>
                  <h3>Thanks! We&apos;ll be in touch shortly.</h3>
                  <p>
                    Your request has been received. Our team will reach out to
                    the email you provided to arrange your walkthrough.
                  </p>
                </div>
              ) : (
                <form className="pe-demo-form" onSubmit={submit} noValidate>
                  <div className="pe-demo-grid">
                    <div className="pe-demo-field">
                      <label htmlFor="demo-name" className="pe-demo-label">
                        Name <span className="pe-demo-req">*</span>
                      </label>
                      <input
                        id="demo-name"
                        type="text"
                        className="pe-demo-input"
                        placeholder="Your Name"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        autoComplete="name"
                        required
                      />
                    </div>

                    <div className="pe-demo-field">
                      <label htmlFor="demo-mobile" className="pe-demo-label">
                        Mobile number <span className="pe-demo-req">*</span>
                      </label>
                      <input
                        id="demo-mobile"
                        type="tel"
                        className="pe-demo-input"
                        placeholder="+91 12345 67890"
                        value={mobile}
                        onChange={(e) => setMobile(e.target.value)}
                        autoComplete="tel"
                        required
                      />
                    </div>

                    <div className="pe-demo-field">
                      <label htmlFor="demo-email" className="pe-demo-label">
                        Email <span className="pe-demo-req">*</span>
                      </label>
                      <input
                        id="demo-email"
                        type="email"
                        className="pe-demo-input"
                        placeholder="you@email.com"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        autoComplete="email"
                        required
                      />
                    </div>

                    <div className="pe-demo-field">
                      <label htmlFor="demo-company" className="pe-demo-label">
                        Company / Institution{" "}
                        <span className="pe-demo-optional">(optional)</span>
                      </label>
                      <input
                        id="demo-company"
                        type="text"
                        className="pe-demo-input"
                        placeholder="Petal Automations"
                        value={company}
                        onChange={(e) => setCompany(e.target.value)}
                        autoComplete="organization"
                      />
                    </div>

                    <div className="pe-demo-field pe-demo-field-full">
                      <label htmlFor="demo-source" className="pe-demo-label">
                        How did you hear about us?{" "}
                        <span className="pe-demo-req">*</span>
                      </label>
                      <select
                        id="demo-source"
                        className="pe-demo-input pe-demo-select"
                        value={source}
                        onChange={(e) => setSource(e.target.value)}
                        required
                      >
                        <option value="" disabled>
                          Select an option…
                        </option>
                        {SOURCES.map((s) => (
                          <option key={s} value={s}>
                            {s}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="pe-demo-field pe-demo-field-full">
                      <label htmlFor="demo-description" className="pe-demo-label">
                        What would you like to see / your use case{" "}
                        <span className="pe-demo-req">*</span>
                      </label>
                      <textarea
                        id="demo-description"
                        className="pe-demo-input pe-demo-textarea"
                        placeholder="Tell us about your project, the problem you're solving, and what you'd like to explore in the demo."
                        rows={5}
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        required
                      />
                    </div>
                  </div>

                  {status === "error" && (
                    <p className="pe-demo-error" role="alert">
                      Something went wrong sending your request. Please try again.
                    </p>
                  )}

                  <button
                    type="submit"
                    className="pe-btn pe-btn-primary pe-btn-lg pe-demo-submit"
                    disabled={!valid || status === "loading"}
                  >
                    {status === "loading" ? (
                      <>
                        <Loader2 className="pe-demo-spin" /> Sending…
                      </>
                    ) : (
                      "Request Demo"
                    )}
                  </button>
                </form>
              )}
            </Reveal>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
