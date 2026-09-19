"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Headset,
  Wrench,
  PlaySquare,
  Lightbulb,
  Loader2,
  CheckCircle2,
  ArrowLeft,
} from "lucide-react";
import "../marketing.css";
import { useHomeTheme, Reveal } from "@/components/marketing/primitives";
import HomeBackground from "@/components/marketing/HomeBackground";
import Navbar from "@/components/marketing/Navbar";
import { Footer } from "@/components/marketing/Closing";
import { contactApi, type ContactCategory } from "@/utils/api";

const accent = (c: string) => ({ ["--c"]: c }) as React.CSSProperties;

type Category = {
  key: ContactCategory;
  title: string;
  blurb: string;
  icon: typeof Headset;
  c: string;
  /** When set, the card navigates here instead of revealing the inline form. */
  href?: string;
};

const CATEGORIES: Category[] = [
  {
    key: "Sales inquiries",
    title: "Sales inquiries",
    blurb: "Let us know how we can help!",
    icon: Headset,
    c: "#6366f1",
  },
  {
    key: "Technical support",
    title: "Technical support",
    blurb: "Are you an existing user having trouble with Petal Edge studio?",
    icon: Wrench,
    c: "#8b5cf6",
  },
  {
    key: "Product demo",
    title: "Product demo",
    blurb:
      "Looking to evaluate Petal Edge for your use case or need help with a custom solution?",
    icon: PlaySquare,
    c: "#0ea5e9",
    href: "/book-demo",
  },
  {
    key: "Product feedback",
    title: "Product feedback",
    blurb:
      "Tell us how we're doing! We'd love to hear from you to learn how we can do things better.",
    icon: Lightbulb,
    c: "#10b981",
  },
];

// Per-category copy for the free-text field.
function descriptionCopy(cat: ContactCategory): {
  label: string;
  placeholder: string;
} {
  switch (cat) {
    case "Product feedback":
      return {
        label: "Feedback",
        placeholder:
          "Tell us what's working, what isn't, and how we can do better.",
      };
    case "Technical support":
      return {
        label: "How can we help?",
        placeholder:
          "Describe the issue you're running into, including any steps to reproduce it.",
      };
    case "Product demo":
      return {
        label: "Your use case",
        placeholder:
          "Tell us about your project and what you'd like to explore in a demo.",
      };
    default:
      // Sales inquiries
      return {
        label: "What are you hoping to solve with AI?",
        placeholder:
          "Tell us about your use case and what you're hoping to build.",
      };
  }
}

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

export default function ContactPage() {
  const [theme, toggleTheme] = useHomeTheme();
  const glowRef = useRef<HTMLDivElement | null>(null);

  const [category, setCategory] = useState<ContactCategory | null>(null);
  const [name, setName] = useState("");
  const [mobile, setMobile] = useState("");
  const [email, setEmail] = useState("");
  const [company, setCompany] = useState("");
  const [jobTitle, setJobTitle] = useState("");
  const [description, setDescription] = useState("");
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

  const copy = descriptionCopy(category ?? "Sales inquiries");

  // All fields required except job title.
  const valid = useMemo(
    () =>
      category !== null &&
      name.trim().length > 0 &&
      isValidMobile(mobile) &&
      EMAIL_RE.test(email.trim()) &&
      company.trim().length > 0 &&
      description.trim().length > 0,
    [category, name, mobile, email, company, description]
  );

  function selectCategory(key: ContactCategory) {
    setCategory(key);
    setStatus("idle");
  }

  function changeCategory() {
    setCategory(null);
    setStatus("idle");
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!valid || status === "loading" || category === null) return;
    setStatus("loading");
    try {
      await contactApi.create({
        category,
        name: name.trim(),
        mobile: mobile.trim(),
        email: email.trim(),
        company: company.trim(),
        description: description.trim(),
        job_title: jobTitle.trim() || undefined,
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

      <main className="pe-boost">
        <section className="pe-section">
          <div className="pe-container">
            <div className="pe-section-head">
              <Reveal as="span" className="pe-eyebrow" delay={0}>
                <Headset /> Contact us
              </Reveal>
              <Reveal as="h2" className="pe-h2" delay={60}>
                How can we <span className="pe-grad-text">help you?</span>
              </Reveal>
              <Reveal as="p" className="pe-lede" delay={120}>
                Contact us with any questions, bugs, feedback — just let us know
                what we can help with.
              </Reveal>
            </div>

            {category === null ? (
              /* ---- Category picker -------------------------------------- */
              <div className="pe-contact-grid">
                {CATEGORIES.map((cat, i) => {
                  const Icon = cat.icon;
                  // Cards with an href navigate away (e.g. Product demo →
                  // the existing Book-a-Demo page); the rest reveal the form.
                  const navProps = cat.href
                    ? ({ as: "a", href: cat.href } as const)
                    : ({
                      as: "button",
                      onClick: () => selectCategory(cat.key),
                    } as const);
                  return (
                    <Reveal
                      key={cat.key}
                      {...navProps}
                      className="pe-card pe-card-hover pe-feat-card pe-contact-cat"
                      delay={(i % 2) * 80}
                      aria-label={`${cat.title} — select`}
                    >
                      <span className="pe-ico" style={accent(cat.c)}>
                        <Icon />
                      </span>
                      <h3>{cat.title}</h3>
                      <p>{cat.blurb}</p>
                    </Reveal>
                  );
                })}
              </div>
            ) : (
              /* ---- Selected-category form ------------------------------- */
              <Reveal className="pe-card pe-demo-card" delay={80}>
                {status === "success" ? (
                  <div className="pe-demo-success">
                    <span className="pe-demo-success-ico">
                      <CheckCircle2 />
                    </span>
                    <h3>Thanks! We&apos;ll be in touch shortly.</h3>
                    <p>
                      Your message has been received. Our team will follow up at
                      the email you provided.
                    </p>
                  </div>
                ) : (
                  <form className="pe-demo-form" onSubmit={submit} noValidate>
                    <div className="pe-contact-form-head">
                      <button
                        type="button"
                        className="pe-contact-back"
                        onClick={changeCategory}
                      >
                        <ArrowLeft /> Change category
                      </button>
                      <span className="pe-contact-cat-badge">{category}</span>
                    </div>

                    <div className="pe-demo-grid">
                      <div className="pe-demo-field">
                        <label htmlFor="c-name" className="pe-demo-label">
                          Name <span className="pe-demo-req">*</span>
                        </label>
                        <input
                          id="c-name"
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
                        <label htmlFor="c-mobile" className="pe-demo-label">
                          Mobile number <span className="pe-demo-req">*</span>
                        </label>
                        <input
                          id="c-mobile"
                          type="tel"
                          className="pe-demo-input"
                          placeholder="+91 12345 67890"
                          value={mobile}
                          onChange={(e) => setMobile(e.target.value)}
                          autoComplete="tel"
                          required
                        />
                      </div>

                      <div className="pe-demo-field pe-demo-field-full">
                        <label htmlFor="c-email" className="pe-demo-label">
                          Work email <span className="pe-demo-req">*</span>
                        </label>
                        <input
                          id="c-email"
                          type="email"
                          className="pe-demo-input"
                          placeholder="you@email.com"
                          value={email}
                          onChange={(e) => setEmail(e.target.value)}
                          autoComplete="email"
                          required
                        />
                      </div>

                      <div className="pe-demo-field pe-demo-field-full">
                        <label htmlFor="c-company" className="pe-demo-label">
                          Company / Institution{" "}
                          <span className="pe-demo-req">*</span>
                        </label>
                        <input
                          id="c-company"
                          type="text"
                          className="pe-demo-input"
                          placeholder="Petal Automations"
                          value={company}
                          onChange={(e) => setCompany(e.target.value)}
                          autoComplete="organization"
                          required
                        />
                      </div>

                      <div className="pe-demo-field pe-demo-field-full">
                        <label htmlFor="c-job" className="pe-demo-label">
                          Job title{" "}
                          <span className="pe-demo-optional">(optional)</span>
                        </label>
                        <input
                          id="c-job"
                          type="text"
                          className="pe-demo-input"
                          placeholder="ML Engineer"
                          value={jobTitle}
                          onChange={(e) => setJobTitle(e.target.value)}
                          autoComplete="organization-title"
                        />
                      </div>

                      <div className="pe-demo-field pe-demo-field-full">
                        <label htmlFor="c-description" className="pe-demo-label">
                          {copy.label} <span className="pe-demo-req">*</span>
                        </label>
                        <textarea
                          id="c-description"
                          className="pe-demo-input pe-demo-textarea"
                          placeholder={copy.placeholder}
                          rows={5}
                          value={description}
                          onChange={(e) => setDescription(e.target.value)}
                          required
                        />
                      </div>
                    </div>

                    {status === "error" && (
                      <p className="pe-demo-error" role="alert">
                        Something went wrong sending your message. Please try
                        again.
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
                        "Send message"
                      )}
                    </button>
                  </form>
                )}
              </Reveal>
            )}
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
