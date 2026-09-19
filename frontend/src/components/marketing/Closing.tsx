"use client";

import { useState } from "react";
import {
  Star, Check, Plus, ArrowRight, Cpu, Quote, MessageSquare,
  Instagram, Linkedin, Youtube, MapPin, Phone, Mail,
} from "lucide-react";
import { FaXTwitter } from "react-icons/fa6";
import { Reveal } from "./primitives";


/* ============================ PRICING ============================ */
export const PLANS = [
  {
    name: "STARTER", price: "$0", period: "/ forever", highlight: false,
    desc: "For individuals exploring edge AI.",
    cta: "Start Free", ctaClass: "pe-btn-ghost",
    feats: ["1 project", "Community GPU queue", "Image & audio models", "TensorFlow Lite export", "Community support"],
  },
  {
    name: "PROFESSIONAL", price: "$199", period: "/ month", highlight: true,
    desc: "For teams shipping to production.",
    cta: "Start Free", ctaClass: "pe-btn-primary",
    feats: ["Unlimited projects", "Priority GPU training", "All model architectures", "Every deploy target + .pxe", "AI labeling & versioning", "Fleet monitoring", "REST API access"],
  },
  {
    name: "ENTERPRISE", price: "Custom", period: "", highlight: false,
    desc: "For organizations at scale.",
    cta: "Contact Sales", ctaClass: "pe-btn-ghost",
    feats: ["Everything in Professional", "SSO & RBAC", "Dedicated GPU capacity", "Audit logs & data isolation", "SLA & priority support", "On-prem / VPC option"],
  },
];

export function Pricing() {
  return (
    <section className="pe-section" id="pricing">
      <div className="pe-container">
        <div className="pe-section-head">
          <Reveal as="span" className="pe-eyebrow"><Cpu /> Pricing</Reveal>
          <Reveal as="h2" className="pe-h2" delay={60}>
            Simple, <span className="pe-grad-text">scalable</span> pricing
          </Reveal>
          <Reveal as="p" className="pe-lede" delay={120}>
            Start free and grow into production. No credit card required to begin.
          </Reveal>
        </div>

        <div className="pe-price-grid">
          {PLANS.map((p, i) => (
            <Reveal key={p.name} className={`pe-card pe-price-card ${p.highlight ? "pe-hi" : ""}`} delay={i * 80}>
              {p.highlight && <span className="pe-price-tag">Most Popular</span>}
              <div className="pe-price-name pe-grad-text">{p.name}</div>
              <p className="pe-price-desc">{p.desc}</p>
              <div className="pe-price-amt">
                <span className="n">{p.price}</span>
                <span className="p">{p.period}</span>
              </div>
              <ul className="pe-price-feats">
                {p.feats.map((f) => (
                  <li key={f}><Check strokeWidth={3} /> {f}</li>
                ))}
              </ul>
              <a href="/login/" className={`pe-btn ${p.ctaClass}`}>{p.cta}</a>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}


/* ============================ FINAL CTA ============================ */
export function FinalCTA() {
  return (
    <section className="pe-final" id="book-demo">
      <Reveal className="pe-final-inner">
        {[...Array(10)].map((_, i) => (
          <span
            key={i}
            className="pe-final-particle"
            style={{
              width: `${4 + (i % 3) * 3}px`,
              height: `${4 + (i % 3) * 3}px`,
              left: `${8 + i * 9}%`,
              top: `${20 + (i % 4) * 18}%`,
              animationDuration: `${6 + (i % 5)}s`,
              animationDelay: `${i * 0.5}s`,
              opacity: 0.5,
            }}
          />
        ))}
        <h2>Start Building Production-Ready Edge AI Today</h2>
        <p>Join the teams shipping intelligent products to every edge device.</p>
        <div className="pe-final-cta">
          <a href="/login/" className="pe-btn pe-btn-primary pe-btn-lg">
            Get Started <ArrowRight />
          </a>
          <a href="/book-demo" className="pe-btn pe-btn-ghost pe-btn-lg">
            Book Demo
          </a>
        </div>
      </Reveal>
    </section>
  );
}

/* ============================ FOOTER ============================ */
export function Footer() {
  return (
    <footer className="pe-footer">
      <div className="pe-container">
        <div className="pe-foot-grid pe-foot-grid-compact">
          <div className="pe-foot-brand">
            <a href="/login" className="pe-brand">
              <img src="/favicon.svg" alt="Petal Edge logo" width={30} height={30} className="pe-brand-mark" style={{ background: "transparent", boxShadow: "none" }} />
              Petal&nbsp;Edge
            </a>
            <p>The complete enterprise platform for building, training, deploying, and monitoring AI on every edge device.</p>
            <div className="pe-foot-social">
              {[
                { Icon: Instagram, href: "https://www.instagram.com/petals_automation/", label: "Instagram" },
                { Icon: FaXTwitter, href: "https://x.com/Petals_Auto", label: "X" },
                { Icon: Linkedin, href: "https://www.linkedin.com/company/petals-automation/posts/?feedView=all", label: "LinkedIn" },
                { Icon: Youtube, href: "https://www.youtube.com/@PetalAIandRobotics-bv4vl", label: "YouTube" },
              ].map(({ Icon, href, label }) => (
                <a key={label} className="pe-icon-btn" href={href} target="_blank" rel="noopener noreferrer" aria-label={label}><Icon /></a>
              ))}
            </div>
          </div>
          <div className="pe-foot-col pe-foot-contact">
            <h4>Contact</h4>
            <p className="pe-foot-contact-item">
              <MapPin />
              <span>NO.155, First Floor, LIG Colony, KK Nagar,<br />Madurai - 625020, Tamil Nadu - India.</span>
            </p>
            <a className="pe-foot-contact-item" href="tel:+918838508804">
              <Phone />
              <span>+91 8838508804</span>
            </a>
            <a className="pe-foot-contact-item" href="mailto:contact@petalautomations.com">
              <Mail />
              <span>contact@petalautomations.com</span>
            </a>
            <a className="pe-foot-contact-item" href="tel:+9104524396547">
              <Phone />
              <span>+91 0452 439 6547</span>
            </a>
          </div>
        </div>
        <div className="pe-foot-bottom">
          <span>© {new Date().getFullYear()} Petal Edge. All rights reserved.</span>
          <span>Privacy · Terms · Security</span>
        </div>
      </div>
    </footer>
  );
}
