"use client";
import { useEffect, useState } from "react";
import { Mail } from "lucide-react";
import toast from "react-hot-toast";
import Link from "next/link";

import { authApi } from "@/utils/api";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [cooldown, setCooldown] = useState(0);

  useEffect(() => {
    if (cooldown <= 0) return;
    const t = setTimeout(() => setCooldown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [cooldown]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (loading || cooldown > 0) return;
    setLoading(true);
    try {
      await authApi.forgotPassword(email);
      setSubmitted(true);
      setCooldown(60);
    } catch {
      // The endpoint always returns 200 — network errors are the only failure path.
      toast.error("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="pe-signin-right" style={{ minHeight: "100vh", width: "100%" }}>
      <div className="pe-signin-form-wrap">
        <header className="pe-signin-head">
          <h1 className="pe-signin-title">Forgot your password?</h1>
          <p className="pe-signin-sub">
            {submitted
              ? "If an account exists for this email, a password reset link has been sent. Check your inbox."
              : "Enter your email address and we'll send you a reset link."}
          </p>
        </header>

        {!submitted ? (
          <form onSubmit={submit} className="pe-signin-form">
            <div className="pe-signin-field">
              <div className="pe-signin-label-row">
                <label className="pe-signin-label">Email</label>
              </div>
              <div className="pe-signin-input-wrap">
                <span className="pe-signin-input-icon">
                  <Mail size={17} />
                </span>
                <input
                  type="email"
                  className="pe-signin-input"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoComplete="email"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading || cooldown > 0}
              className="pe-signin-submit"
            >
              {loading
                ? "Sending…"
                : cooldown > 0
                  ? `Resend available in ${cooldown}s`
                  : "Send Reset Link"}
            </button>
          </form>
        ) : (
          <button
            type="button"
            disabled={cooldown > 0}
            className="pe-signin-submit"
            style={{ marginTop: "0.5rem" }}
            onClick={() => {
              if (cooldown > 0) return;
              setSubmitted(false);
            }}
          >
            {cooldown > 0 ? `Resend available in ${cooldown}s` : "Send another link"}
          </button>
        )}

        <p className="pe-signin-switch" style={{ marginTop: "1.25rem" }}>
          <Link href="/login" className="pe-signin-link">
            Back to sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
