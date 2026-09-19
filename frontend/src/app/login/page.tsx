"use client";
import { Suspense, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import toast from "react-hot-toast";
import { Mail, Lock, User as UserIcon, Eye, EyeOff } from "lucide-react";

import { useAppStore } from "@/store/appStore";
import { authApi } from "@/utils/api";
import PetalEdgeLogo from "@/components/PetalEdgeLogo";

const GOOGLE_CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || "";

export default function LoginPage() {
  // useSearchParams in LoginContent requires a Suspense boundary for static prerender.
  return (
    <Suspense fallback={null}>
      <LoginContent />
    </Suspense>
  );
}

function LoginContent() {
  const searchParams = useSearchParams();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [registered, setRegistered] = useState(false);
  const [showUnverified, setShowUnverified] = useState(false);
  const [resendCooldown, setResendCooldown] = useState(0);
  const [resendLoading, setResendLoading] = useState(false);
  const { setAuth, startOnboarding } = useAppStore();
  const router = useRouter();
  const googleButtonRef = useRef<HTMLDivElement>(null);

  // Arriving from an expired verification link (/login?verify=expired):
  // auto-show the unverified banner so the resend button is immediately
  // available; prefill the email if the redirect carried one.
  useEffect(() => {
    if (searchParams.get("verify") === "expired") {
      setShowUnverified(true);
      const emailParam = searchParams.get("email");
      if (emailParam) setEmail(emailParam);
      // Clear any stale/autofilled password so another account's saved
      // credentials don't linger next to the link's email.
      setPassword("");
    }
  }, [searchParams]);

  // On the expired-link arrival, suppress the browser's convenience autofill so
  // it can't overwrite the prefilled email or inject a saved password from
  // another account (the bug this flow fixes).
  const fromExpiredLink = searchParams.get("verify") === "expired";

  // Ticks the resend cooldown down once per second until it reaches zero.
  useEffect(() => {
    if (resendCooldown <= 0) return;
    const t = setTimeout(() => setResendCooldown(c => c - 1), 1000);
    return () => clearTimeout(t);
  }, [resendCooldown]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErrorMsg(null);
    setShowUnverified(false);
    setLoading(true);
    try {
      if (mode === "login") {
        const { data } = await authApi.login({ email, password });
        setAuth({ id: data.user_id, email: data.email || email, username: data.username, role: "developer" }, data.access_token);
        router.push("/dashboard");
      } else {
        await authApi.register({ email, username, password });
        setRegistered(true);
      }
    } catch (e: any) {
      const detail = e?.response?.data?.detail;
      if (e?.response?.status === 403 && typeof detail === "string" && /not verified/i.test(detail)) {
        setShowUnverified(true);
        return;
      }
      const msg = typeof detail === "string" ? detail : "Authentication failed";
      setErrorMsg(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  }

  async function resendVerification() {
    if (!email || resendLoading || resendCooldown > 0) return;
    setResendLoading(true);
    try {
      await authApi.resendVerification(email);
      toast.success("A new verification email has been sent. Please check your inbox.");
      setResendCooldown(60);
    } catch {
      toast.error("Could not send the verification email. Please try again.");
    } finally {
      setResendLoading(false);
    }
  }

  async function onGoogleCredential(response: { credential: string }) {
    setErrorMsg(null);
    setLoading(true);
    try {
      const { data } = await authApi.google({ credential: response.credential });
      setAuth(
        { id: data.user_id, email: data.email || "", username: data.username, role: "developer" },
        data.access_token,
      );
      // First-ever sign-in → arm the onboarding tour (no-op for returning
      // users, and idempotent if they've already seen/skipped it).
      if (data.is_new_user) startOnboarding(data.user_id);
      router.push("/dashboard");
    } catch (e: any) {
      const detail = e?.response?.data?.detail;
      const msg = typeof detail === "string" ? detail : "Google sign-in failed";
      setErrorMsg(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  }

  // Load the Google Identity Services script once and render the button into
  // googleButtonRef. Google's own widget (not our markup) owns the click
  // handling and returns an ID token via onGoogleCredential.
  useEffect(() => {
    if (!GOOGLE_CLIENT_ID) return;

    function render() {
      const google = (window as any).google;
      if (!google?.accounts?.id || !googleButtonRef.current) return;
      google.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID,
        callback: onGoogleCredential,
      });
      google.accounts.id.renderButton(googleButtonRef.current, {
        theme: "outline",
        size: "large",
        width: 328,
      });
    }

    const existing = document.getElementById("google-identity-script");
    if (existing) {
      render();
      return;
    }
    const script = document.createElement("script");
    script.id = "google-identity-script";
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.defer = true;
    script.onload = render;
    document.head.appendChild(script);
  }, []);

  const year = new Date().getFullYear();

  return (
    <div className="pe-signin-page">
      {/* ── LEFT — branded panel ───────────────────────────────────────── */}
      <aside className="pe-signin-left" aria-hidden="false">
        <span className="pe-signin-bg-blob pe-signin-bg-blob--tl" aria-hidden="true" />
        <span className="pe-signin-bg-blob pe-signin-bg-blob--br" aria-hidden="true" />
        <span className="pe-signin-bg-dots pe-signin-bg-dots--tl" aria-hidden="true" />
        <span className="pe-signin-bg-dots pe-signin-bg-dots--br" aria-hidden="true" />

        <div className="pe-signin-brand">
          <PetalEdgeLogo variant="login" />
          <p className="pe-signin-tagline">Open-source Edge AI Platform</p>
        </div>

        <div className="pe-signin-marketing">
          <h2 className="pe-signin-marketing-title">
            <span className="pe-signin-mk-line pe-signin-mk-line--solid">Build Smarter.</span>
            <span className="pe-signin-mk-line pe-signin-mk-line--soft">Deploy Everywhere.</span>
          </h2>
          <p className="pe-signin-marketing-body">
            PetalEdge empowers you to build, train, and deploy AI models to edge
            devices effortlessly. Open source. Flexible. Powerful.
          </p>
          <div className="pe-signin-foot">
            {/* `year` comes from new Date() at render time. This page is
                statically prerendered, so the build-time year is baked into
                the HTML and can differ from the client's current year (e.g.
                built in Dec, opened in Jan), causing a hydration mismatch.
                suppressHydrationWarning lets the client's value win cleanly. */}
            <span suppressHydrationWarning>© {year} PetalEdge</span>
            <span className="pe-signin-foot-sep" aria-hidden="true"></span>
          </div>
        </div>
      </aside>

      {/* ── RIGHT — form panel ─────────────────────────────────────────── */}
      <section className="pe-signin-right">
        <div className="pe-signin-form-wrap">
          {registered ? (
            <>
              <header className="pe-signin-head">
                <h1 className="pe-signin-title">Check your email</h1>
                <p className="pe-signin-sub">
                  We sent a verification link to <strong>{email}</strong>.
                  Verify your
                  account to activate it.
                </p>
              </header>

            </>
          ) : (
            <>
              <header className="pe-signin-head">
                <h1 className="pe-signin-title">
                  {mode === "login" ? "Welcome Back" : "Create your account"}
                </h1>
                <p className="pe-signin-sub">
                  {mode === "login"
                    ? "Sign in to access your dashboard."
                    : "Register with your email to get started."}
                </p>
              </header>

              <form onSubmit={submit} className="pe-signin-form" autoComplete="off">
                <div className="pe-signin-field">
                  <div className="pe-signin-label-row">
                    <label className="pe-signin-label">Email</label>
                  </div>
                  <div className="pe-signin-input-wrap">
                    <span className="pe-signin-input-icon"><Mail size={17} /></span>
                    <input
                      type="email"
                      name="login-email"
                      className="pe-signin-input"
                      placeholder="you@example.com"
                      value={email}
                      onChange={e => setEmail(e.target.value)}
                      autoComplete="off"
                      required
                    />
                  </div>
                </div>

                {mode === "register" && (
                  <div className="pe-signin-field">
                    <div className="pe-signin-label-row">
                      <label className="pe-signin-label">Username</label>
                    </div>
                    <div className="pe-signin-input-wrap">
                      <span className="pe-signin-input-icon"><UserIcon size={17} /></span>
                      <input
                        type="text"
                        className="pe-signin-input"
                        placeholder="yourusername"
                        value={username}
                        onChange={e => setUsername(e.target.value)}
                        required
                      />
                    </div>
                  </div>
                )}

                <div className="pe-signin-field">
                  <div className="pe-signin-label-row">
                    <label className="pe-signin-label">Password</label>
                    {mode === "login" && (
                      <a href="/forgot-password" className="pe-signin-forgot">
                        Forgot password?
                      </a>
                    )}
                  </div>
                  <div className="pe-signin-input-wrap">
                    <span className="pe-signin-input-icon"><Lock size={17} /></span>
                    <input
                      type={showPassword ? "text" : "password"}
                      className="pe-signin-input"
                      placeholder="••••••••"
                      value={password}
                      onChange={e => setPassword(e.target.value)}
                      autoComplete="new-password"
                      required
                      minLength={8}
                    />
                    <button
                      type="button"
                      className="pe-signin-eye"
                      aria-label={showPassword ? "Hide password" : "Show password"}
                      onClick={() => setShowPassword(v => !v)}
                    >
                      {showPassword ? <EyeOff size={17} /> : <Eye size={17} />}
                    </button>
                  </div>
                </div>

                {mode === "login" && showUnverified && (
                  <div role="alert" aria-live="assertive" className="pe-signin-warning">
                    <span>
                      Your email address has not been verified. Please verify your
                      email before signing in.
                    </span>
                    <button
                      type="button"
                      className="pe-signin-warning-resend"
                      onClick={resendVerification}
                      disabled={!email || resendLoading || resendCooldown > 0}
                    >
                      {resendCooldown > 0
                        ? `Resend available in ${resendCooldown}s`
                        : resendLoading
                          ? "Sending…"
                          : "Resend Verification Email"}
                    </button>
                  </div>
                )}

                {errorMsg && (
                  <div role="alert" aria-live="assertive" className="pe-signin-error">
                    {errorMsg}
                  </div>
                )}

                <button type="submit" disabled={loading} className="pe-signin-submit">
                  {loading ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
                </button>
              </form>

              <p className="pe-signin-switch">
                {mode === "login" ? (
                  <>
                    Don&apos;t have an account?{" "}
                    <button type="button" className="pe-signin-link" onClick={() => setMode("register")}>
                      Sign up
                    </button>
                  </>
                ) : (
                  <>
                    Already have an account?{" "}
                    <button type="button" className="pe-signin-link" onClick={() => setMode("login")}>
                      Sign in
                    </button>
                  </>
                )}
              </p>

              {GOOGLE_CLIENT_ID && (
                <div className="pe-signin-google-primary">
                  <div className="pe-signin-divider">or</div>
                  <div ref={googleButtonRef} className="pe-signin-google" />
                </div>
              )}

              {loading && <p className="pe-signin-sub">Signing you in…</p>}
            </>
          )}
        </div>
      </section>
    </div>
  );
}
