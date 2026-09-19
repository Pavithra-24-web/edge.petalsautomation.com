"use client";
import { Suspense, useRef, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Lock, Eye, EyeOff } from "lucide-react";
import toast from "react-hot-toast";

import { authApi } from "@/utils/api";

export default function ResetPasswordPage() {
  return (
    <Suspense fallback={null}>
      <ResetPasswordForm />
    </Suspense>
  );
}

function ResetPasswordForm() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const token = searchParams.get("token") ?? "";

  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [clientError, setClientError] = useState<string | null>(null);

  // Prevent double-submit in Strict Mode / accidental double-click.
  const submitting = useRef(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setClientError(null);

    if (!token) {
      toast.error("This password reset link is invalid or has expired. Please request a new password reset email.");
      router.replace("/login");
      return;
    }
    if (password !== confirm) {
      setClientError("Passwords do not match.");
      return;
    }
    if (password.length < 8) {
      setClientError("Password must be at least 8 characters.");
      return;
    }
    if (submitting.current) return;
    submitting.current = true;
    setLoading(true);

    try {
      await authApi.resetPassword({ token, password, confirm_password: confirm });
      toast.success("Password changed successfully.");
      router.replace("/login");
    } catch (err: any) {
      // Any 400 means the token is invalid, expired, or already used.
      toast.error(
        "This password reset link is invalid or has expired. Please request a new password reset email."
      );
      router.replace("/login");
    } finally {
      setLoading(false);
      submitting.current = false;
    }
  }

  return (
    <div className="pe-signin-right" style={{ minHeight: "100vh", width: "100%" }}>
      <div className="pe-signin-form-wrap">
        <header className="pe-signin-head">
          <h1 className="pe-signin-title">Choose a new password</h1>
          <p className="pe-signin-sub">
            Enter and confirm your new password below.
          </p>
        </header>

        <form onSubmit={submit} className="pe-signin-form">
          <div className="pe-signin-field">
            <div className="pe-signin-label-row">
              <label className="pe-signin-label">New Password</label>
            </div>
            <div className="pe-signin-input-wrap">
              <span className="pe-signin-input-icon">
                <Lock size={17} />
              </span>
              <input
                type={showPassword ? "text" : "password"}
                className="pe-signin-input"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="new-password"
                required
                minLength={8}
              />
              <button
                type="button"
                className="pe-signin-eye"
                aria-label={showPassword ? "Hide password" : "Show password"}
                onClick={() => setShowPassword((v) => !v)}
              >
                {showPassword ? <EyeOff size={17} /> : <Eye size={17} />}
              </button>
            </div>
          </div>

          <div className="pe-signin-field">
            <div className="pe-signin-label-row">
              <label className="pe-signin-label">Confirm Password</label>
            </div>
            <div className="pe-signin-input-wrap">
              <span className="pe-signin-input-icon">
                <Lock size={17} />
              </span>
              <input
                type={showConfirm ? "text" : "password"}
                className="pe-signin-input"
                placeholder="••••••••"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                autoComplete="new-password"
                required
                minLength={8}
              />
              <button
                type="button"
                className="pe-signin-eye"
                aria-label={showConfirm ? "Hide password" : "Show password"}
                onClick={() => setShowConfirm((v) => !v)}
              >
                {showConfirm ? <EyeOff size={17} /> : <Eye size={17} />}
              </button>
            </div>
          </div>

          {clientError && (
            <div role="alert" aria-live="assertive" className="pe-signin-error">
              {clientError}
            </div>
          )}

          <button type="submit" disabled={loading} className="pe-signin-submit">
            {loading ? "Saving…" : "Reset Password"}
          </button>
        </form>
      </div>
    </div>
  );
}
