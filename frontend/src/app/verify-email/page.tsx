"use client";
import { Suspense, useEffect, useRef } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import toast from "react-hot-toast";

import { authApi } from "@/utils/api";
import { useAppStore } from "@/store/appStore";

export default function VerifyEmailPage() {
  return (
    <Suspense fallback={null}>
      <VerifyEmailRedirect />
    </Suspense>
  );
}

function VerifyEmailRedirect() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const token = searchParams.get("token");

  // Guards against React Strict Mode's dev double-invoke, remounts, or a
  // duplicate click firing this effect twice for the same token — the
  // verification token is only valid for one successful use, so a second
  // real network call would otherwise race the first.
  const requestedTokenRef = useRef<string | null>(null);

  useEffect(() => {
    if (!token) {
      toast.error("Invalid verification link.");
      router.replace("/login");
      return;
    }
    if (requestedTokenRef.current === token) {
      return;
    }
    requestedTokenRef.current = token;

    authApi
      .verifyEmail(token)
      .then(({ data }) => {
        const responseMessage: string = data?.message || "";
        if (/already verified/i.test(responseMessage)) {
          toast.success("Email already verified.");
        } else {
          toast.success("Email verified successfully. Welcome!");
        }
        if (data?.access_token) {
          useAppStore.getState().setAuth(
            { id: data.user_id, email: data.email || "", username: data.username, role: "developer" },
            data.access_token,
          );
          router.replace("/dashboard");
        } else {
          router.replace(useAppStore.getState().isAuthenticated() ? "/dashboard" : "/login");
        }
      })
      .catch((e: any) => {
        const detail = e?.response?.data?.detail;
        // The expired path now returns a structured detail ({ code: "expired",
        // email }); older/other errors are bare strings.
        const isExpired =
          (detail && typeof detail === "object" && detail.code === "expired") ||
          (typeof detail === "string" && /expired/i.test(detail));
        if (isExpired) {
          toast.error("Your verification link has expired. Please request a new verification email.");
          // Prefill the login form with the link's own email (only the token
          // holder reaches this branch), falling back to no prefill.
          const email = detail && typeof detail === "object" ? detail.email : "";
          router.replace(
            email
              ? `/login?verify=expired&email=${encodeURIComponent(email)}`
              : "/login?verify=expired"
          );
        } else {
          toast.error("Invalid verification link.");
          router.replace("/login");
        }
      });
  }, [token, router]);

  return null;
}
