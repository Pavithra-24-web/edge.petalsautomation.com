"use client";

import { useEffect } from "react";

/**
 * Verify-email route layout — shares the login page's branded split-screen
 * styling. See src/app/login/layout.tsx for why data-pe-route="login" is
 * used to opt out of the app-wide theme.
 */
export default function VerifyEmailLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  useEffect(() => {
    const html = document.documentElement;
    const body = document.body;
    html.setAttribute("data-pe-route", "login");
    body.setAttribute("data-pe-route", "login");
    return () => {
      html.removeAttribute("data-pe-route");
      body.removeAttribute("data-pe-route");
    };
  }, []);

  return <div className="pe-signin-isolate">{children}</div>;
}
