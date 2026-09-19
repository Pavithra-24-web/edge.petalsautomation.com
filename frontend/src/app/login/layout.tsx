"use client";

import { useEffect } from "react";

/**
 * Login route layout — isolates the login page from the global theme.
 *
 * The app-wide ThemeController toggles `theme-dark` / `theme-light` classes
 * on <html> and <body>, which in turn drive `var(--app-*)` tokens used by
 * every dashboard surface. The login page should NOT participate in that:
 * its branded purple split-screen always renders the same way regardless
 * of the user's saved theme preference.
 *
 * Strategy:
 *   1. While mounted, tag <body> with `data-pe-route="login"`. The login
 *      CSS uses that attribute to force its own colors with high enough
 *      specificity to override globals.css theme rules.
 *   2. On unmount, remove the tag so the rest of the app reverts to the
 *      user-selected theme cleanly.
 */
export default function LoginLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  useEffect(() => {
    // Tag BOTH <html> and <body>:
    //   - <html> matches the pre-hydration marker set by the root-layout
    //     inline script, keeping the no-flash CSS rule active during SPA
    //     navigations into /login.
    //   - <body> kept for backwards-compat with any selector that targets it.
    // Both are cleared on unmount so the dashboard reverts to the user's
    // saved theme as soon as the user navigates away.
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
