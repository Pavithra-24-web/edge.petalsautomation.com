import type { Metadata } from "next";
import "./globals.css";
import ThemeController from "@/components/ThemeController";

export const metadata: Metadata = {
  title: "PetalEdge — Edge AI Platform",
  description: "Open-source edge ML platform for collecting data, training models, and deploying to edge devices",
  icons: {
    icon: [
      { url: "/favicon.svg", type: "image/svg+xml" },
      { url: "/favicon-48.png", type: "image/png", sizes: "48x48" },
      { url: "/favicon-32.png", type: "image/png", sizes: "32x32" },
      { url: "/favicon-16.png", type: "image/png", sizes: "16x16" },
    ],
    apple: { url: "/apple-touch-icon.png", type: "image/png", sizes: "180x180" },
    shortcut: "/favicon.ico",
  },
};

/**
 * Pre-hydration route-marker script.
 *
 * The login page is theme-isolated via `html[data-pe-route="login"]` CSS
 * overrides. If we only set that attribute in the LoginLayout's useEffect,
 * a hard refresh of `/login` would still show a single-frame flash of the
 * user's saved app theme (whatever ThemeController applies) before the
 * effect runs.
 *
 * This script runs synchronously while the browser is still parsing
 * `<head>`, BEFORE any body content paints. It checks `location.pathname`
 * and sets the marker on `<html>` immediately, so the login CSS overrides
 * apply on the very first paint with zero flash. `document.documentElement`
 * always exists in `<head>`; `document.body` does not yet, so we only
 * touch the html element here. The LoginLayout's useEffect still handles
 * SPA navigations into / out of the login route.
 */
const PRE_HYDRATION_ROUTE_SCRIPT = `
(function () {
  try {
    var p = window.location.pathname;
    if (p === "/login" || p.indexOf("/login/") === 0) {
      document.documentElement.setAttribute("data-pe-route", "login");
    }
  } catch (e) { /* never break first paint */ }
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full bg-gray-950">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
        {/* Must be inline + synchronous so the route marker exists before
            the first paint of <body>. Do not move this below body content. */}
        <script dangerouslySetInnerHTML={{ __html: PRE_HYDRATION_ROUTE_SCRIPT }} />
      </head>
      <body className="h-full font-sans antialiased">
        <ThemeController>
          {children}
        </ThemeController>
      </body>
    </html>
  );
}
