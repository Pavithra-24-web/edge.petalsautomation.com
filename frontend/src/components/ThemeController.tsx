"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { Toaster } from "react-hot-toast";
import { useAppStore } from "@/store/appStore";

/**
 * Default toast style used on the login route — must NEVER follow the
 * user-selected app theme. Mirrors the visual language of the login form
 * (light surface, slate text, subtle border) so notifications fired from
 * /login match the rest of the page regardless of saved app theme.
 */
const LOGIN_TOAST_STYLE = {
  background: "#ffffff",
  color: "#0f172a",
  border: "1px solid #e2e8f0",
};
const LOGIN_TOAST_SECONDARY = "#ffffff";

export default function ThemeController({
  children,
}: {
  children: React.ReactNode;
}) {
  const theme = useAppStore((s) => s.theme);
  const pathname = usePathname();
  const isLoginRoute = pathname === "/login" || pathname?.startsWith("/login/") === true;

  // The persisted theme is only knowable on the client (localStorage). The
  // server always renders with the store's default ("dark"), so reading the
  // rehydrated `theme` during the first client render would diverge from the
  // server markup and break hydration. We can't gate on the store's
  // `hasHydrated` flag — zustand's persist rehydrates synchronously and flips
  // that flag true before this component's first render. A local mount gate is
  // guaranteed false on the first render (server + client), so both agree.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const activeTheme = mounted ? theme : "dark";

  useEffect(() => {
    const root = document.documentElement;
    const body = document.body;
    root.classList.remove("theme-dark", "theme-light");
    body.classList.remove("theme-dark", "theme-light");
    root.classList.add(theme === "light" ? "theme-light" : "theme-dark");
    body.classList.add(theme === "light" ? "theme-light" : "theme-dark");
  }, [theme]);

  // Login route is theme-isolated: force the default light-style toast
  // appearance regardless of the user's saved app theme. Without this the
  // single global Toaster would still leak the app's dark/light theme into
  // login notifications, violating the "login page always default theme"
  // requirement.
  const effectiveStyle = isLoginRoute
    ? LOGIN_TOAST_STYLE
    : activeTheme === "light"
      ? { background: "#ffffff", color: "#111827", border: "1px solid #d1d5db" }
      : { background: "#1e293b", color: "#f1f5f9", border: "1px solid #334155" };

  const effectiveSecondary = isLoginRoute
    ? LOGIN_TOAST_SECONDARY
    : activeTheme === "light"
      ? "#ffffff"
      : "#1e293b";

  return (
    <>
      {children}
      <Toaster
        position="top-right"
        toastOptions={{
          style: effectiveStyle,
          success: { iconTheme: { primary: "#22c55e", secondary: effectiveSecondary } },
          error: { iconTheme: { primary: "#ef4444", secondary: effectiveSecondary } },
        }}
      />
    </>
  );
}
