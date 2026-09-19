"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAppStore } from "@/store/appStore";
import { authApi } from "@/utils/api";
import Sidebar from "@/components/dashboard/Sidebar";
import TopBar from "@/components/dashboard/TopBar";
import OnboardingProvider from "@/components/onboarding/OnboardingProvider";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { token, user, setAuth } = useAppStore();
  const router = useRouter();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (mounted && !token) {
      router.push("/login");
    }
  }, [token, router, mounted]);

  // Self-heal sessions persisted before the API returned email (older Google
  // logins stored email:"") — backfill the full profile from /auth/me once.
  useEffect(() => {
    if (mounted && token && user && !user.email) {
      authApi.me().then(({ data }) => setAuth(data, token)).catch(() => {});
    }
  }, [mounted, token, user, setAuth]);

  if (!mounted || !token) return null;

  return (
    <div className="shell-app flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex flex-col flex-1 min-w-0">
        <TopBar />
        <main className="dashboard-main flex-1 overflow-y-auto p-6 lg:p-7">
          {children}
        </main>
      </div>
      <OnboardingProvider />
    </div>
  );
}
