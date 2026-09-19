/**
 * Global Zustand store — auth, active project, UI state
 */
import { create } from "zustand";
import { persist } from "zustand/middleware";

interface User {
  id: string;
  email: string;
  username: string;
  role: string;
}

/** A catalog entry as served by /device-catalog. `deploy_target` is null only
 *  for an unresolved selection — a slug whose catalog row no longer exists. */
export interface TargetDevice {
  slug: string;
  display_name: string;
  family: string | null;
  deploy_target: string | null;
  accelerator_note: string | null;
  unresolved?: boolean;
}

/** One overridable application-budget value (Target Device Phase 3). `value`
 *  is what later phases read — the override if set, otherwise `board_default`.
 *  Either can be null: an unresearched specification stays null through
 *  resolution rather than being replaced with a guessed number. */
export interface TargetDeviceOverrideField {
  value: number | null;
  board_default: number | null;
  overridden: boolean;
}

/** The resolved per-project target-device configuration — override wins over
 *  specification, board default carried alongside for comparison. `null`
 *  when no device is selected, same as `target_device`. Processor identity
 *  fields have no project-level storage at all, so they're always read-only. */
export interface TargetDeviceConfig {
  custom_name: string | null;
  processor_family: string | null;
  processor: string | null;
  cpu_architecture: string | null;
  clock_rate_mhz: number | null;
  ram_kb: TargetDeviceOverrideField;
  rom_kb: TargetDeviceOverrideField;
  latency_ms: TargetDeviceOverrideField;
}

interface Project {
  id: string;
  name: string;
  description?: string;
  /** Which content the shared dashboard pages render for this project.
   *  Presentation-only — set once at creation, never re-read by the
   *  pipeline. Absent on state persisted before this field existed;
   *  treated the same as "object_detection" wherever it's read. */
  project_type?: "object_detection" | "motion";
  /** The hardware this project targets; null/absent = not chosen, a valid state. */
  target_device_slug?: string | null;
  target_device?: TargetDevice | null;
  target_device_config?: TargetDeviceConfig | null;
}

/** Persisted state for the post-processing page, keyed by a page-scope key. */
export interface PersistedPPState {
  selectedSampleId: string | null;
  jobId: string | null;
  /** null = no job started yet; other values mirror VideoJobStatus. */
  status: string | null;
  outputUrl: string | null;
  errorMessage: string | null;
}

export interface PersistedFeatureGenerationState {
  logs: string[];
  jobId: string | null;
  status: string;
  progress: number;
}

/** First-time onboarding tour progress, kept per user.id.
 *  - active:    tour is currently showing (resumes at `step` on reload)
 *  - completed: user finished the tour — never auto-shows again
 *  - skipped:   user dismissed the tour — never auto-shows again
 */
export interface OnboardingState {
  status: "active" | "completed" | "skipped";
  step: number;
}

interface AppState {
  hasHydrated: boolean;
  setHasHydrated: (value: boolean) => void;

  // Auth
  user: User | null;
  token: string | null;
  setAuth: (user: User, token: string) => void;
  logout: () => void;
  isAuthenticated: () => boolean;

  // Active project
  activeProject: Project | null;
  setActiveProject: (p: Project | null) => void;

  // Sidebar
  sidebarOpen: boolean;
  setSidebarOpen: (v: boolean) => void;

  // Theme
  theme: "dark" | "light";
  setTheme: (theme: "dark" | "light") => void;

  // Active training job (for live progress)
  activeTrainingJobId: string | null;
  setActiveTrainingJobId: (id: string | null) => void;

  // Automatically persists cross-route DSP navigation
  activeImpulse: any | null;
  setActiveImpulse: (i: any | null) => void;
  savedActiveImpulse: any | null;
  setSavedActiveImpulse: (i: any | null) => void;

  // Profile extras — fields that the backend User model does not (yet)
  // expose: name (display name), job title, company. Keyed by user.id so
  // every signed-in account keeps its own values. Backend-free for now;
  // when an `/auth/me` update endpoint lands, the settings page can sync
  // these up without changing the slice shape.
  profileExtras: Record<string, { name?: string; jobTitle?: string; companyName?: string }>;
  setProfileExtras: (userId: string, extras: { name?: string; jobTitle?: string; companyName?: string }) => void;

  // Post-processing page state — persisted per project/impulse scope key
  ppPage: Record<string, PersistedPPState>;
  setPPPage: (pageKey: string, state: PersistedPPState) => void;
  clearPPPage: (pageKey: string) => void;
  featureGenerationPage: Record<string, PersistedFeatureGenerationState>;
  setFeatureGenerationPage: (impulseId: string, state: PersistedFeatureGenerationState) => void;
  clearFeatureGenerationPage: (impulseId: string) => void;

  // First-time onboarding tour — keyed by user.id so it is per-account and
  // survives reloads via the persisted store (mirrors profileExtras).
  onboarding: Record<string, OnboardingState>;
  // Begin the tour for a first-time user. No-op if the user already has a
  // record (they finished/skipped it, or it is already running) — this makes
  // it safe to call on every `is_new_user` login without re-showing.
  startOnboarding: (userId: string) => void;
  // Force the tour back to step 0 regardless of prior status (Restart action).
  restartOnboarding: (userId: string) => void;
  setOnboardingStep: (userId: string, step: number) => void;
  finishOnboarding: (userId: string) => void;
  skipOnboarding: (userId: string) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      hasHydrated: false,
      setHasHydrated: (value) => set({ hasHydrated: value }),

      user:  null,
      token: null,
      setAuth: (user, token) => {
        if (typeof window !== "undefined") {
          localStorage.setItem("access_token", token);
        }
        // Guard against cross-account state bleed: when a *different* user
        // signs in, drop the previous account's persisted project/impulse
        // selection before installing the new identity. (A fresh login after
        // a tab close skips logout(), which is the other reset point.)
        const prev = get().user;
        const switchingUser = prev && prev.id !== user.id;
        set({
          user,
          token,
          ...(switchingUser
            ? { activeProject: null, activeImpulse: null, savedActiveImpulse: null }
            : {}),
        });
      },
      logout: () => {
        if (typeof window !== "undefined") {
          localStorage.removeItem("access_token");
        }
        set({
          user: null,
          token: null,
          activeProject: null,
          activeImpulse: null,
          savedActiveImpulse: null,
          ppPage: {},
          featureGenerationPage: {},
        });
      },
      isAuthenticated: () => !!get().token,

      activeProject: null,
      setActiveProject: (p) => set({ activeProject: p }),

      sidebarOpen: true,
      setSidebarOpen: (v) => set({ sidebarOpen: v }),

      theme: "dark",
      setTheme: (theme) => set({ theme }),

      activeTrainingJobId: null,
      setActiveTrainingJobId: (id) => set({ activeTrainingJobId: id }),

      activeImpulse: null,
      setActiveImpulse: (i) => set({ activeImpulse: i }),
      savedActiveImpulse: null,
      setSavedActiveImpulse: (i) => set({ savedActiveImpulse: i }),

      profileExtras: {},
      setProfileExtras: (userId, extras) =>
        set((s) => ({
          profileExtras: {
            ...s.profileExtras,
            [userId]: { ...(s.profileExtras[userId] || {}), ...extras },
          },
        })),

      ppPage: {},
      setPPPage: (pageKey, state) =>
        set((s) => ({ ppPage: { ...s.ppPage, [pageKey]: state } })),
      clearPPPage: (pageKey) =>
        set((s) => {
          const next = { ...s.ppPage };
          delete next[pageKey];
          return { ppPage: next };
        }),

      featureGenerationPage: {},
      setFeatureGenerationPage: (impulseId, state) =>
        set((s) => ({
          featureGenerationPage: {
            ...s.featureGenerationPage,
            [impulseId]: state,
          },
        })),
      clearFeatureGenerationPage: (impulseId) =>
        set((s) => {
          const next = { ...s.featureGenerationPage };
          delete next[impulseId];
          return { featureGenerationPage: next };
        }),

      onboarding: {},
      startOnboarding: (userId) =>
        set((s) => {
          if (!userId || s.onboarding[userId]) return s; // already seen/running
          return { onboarding: { ...s.onboarding, [userId]: { status: "active", step: 0 } } };
        }),
      restartOnboarding: (userId) =>
        set((s) => ({
          onboarding: { ...s.onboarding, [userId]: { status: "active", step: 0 } },
        })),
      setOnboardingStep: (userId, step) =>
        set((s) => {
          const cur = s.onboarding[userId];
          if (!cur) return s;
          return { onboarding: { ...s.onboarding, [userId]: { ...cur, step } } };
        }),
      finishOnboarding: (userId) =>
        set((s) => ({
          onboarding: {
            ...s.onboarding,
            [userId]: { status: "completed", step: s.onboarding[userId]?.step ?? 0 },
          },
        })),
      skipOnboarding: (userId) =>
        set((s) => ({
          onboarding: {
            ...s.onboarding,
            [userId]: { status: "skipped", step: s.onboarding[userId]?.step ?? 0 },
          },
        })),
    }),
    {
      name: "petaledge-store",
      onRehydrateStorage: () => (state) => {
        state?.setHasHydrated(true);
      },
      partialize: (state) => ({
        user:          state.user,
        token:         state.token,
        theme:         state.theme,
        activeProject: state.activeProject,
        activeImpulse: state.activeImpulse,
        savedActiveImpulse: state.savedActiveImpulse,
        profileExtras: state.profileExtras,
        ppPage:        state.ppPage,
        featureGenerationPage: state.featureGenerationPage,
        onboarding:    state.onboarding,
      }),
    }
  )
);
