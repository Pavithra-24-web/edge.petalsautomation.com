"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAppStore } from "@/store/appStore";
import { impulsesApi } from "@/utils/api";
import toast from "react-hot-toast";
import {
  LayoutDashboard, Database, Cpu,
  Rocket, Smartphone, LogOut,
  PlayCircle, CheckCircle2, SlidersHorizontal, Tags, RefreshCw,
  ChevronDown, ChevronUp, Activity, ImageIcon, ScanSearch,
  FlaskConical,
  SquareBottomDashedScissorsIcon,
  MenuIcon,
  AlbumIcon,
  WifiIcon,
  BoxesIcon,
  LocateIcon,
  GamepadIcon,
  GitPullRequestCreateArrowIcon,
  CircleArrowOutDownLeftIcon,
  AlignVerticalJustifyEnd,
  AlignVerticalJustifyEndIcon,
  LucideAlignVerticalJustifyCenter,
  MessageSquareCodeIcon,
  RotateCw,
  Radar,
  BadgeCheck,
  ShieldCheck,
  CloudUpload,
  Package,
  History,
  ListChecks,
  Workflow,
  ClipboardList,
} from "lucide-react";
import PetalEdgeLogo from "@/components/PetalEdgeLogo";

type NavItemDef = {
  label: string;
  href: string;
  icon?: any;
  isDynamic?: boolean;
  dspBlock?: boolean;
  iconColor?: string;
  /** Anchor id for the onboarding tour (data-tour attribute). */
  tourId?: string;
};

const STATIC_IMPULSE_ITEMS_BEFORE: NavItemDef[] = [
  { label: "Create impulse", href: "/dashboard/impulse", icon: MenuIcon, iconColor: "#6366F1", tourId: "nav-create-impulse" },
];
const STATIC_IMPULSE_ITEMS_AFTER: NavItemDef[] = [
  { label: "Evaluation", href: "/dashboard/impulse/evaluation", icon: ListChecks, iconColor: "#7C3AED" },
  { label: "Retrain model", href: "/dashboard/impulse/retrain", icon: RotateCw, iconColor: "#EF4444" },
  { label: "Live classification", href: "/dashboard/impulse/live-classification", icon: Radar, iconColor: "#0e7fe9be" },
  { label: "Model testing", href: "/dashboard/impulse/model-testing", icon: FlaskConical, iconColor: "#c413acbe", tourId: "nav-testing" },
  { label: "Post-processing", href: "/dashboard/impulse/post-processing", icon: SlidersHorizontal, iconColor: "#F97316" },
  { label: "Deployment", href: "/dashboard/impulse/deployment", icon: Package, iconColor: "#cc1111ff", tourId: "nav-deployment" },
  { label: "Jobs", href: "/dashboard/jobs", icon: ClipboardList, iconColor: "#f53302ff" },

];

const NAV_MENU = [
  {
    header: null,
    items: [
      { label: "Dashboard", href: "/dashboard", icon: LayoutDashboard, iconColor: "#3B82F6" },
      { label: "Devices", href: "/dashboard/devices", icon: WifiIcon, iconColor: "#06B6D4" },
      { label: "Data acquisition", href: "/dashboard/data", icon: Database, iconColor: "#F59E0B", tourId: "nav-data" },
      { label: "Data labeling", href: "/dashboard/data/dataset", icon: BoxesIcon, iconColor: "#8B5CF6", tourId: "nav-labeling" },
    ] as NavItemDef[]
  },
  {
    header: "Impulse design",
    items: [] as NavItemDef[],
  },
  {
    // Project-level, standalone surface — comes after the impulse design
    // section rather than inside it. Renders the same regardless of which
    // impulse is selected (docs/Action/parityfix.md §1.0, §7.1).
    header: null,
    items: [
      { label: "Versioning", href: "/dashboard/versions", icon: History, iconColor: "#10B981", tourId: "nav-versioning" },
    ] as NavItemDef[],
  },

];

function buildImpulseDesignItems(activeImpulse: any): NavItemDef[] {
  const dspItems: NavItemDef[] = (activeImpulse?.dsp_blocks ?? []).map((block: any) => ({
    label: block.name || block.type || "Processing block",
    href: `/dashboard/impulse/${block.type}/parameters?impulseId=${activeImpulse.id}`,
    isDynamic: true,
    dspBlock: true,
    icon: AlbumIcon,
    iconColor: "#38BDF8",
  }));
  const mlItems: NavItemDef[] = (activeImpulse?.ml_blocks ?? []).map((block: any) => ({
    label: block.name || block.type || "Learning block",
    href: `/dashboard/impulse/training?impulseId=${activeImpulse.id}`,
    isDynamic: true,
    dspBlock: false,
    icon: LocateIcon,
    iconColor: "#EC4899",
  }));
  return [...STATIC_IMPULSE_ITEMS_BEFORE, ...dspItems, ...mlItems, ...STATIC_IMPULSE_ITEMS_AFTER];
}

export default function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, logout, activeProject, activeImpulse, savedActiveImpulse, setActiveImpulse, setSavedActiveImpulse } = useAppStore();
  const [impulses, setImpulses] = useState<any[]>([]);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const sidebarImpulse =
    savedActiveImpulse?.project_id === activeProject?.id
      ? savedActiveImpulse
      : activeImpulse;

  async function loadImpulses(projectId: string) {
    try {
      const { data } = await impulsesApi.list(projectId);
      // Mirror the Manage Impulses table: oldest first by `created_at`, with
      // id-compare as a stable tiebreaker for collisions or missing timestamps.
      const ts = (imp: any) => {
        const t = Date.parse(imp?.created_at ?? "");
        return Number.isFinite(t) ? t : 0;
      };
      const sorted = Array.isArray(data)
        ? [...data].sort((a: any, b: any) => {
          const diff = ts(a) - ts(b);
          if (diff !== 0) return diff;
          return String(a?.id ?? "").localeCompare(String(b?.id ?? ""));
        })
        : [];
      setImpulses(sorted);

      // Read the active/saved impulse fresh from the store rather than the
      // destructured render snapshot — `impulses:changed` events can race
      // with the latest setActiveImpulse call.
      const { activeImpulse: liveActive, savedActiveImpulse: liveSaved } = useAppStore.getState();
      const activeStillExists = !liveActive || sorted.some((imp: any) => imp.id === liveActive.id);
      const savedStillExists = !liveSaved || sorted.some((imp: any) => imp.id === liveSaved.id);
      if (!activeStillExists) setActiveImpulse(null);
      if (!savedStillExists) setSavedActiveImpulse(null);
    } catch {
      // Keep the existing list if refresh fails.
    }
  }

  useEffect(() => {
    if (!activeProject) {
      setImpulses([]);
      return;
    }
    void loadImpulses(activeProject.id);
  }, [activeProject?.id]);

  useEffect(() => {
    if (!activeProject || !dropdownOpen) return;
    void loadImpulses(activeProject.id);
  }, [dropdownOpen, activeProject?.id]);

  useEffect(() => {
    function handleImpulsesChanged() {
      if (!activeProject) return;
      void loadImpulses(activeProject.id);
    }

    window.addEventListener("impulses:changed", handleImpulsesChanged);
    return () => window.removeEventListener("impulses:changed", handleImpulsesChanged);
  }, [activeProject?.id, activeImpulse?.id, savedActiveImpulse?.id]);

  // Close dropdown on any outside click
  useEffect(() => {
    if (!dropdownOpen) return;
    const close = () => setDropdownOpen(false);
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, [dropdownOpen]);

  async function handleCreateImpulse() {
    if (!activeProject) {
      toast.error("No active project selected");
      return;
    }
    // Persist immediately — the impulse lands in the backend list before the
    // user even sees the builder. Empty `dsp_blocks` / `ml_blocks` mean it
    // shows as an empty pipeline; the user fills it in over time and clicks
    // Save Impulse to commit changes. Name is server-assigned via the
    // monotonic per-project counter so display matches what's persisted.
    try {
      const { data } = await impulsesApi.create({
        project_id: activeProject.id,
        window_size_ms: 1000,
        window_increase_ms: 500,
        frequency_hz: 100,
        input_type: "image",
        input_axes: ["image"],
        sensor_type: "camera",
        image_width: 96,
        image_height: 96,
        resize_mode: "Fit shortest axis",
        train_subset_percent: 100,
        dsp_blocks: [],
        ml_blocks: [],
      });
      setActiveImpulse(data);
      setSavedActiveImpulse(data);
      setDropdownOpen(false);
      window.dispatchEvent(new Event("impulses:changed"));
      router.push(`/dashboard/impulse?impulseId=${data.id}`);
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || "Failed to create impulse");
    }
  }

  return (
    <aside className="shell-sidebar app-sidebar w-72 flex-shrink-0 flex flex-col">
      {/* Logo */}
      <div className="sidebar-brand h-16 flex items-center px-5 border-b">
        <PetalEdgeLogo variant="sidebar" size={50} />
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav flex-1 px-5 py-4 space-y-4 overflow-y-auto">
        {NAV_MENU.map((section, idx) => {
          const isImpulseSection = section.header === "Impulse design";
          return (
            <div key={idx} className="space-y-1">
              {section.header === "Impulse design" ? (
                /* ── Impulse switcher ─────────────────────────────────────── */
                <div className="relative mb-5">
                  <button
                    onClick={(e) => { e.stopPropagation(); setDropdownOpen(v => !v); }}
                    className="sidebar-impulse-toggle w-full flex items-center justify-between px-3 py-2 rounded-lg hover:bg-gray-800 transition-colors"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <Activity size={14} className="text-brand-400 flex-shrink-0" />
                      <span className="text-xs font-bold text-brand-400 uppercase tracking-[0.12em] truncate">
                        {activeImpulse?.name || "No impulse"}
                      </span>
                    </div>
                    {dropdownOpen
                      ? <ChevronUp size={12} className="text-gray-500 flex-shrink-0" />
                      : <ChevronDown size={12} className="text-gray-500 flex-shrink-0" />}
                  </button>

                  {dropdownOpen && (
                    <div
                      className="sidebar-impulse-dropdown dropdown-panel absolute left-0 top-full mt-1 z-50 w-full rounded-xl overflow-hidden"
                      onClick={e => e.stopPropagation()}
                    >
                      {/* Impulse list */}
                      <div className="py-1">
                        {impulses.length === 0 ? (
                          <p className="px-4 py-2.5 text-xs text-gray-500">No impulses yet</p>
                        ) : (
                          impulses.map(imp => (
                            <button
                              key={imp.id}
                              onClick={() => {
                                setActiveImpulse(imp);
                                setSavedActiveImpulse(imp);
                                setDropdownOpen(false);
                                // Always land on the impulse overview page with
                                // the selected id in the URL — that page reads
                                // `impulseId` from searchParams as the source of
                                // truth, so passing it explicitly guarantees the
                                // clicked impulse loads regardless of stale state.
                                router.push(`/dashboard/impulse?impulseId=${imp.id}`);
                              }}
                              className={`sidebar-impulse-dropdown-item w-full text-left px-4 py-2.5 text-sm transition-colors hover:bg-gray-700/40 ${activeImpulse?.id === imp.id
                                ? "font-bold text-gray-100"
                                : "font-normal text-gray-300"
                                }`}
                            >
                              {imp.name}
                            </button>
                          ))
                        )}
                      </div>

                      {/* Footer actions */}
                      <div className="border-t border-gray-700/60 py-1">
                        <button
                          onClick={() => { setDropdownOpen(false); router.push("/dashboard/impulse/manage"); }}
                          className="sidebar-impulse-dropdown-action w-full text-left px-4 py-2.5 text-sm text-gray-400 hover:text-gray-200 hover:bg-gray-700/40 transition-colors"
                        >
                          Manage impulses
                        </button>
                        <button
                          onClick={handleCreateImpulse}
                          className="sidebar-impulse-dropdown-action w-full text-left px-4 py-2.5 text-sm text-gray-400 hover:text-gray-200 hover:bg-gray-700/40 transition-colors"
                        >
                          Create new impulse
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ) : section.header ? (
                <h3 className="sidebar-section-label px-3 text-xs font-bold text-brand-400 uppercase tracking-[0.12em] mb-2 mt-2">
                  {section.header}
                </h3>
              ) : null}

              <div className={isImpulseSection ? "space-y-2 relative sidebar-impulse-group" : "space-y-1.5"}>
                {(isImpulseSection
                  ? buildImpulseDesignItems(sidebarImpulse)
                  : section.items
                ).map((item) => {
                  const { label, href } = item;
                  const Icon = item.icon;
                  const baseHref = href.split("?")[0];
                  const requiresExactMatch =
                    baseHref === "/dashboard" ||
                    baseHref === "/dashboard/data" ||
                    baseHref === "/dashboard/impulse";

                  const reallyActive = !!pathname && (
                    pathname === baseHref ||
                    (!requiresExactMatch && pathname.startsWith(baseHref))
                  );

                  const impulseClass = isImpulseSection ? "sidebar-impulse-item" : "";
                  const iconColor = item.iconColor || "#818CF8";

                  return (
                    /* Real <a href> via next/link: navigates natively even
                       before React hydrates and gets automatic prefetching, so
                       the first click always works without a hard refresh.
                       Query-string hrefs (impulse items with ?impulseId=) are
                       preserved as-is; active state still keys off baseHref. */
                    <Link
                      key={href}
                      href={href}
                      data-tour={item.tourId}
                      aria-current={reallyActive ? "page" : undefined}
                      onClick={
                        href === "/dashboard/data/dataset"
                          ? // The dataset page's AI-Labeling sub-view is local
                            // React state, not URL-driven. When we're already
                            // sitting on this route, Next treats a click on an
                            // identical-URL Link as a no-op (no remount, no
                            // effect re-run), so nothing resets the sub-view
                            // and the click looks dead. Broadcast instead so
                            // the page can reset itself even without a route
                            // transition — mirrors the `impulses:changed`
                            // event pattern used elsewhere in this file.
                            () => window.dispatchEvent(new Event("data-labeling:go-home"))
                          : undefined
                      }
                      style={{ ["--item-glow" as any]: iconColor }}
                      className={[
                        reallyActive ? "nav-item-active" : "nav-item",
                        impulseClass,
                        "w-full flex items-center gap-3 px-3 py-2 text-sm",
                      ].filter(Boolean).join(" ")}
                    >
                      {Icon && (
                        <span className="nav-icon-wrap">
                          <Icon
                            size={22}
                            strokeWidth={2}
                            className="nav-icon"
                          />
                        </span>
                      )}
                      <span className={`sidebar-item-label ${isImpulseSection ? "sidebar-item-label-impulse" : ""}`}>
                        {label}
                      </span>
                    </Link>
                  );
                })}
              </div>
            </div>
          );
        })}
      </nav>
      <div className="border-t border-gray-800 my-4" />

    </aside>
  );
}
