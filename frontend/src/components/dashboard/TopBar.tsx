"use client";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useRouter, usePathname } from "next/navigation";
import { useAppStore, type TargetDevice } from "@/store/appStore";
import { projectsApi } from "@/utils/api";
import { Sun, Moon, Plus, Settings, LogOut, Network, GraduationCap } from "lucide-react";
import { useOnboarding } from "@/hooks/useOnboarding";
import TargetDeviceChip from "./TargetDeviceChip";

// Mirrors what GET /projects/ returns. The target-device fields are carried
// through so switching projects from this menu sets an activeProject the topbar
// chip can render immediately, rather than one missing its selection.
type ProjectSummary = {
  id: string;
  name: string;
  target_device_slug?: string | null;
  target_device?: TargetDevice | null;
};

/**
 * Single-letter initial for the compact avatar button. Falls back through
 * display name → email local-part → "U" so the circle is never blank.
 */
function avatarInitial(name?: string | null, email?: string | null): string {
  const source = (name || (email ? email.split("@")[0] : "") || "U").trim();
  return (source[0] || "U").toUpperCase();
}

const MAX_PROJECTS_IN_DROPDOWN = 5;

export default function TopBar() {
  const { activeProject, theme, setTheme, user, profileExtras, setActiveProject, logout } =
    useAppStore();
  const { restart: restartTutorial } = useOnboarding();
  const router = useRouter();
  const pathname = usePathname();
  // /dashboard surfaces the project list inline; /dashboard/projects is the
  // global projects directory — in both cases the centered topbar label would
  // be either redundant (dashboard) or actively misleading (projects: the
  // user isn't looking at a specific project).
  const showProjectName =
    pathname !== "/dashboard" && pathname !== "/dashboard/projects";

  const [menuOpen, setMenuOpen] = useState(false);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [mounted, setMounted] = useState(false);
  // Anchor coordinates for the portaled dropdown. The dropdown lives on
  // document.body (so shell-app's overflow:hidden can't clip it), so we have
  // to compute its position from the trigger button's bounding rect.
  const [menuAnchor, setMenuAnchor] = useState<{ top: number; right: number } | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Lazy-load projects when the menu first opens so we don't refetch on every
  // re-render. Refetch each open so newly created projects show up.
  useEffect(() => {
    if (!menuOpen) return;
    let cancelled = false;
    projectsApi
      .list()
      .then(({ data }) => {
        if (cancelled) return;
        const list: ProjectSummary[] = Array.isArray(data) ? data : [];
        setProjects(list);
      })
      .catch(() => {
        if (!cancelled) setProjects([]);
      });
    return () => {
      cancelled = true;
    };
  }, [menuOpen]);

  useEffect(() => {
    if (!menuOpen) return;
    function onDocClick(e: MouseEvent) {
      const target = e.target as Node;
      // Trigger AND menu are both valid click targets; only close when the
      // click was outside both. Since the menu is portaled to <body> it is
      // not inside the trigger's DOM subtree, so we have to check separately.
      if (triggerRef.current?.contains(target)) return;
      if (menuRef.current?.contains(target)) return;
      setMenuOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setMenuOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [menuOpen]);

  // Track the trigger button's screen position so the portaled dropdown stays
  // glued to it during scroll / window resize / topbar layout shifts.
  useLayoutEffect(() => {
    if (!menuOpen) {
      setMenuAnchor(null);
      return;
    }
    function update() {
      const el = triggerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      setMenuAnchor({
        top: rect.bottom + 8,
        right: window.innerWidth - rect.right,
      });
    }
    update();
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [menuOpen]);

  const savedDisplayName =
    (user?.id && profileExtras[user.id]?.name?.trim()) || "";
  const displayName =
    savedDisplayName ||
    user?.username ||
    (user?.email ? user.email.split("@")[0] : "User");
  const initial = avatarInitial(savedDisplayName || user?.username, user?.email);
  const roleLabel = user?.role ? user.role : "";
  const projectAriaLabel = roleLabel
    ? `Active project: ${activeProject?.name} — ${roleLabel}`
    : `Active project: ${activeProject?.name}`;
  const projectTitle = roleLabel
    ? `${activeProject?.name} · ${roleLabel}`
    : activeProject?.name || "";

  function go(path: string) {
    setMenuOpen(false);
    router.push(path);
  }

  function selectProject(p: ProjectSummary) {
    setMenuOpen(false);
    setActiveProject(p as any);
    router.push("/dashboard");
  }

  function handleLogout() {
    setMenuOpen(false);
    logout();
    router.push("/login");
  }

  function handleRestartTutorial() {
    setMenuOpen(false);
    restartTutorial();
    router.push("/dashboard");
  }

  const visibleProjects = projects.slice(0, MAX_PROJECTS_IN_DROPDOWN);
  const hasMore = projects.length > MAX_PROJECTS_IN_DROPDOWN;

  return (
    <header className="shell-topbar app-topbar pe-topbar h-16 flex items-center px-5 gap-4 flex-shrink-0 relative">
      <div className="flex-1" />

      {showProjectName && activeProject?.name && (
        <div
          className="pe-project-switcher"
          aria-label={projectAriaLabel}
          title={projectTitle}
        >
          <span className="pe-project-switcher-name">{activeProject.name}</span>
          {roleLabel && (
            <span className="pe-project-switcher-role">{roleLabel}</span>
          )}
        </div>
      )}

      <div className="flex items-center gap-7 pr-1">
        {/* Sits left of the theme toggle. Renders nothing when no project is
            active, so /dashboard and /dashboard/projects are unaffected. */}
        <TargetDeviceChip />

        <button
          type="button"
          className="data-acq-icon-btn"
          onClick={() => setTheme(theme === "light" ? "dark" : "light")}
          aria-label={theme === "light" ? "Switch to dark theme" : "Switch to light theme"}
          title={theme === "light" ? "Switch to dark theme" : "Switch to light theme"}
          aria-pressed={theme === "dark"}
        >
          {theme === "light" ? <Moon size={16} /> : <Sun size={16} />}
        </button>

        {/* Avatar dropdown trigger. Replaces the previous Link → settings so the
            user can jump between projects, create a new one, or sign out from
            anywhere in the dashboard. The menu itself is portaled to <body>
            below so shell-app's overflow:hidden can't clip it. */}
        <button
          ref={triggerRef}
          type="button"
          onClick={() => setMenuOpen((v) => !v)}
          className="pe-topbar-avatar-btn"
          aria-haspopup="menu"
          aria-expanded={menuOpen}
          aria-label={`Account menu — signed in as ${displayName}`}
          title={displayName}
        >
          <span className="pe-topbar-avatar-circle" aria-hidden="true">
            {initial}
          </span>
          <span className="pe-topbar-avatar-status" aria-hidden="true" />
        </button>
      </div>

      {mounted && menuOpen && menuAnchor && createPortal(
        <div
          ref={menuRef}
          role="menu"
          className="pe-avatar-menu dropdown-panel rounded-xl"
          style={{
            position: "fixed",
            top: menuAnchor.top,
            right: menuAnchor.right,
          }}
        >
          <div className="pe-avatar-menu-section">
            <div className="pe-avatar-menu-eyebrow">Welcome!</div>
            <button role="menuitem" className="pe-avatar-menu-item" onClick={() => go("/dashboard/projects")}>
              <Network size={16} aria-hidden="true" />
              <span>Projects</span>
            </button>
          </div>

          <div className="pe-avatar-menu-divider" />

          <div className="pe-avatar-menu-section">
            <button role="menuitem" className="pe-avatar-menu-item" onClick={() => go("/dashboard/projects?new=1")}>
              <Plus size={16} aria-hidden="true" />
              <span>Create new project</span>
            </button>
          </div>

          {visibleProjects.length > 0 && (
            <>
              <div className="pe-avatar-menu-divider" />
              <div className="pe-avatar-menu-section">
                <div className="pe-avatar-menu-eyebrow">Projects</div>
                {visibleProjects.map((p) => (
                  <button
                    key={p.id}
                    role="menuitem"
                    className="pe-avatar-menu-item"
                    onClick={() => selectProject(p)}
                    title={p.name}
                  >
                    <Network size={16} aria-hidden="true" />
                    <span className="truncate">{p.name}</span>
                  </button>
                ))}
                {hasMore && (
                  <button role="menuitem" className="pe-avatar-menu-link" onClick={() => go("/dashboard/projects")}>
                    More projects…
                  </button>
                )}
              </div>
            </>
          )}

          <div className="pe-avatar-menu-divider" />

          <div className="pe-avatar-menu-section">
            <button role="menuitem" className="pe-avatar-menu-item" onClick={() => go("/dashboard/settings")}>
              <Settings size={16} aria-hidden="true" />
              <span>Account settings</span>
            </button>
            <button role="menuitem" className="pe-avatar-menu-item" onClick={handleRestartTutorial}>
              <GraduationCap size={16} aria-hidden="true" />
              <span>Restart tutorial</span>
            </button>
            <button role="menuitem" className="pe-avatar-menu-item" onClick={handleLogout}>
              <LogOut size={16} aria-hidden="true" />
              <span>Logout</span>
            </button>
          </div>
        </div>,
        document.body
      )}
    </header>
  );
}
