"use client";
import { useEffect, useMemo, useState } from "react";
import { useAppStore } from "@/store/appStore";
import { authApi } from "@/utils/api";
import {
  Key,
  Copy,
  Trash2,
  Plus,
  CheckCircle,
  User,
  Shield,
  Eye,
  EyeOff,
  Lock,
  UserMinus,
} from "lucide-react";
import toast from "react-hot-toast";

type SectionKey = "account" | "password" | "api-keys" | "security";

const NAV_ITEMS: { key: SectionKey; label: string; icon: any }[] = [
  { key: "account", label: "Account", icon: User },
  { key: "password", label: "Password", icon: Lock },
  { key: "api-keys", label: "API Keys", icon: Key },
  { key: "security", label: "Security", icon: Shield },
];

function avatarInitials(name?: string | null, email?: string | null): string {
  const source = (name || (email ? email.split("@")[0] : "") || "U").trim();
  if (!source) return "U";
  const parts = source.split(/[\s._-]+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return source.slice(0, 2).toUpperCase();
}

export default function SettingsPage() {
  const { user, profileExtras, setProfileExtras } = useAppStore();

  const [active, setActive] = useState<SectionKey>("account");
  // null = still loading; false = Google-only (no local password); true = has password
  const [hasPassword, setHasPassword] = useState<boolean | null>(null);

  useEffect(() => {
    authApi.me()
      .then(({ data }) => setHasPassword(data.has_password ?? true))
      .catch(() => setHasPassword(true)); // safe fallback: show the tab
  }, []);

  // If the tab becomes invisible while selected, fall back to account.
  useEffect(() => {
    if (hasPassword === false && active === "password") setActive("account");
  }, [hasPassword, active]);

  const visibleNav = hasPassword === false
    ? NAV_ITEMS.filter((i) => i.key !== "password")
    : NAV_ITEMS;

  return (
    <div className="pe-settings-page">
      {/* ── Left: settings sub-nav ─────────────────────────────────────── */}
      <aside className="pe-settings-nav-col">
        <h1 className="pe-settings-nav-title">Account settings</h1>
        <nav className="pe-settings-nav" aria-label="Account settings sections">
          {visibleNav.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              type="button"
              onClick={() => setActive(key)}
              aria-current={active === key ? "page" : undefined}
              className={`pe-settings-nav-item ${active === key ? "is-active" : ""}`}
            >
              <Icon size={15} className="pe-settings-nav-icon" />
              <span>{label}</span>
            </button>
          ))}
        </nav>
      </aside>

      {/* ── Right: section content ─────────────────────────────────────── */}
      <main className="pe-settings-content">
        {active === "account" && (
          <AccountSection
            user={user}
            profileExtras={profileExtras}
            setProfileExtras={setProfileExtras}
          />
        )}
        {active === "password" && hasPassword !== false && <PasswordSection />}
        {active === "api-keys" && <ApiKeysSection />}
        {active === "security" && <SecuritySection />}
      </main>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Account section
// ─────────────────────────────────────────────────────────────────────────────

function AccountSection({
  user,
  profileExtras,
  setProfileExtras,
}: {
  user: any;
  profileExtras: Record<string, { name?: string; jobTitle?: string; companyName?: string }>;
  setProfileExtras: (id: string, extras: any) => void;
}) {
  const userId = user?.id ?? "";
  // Own selector so this section is self-contained; deleteAccount() calls it.
  const logout = useAppStore((s) => s.logout);
  const savedExtras = useMemo(
    () => profileExtras[userId] || {},
    [profileExtras, userId],
  );

  const [name, setName] = useState<string>(savedExtras.name ?? user?.username ?? "");
  const [jobTitle, setJobTitle] = useState<string>(savedExtras.jobTitle ?? "");
  const [companyName, setCompanyName] = useState<string>(savedExtras.companyName ?? "");
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    setName(savedExtras.name ?? user?.username ?? "");
    setJobTitle(savedExtras.jobTitle ?? "");
    setCompanyName(savedExtras.companyName ?? "");
  }, [user?.id, savedExtras.name, savedExtras.jobTitle, savedExtras.companyName, user?.username]);

  async function save(e?: React.FormEvent) {
    e?.preventDefault();
    if (!userId) return;
    setSaving(true);
    try {
      // No backend `PATCH /auth/me` yet — persist locally so the top bar
      // (which reads from profileExtras) picks the new name up immediately.
      // Drop in the API call here once the endpoint ships.
      setProfileExtras(userId, {
        name: name.trim() || undefined,
        jobTitle: jobTitle.trim() || undefined,
        companyName: companyName.trim() || undefined,
      });
      toast.success("Account details saved");
    } finally {
      setSaving(false);
    }
  }

  async function deleteAccount() {
    const confirmed = window.confirm(
      "Delete your account?\n\nThis cannot be undone. All your projects, datasets, and trained models will be removed permanently.",
    );
    if (!confirmed) return;
    setDeleting(true);
    try {
      await authApi.deleteAccount();
    } catch (err: any) {
      // 401 = the account row is already gone (e.g. a retried click after a
      // successful delete). Treat it as success; anything else is a real failure.
      if (err?.response?.status !== 401) {
        toast.error("Failed to delete your account. Please try again.");
        setDeleting(false);
        return;
      }
    }
    // Success (204) or already-deleted (401): tear down and leave via a HARD
    // redirect so the dashboard layout's token-guard (layout.tsx:17-23) can't
    // race/swallow a soft router.push, and the disabled button can't fire a
    // second DELETE. Intentionally do NOT reset `deleting`.
    logout();
    window.location.replace("/login");
  }

  const initials = avatarInitials(name || user?.username, user?.email);

  return (
    <div className="space-y-6">
      {/* Account card */}
      <section className="pe-account-card">
        <header className="pe-account-card-head">
          <User size={16} className="pe-account-card-head-icon" />
          <h2 className="pe-account-card-title">Account</h2>
        </header>

        <form onSubmit={save} className="pe-account-body" noValidate>
          <div className="pe-account-photo-col">
            <span className="pe-account-photo-label">Profile photo</span>
            <div className="pe-account-avatar" aria-hidden="true">
              {initials.slice(0, 1)}
            </div>
          </div>

          <div className="pe-account-fields">
            <div className="pe-account-field">
              <label className="pe-account-label" htmlFor="acct-name">Name</label>
              <input
                id="acct-name"
                type="text"
                className="pe-account-input"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Your Name"
                autoComplete="name"
              />
            </div>

            <div className="pe-account-field">
              <label className="pe-account-label" htmlFor="acct-email">Email</label>
              <input
                id="acct-email"
                type="email"
                className="pe-account-input pe-account-input--readonly"
                value={user?.email ?? ""}
                readOnly
                aria-readonly="true"
                title="Contact support to change your email"
              />
            </div>

            <div className="pe-account-field">
              <label className="pe-account-label" htmlFor="acct-job">Job title</label>
              <input
                id="acct-job"
                type="text"
                className="pe-account-input"
                value={jobTitle}
                onChange={(e) => setJobTitle(e.target.value)}
                placeholder="e.g. ML Engineer"
              />
            </div>

            <div className="pe-account-field">
              <label className="pe-account-label" htmlFor="acct-company">Company name</label>
              <input
                id="acct-company"
                type="text"
                className="pe-account-input"
                value={companyName}
                onChange={(e) => setCompanyName(e.target.value)}
                placeholder="Company name"
              />
            </div>
          </div>
        </form>

        <footer className="pe-account-card-foot">
          <button
            type="button"
            onClick={() => save()}
            disabled={saving || !userId}
            className="pe-account-btn pe-account-btn--primary"
          >
            {saving ? "Saving…" : "Save changes"}
          </button>
        </footer>
      </section>

      {/* Administrative zone */}
      <section className="pe-account-card">
        <header className="pe-account-card-head">
          <h2 className="pe-account-card-title">Administrative zone</h2>
        </header>
        <div className="pe-account-admin-body">
          <button
            type="button"
            onClick={deleteAccount}
            disabled={deleting}
            className="pe-account-btn pe-account-btn--danger"
          >
            <UserMinus size={16} />
            {deleting ? "Working…" : "Delete your account"}
          </button>
          <p className="pe-account-admin-note">
            Deleting your account removes all projects, datasets, and trained models
            you own. This action cannot be undone.
          </p>
        </div>
      </section>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Password section
// ─────────────────────────────────────────────────────────────────────────────

function PasswordSection() {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [showNext, setShowNext] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function validate(): string | null {
    if (!current) return "Enter your current password.";
    if (next.length < 8) return "New password must be at least 8 characters.";
    if (next === current) return "New password must differ from the current one.";
    if (next !== confirm) return "Confirmation does not match the new password.";
    return null;
  }

  async function save(e: React.FormEvent) {
    e.preventDefault();
    const err = validate();
    if (err) {
      setError(err);
      return;
    }
    setError(null);
    setSaving(true);
    try {
      await authApi.changePassword({ current_password: current, new_password: next });
      toast.success("Password updated successfully.");
      setCurrent("");
      setNext("");
      setConfirm("");
    } catch (e: any) {
      const detail = e?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Failed to update password. Please try again.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="pe-account-card">
      <header className="pe-account-card-head">
        <Lock size={16} className="pe-account-card-head-icon" />
        <h2 className="pe-account-card-title">Password</h2>
      </header>

      <form onSubmit={save} className="pe-account-admin-body" style={{ width: "100%", alignItems: "stretch" }} noValidate>
        <div className="pe-account-fields" style={{ maxWidth: 480 }}>
          <div className="pe-account-field">
            <label className="pe-account-label" htmlFor="pw-current">Current password</label>
            <input
              id="pw-current"
              type="password"
              autoComplete="current-password"
              className="pe-account-input"
              value={current}
              onChange={(e) => setCurrent(e.target.value)}
              placeholder="••••••••"
              required
            />
          </div>

          <div className="pe-account-field">
            <label className="pe-account-label" htmlFor="pw-new">New password</label>
            <div className="pe-pw-input-wrap">
              <input
                id="pw-new"
                type={showNext ? "text" : "password"}
                autoComplete="new-password"
                className="pe-account-input"
                value={next}
                onChange={(e) => setNext(e.target.value)}
                placeholder="At least 8 characters"
                minLength={8}
                required
              />
              <button
                type="button"
                className="pe-pw-eye"
                onClick={() => setShowNext((v) => !v)}
                aria-pressed={showNext}
                aria-label={showNext ? "Hide password" : "Show password"}
              >
                {showNext ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>

          <div className="pe-account-field">
            <label className="pe-account-label" htmlFor="pw-confirm">Confirm new password</label>
            <input
              id="pw-confirm"
              type={showNext ? "text" : "password"}
              autoComplete="new-password"
              className="pe-account-input"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              placeholder="Repeat the new password"
              required
            />
          </div>

          {error && (
            <div role="alert" className="pe-pw-error">
              {error}
            </div>
          )}
        </div>

        <div style={{ marginTop: "0.4rem" }}>
          <button
            type="submit"
            disabled={saving}
            className="pe-account-btn pe-account-btn--primary"
          >
            {saving ? "Saving…" : "Update password"}
          </button>
        </div>
      </form>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// API keys section
// ─────────────────────────────────────────────────────────────────────────────

function ApiKeysSection() {
  const [apiKeys, setApiKeys] = useState<any[]>([]);
  const [newKeyName, setNewKeyName] = useState("");
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [showKey, setShowKey] = useState(false);
  const [loading, setLoading] = useState(false);
  const [copiedKey, setCopiedKey] = useState(false);

  useEffect(() => {
    void load();
  }, []);

  async function load() {
    try {
      const { data } = await authApi.apiKeys();
      setApiKeys(data);
    } catch { }
  }

  async function create() {
    if (!newKeyName.trim()) return;
    setLoading(true);
    try {
      const { data } = await authApi.createKey(newKeyName);
      setCreatedKey(data.key);
      setApiKeys((k) => [...k, data]);
      setNewKeyName("");
      toast.success("API key created — copy it now, it won't be shown again");
    } catch {
      toast.error("Failed to create key");
    } finally {
      setLoading(false);
    }
  }

  async function revoke(id: string) {
    if (!confirm("Revoke this API key? This cannot be undone.")) return;
    try {
      await authApi.revokeKey(id);
      setApiKeys((k) => k.filter((x) => x.id !== id));
      toast.success("Key revoked");
    } catch {
      toast.error("Failed to revoke key");
    }
  }

  function copyKey(key: string) {
    navigator.clipboard.writeText(key);
    setCopiedKey(true);
    setTimeout(() => setCopiedKey(false), 2000);
    toast.success("Copied to clipboard");
  }

  return (
    <section className="pe-account-card">
      <header className="pe-account-card-head">
        <Key size={16} className="pe-account-card-head-icon" />
        <h2 className="pe-account-card-title">API Keys</h2>
      </header>

      <div className="pe-account-admin-body" style={{ width: "100%", alignItems: "stretch", gap: "1rem" }}>
        {createdKey && (
          <div className="pe-keys-callout">
            <div className="pe-keys-callout-head">
              <CheckCircle size={14} className="text-green-500" />
              <span>API key created — copy it now!</span>
            </div>
            <p className="pe-keys-callout-note">
              This is the only time this key will be shown in full.
            </p>
            <div className="pe-keys-callout-box">
              <code className="pe-keys-callout-code">
                {showKey ? createdKey : createdKey.slice(0, 10) + "•".repeat(40)}
              </code>
              <button type="button" onClick={() => setShowKey(!showKey)} className="pe-keys-callout-icon-btn" aria-label={showKey ? "Hide key" : "Show key"}>
                {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
              <button type="button" onClick={() => copyKey(createdKey)} className="pe-keys-callout-icon-btn" aria-label="Copy key">
                {copiedKey ? <CheckCircle size={14} className="text-green-500" /> : <Copy size={14} />}
              </button>
            </div>
            <button
              type="button"
              onClick={() => setCreatedKey(null)}
              className="pe-keys-callout-dismiss"
            >
              I&apos;ve saved the key, dismiss this
            </button>
          </div>
        )}

        <div className="pe-keys-create-row">
          <input
            className="pe-account-input"
            placeholder="Key name (e.g. ESP32 Lab Device)"
            value={newKeyName}
            onChange={(e) => setNewKeyName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && create()}
          />
          <button
            type="button"
            onClick={create}
            disabled={loading || !newKeyName.trim()}
            className="pe-account-btn pe-account-btn--primary"
          >
            <Plus size={14} /> Create
          </button>
        </div>

        {apiKeys.length === 0 ? (
          <p className="pe-keys-empty">No API keys yet</p>
        ) : (
          <div className="pe-keys-list">
            {apiKeys.map((k) => (
              <div key={k.id} className="pe-keys-row">
                <div className="min-w-0">
                  <p className="pe-keys-row-name">{k.name}</p>
                  <p className="pe-keys-row-meta">
                    Created {new Date(k.created_at).toLocaleDateString()}
                    {k.last_used && ` · Last used ${new Date(k.last_used).toLocaleDateString()}`}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => revoke(k.id)}
                  className="pe-keys-row-revoke"
                  aria-label={`Revoke key ${k.name}`}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Security section
// ─────────────────────────────────────────────────────────────────────────────

function SecuritySection() {
  return (
    <section className="pe-account-card">
      <header className="pe-account-card-head">
        <Shield size={16} className="pe-account-card-head-icon" />
        <h2 className="pe-account-card-title">Security</h2>
      </header>
      <div className="pe-account-admin-body" style={{ width: "100%", alignItems: "stretch", gap: "0.6rem" }}>
        <div className="pe-security-row">
          <div className="min-w-0">
            <p className="pe-security-row-title">JWT token expiry</p>
            <p className="pe-security-row-meta">Tokens expire after 24 hours</p>
          </div>
          <span className="badge-green">Active</span>
        </div>
        <div className="pe-security-row">
          <div className="min-w-0">
            <p className="pe-security-row-title">Password hashing</p>
            <p className="pe-security-row-meta">bcrypt with salt rounds</p>
          </div>
          <span className="badge-green">bcrypt</span>
        </div>
      </div>
    </section>
  );
}
