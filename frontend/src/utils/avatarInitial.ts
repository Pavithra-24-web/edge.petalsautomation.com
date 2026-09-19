/**
 * Returns a single uppercase letter to use as an avatar initial.
 * Derived from display name first, then email local-part, falling back to "U".
 */
export function avatarInitial(name?: string | null, email?: string | null): string {
  const source = (name || (email ? email.split("@")[0] : "") || "U").trim();
  return (source[0] || "U").toUpperCase();
}
