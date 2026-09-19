// Edge-Impulse-style download filename builder for .pxe packages.
//
// Canonical pattern:  {project-name}-{impulse-name}-v{deploy-version}.pxe
//
// Each name component is slugified to lowercase ASCII letters, digits and
// single hyphens (any other character becomes a hyphen; runs collapse; leading
// and trailing hyphens strip). Must stay in sync with backend/app/ml/pxe_filename.py.

const NON_SLUG_CHAR_RE = /[^a-z0-9-]+/g;
const MULTI_HYPHEN_RE = /-{2,}/g;

export function slugify(value: string | null | undefined): string {
  if (value == null) return "";
  return String(value)
    .toLowerCase()
    .replace(NON_SLUG_CHAR_RE, "-")
    .replace(MULTI_HYPHEN_RE, "-")
    .replace(/^-+|-+$/g, "");
}

export function buildPxeDownloadFilename(
  projectName: string,
  impulseName: string,
  deployVersion: number,
): string {
  const projectSlug = slugify(projectName) || "project";
  const impulseSlug = slugify(impulseName) || "impulse";
  const version = Math.max(1, Math.floor(Number(deployVersion) || 1));
  return `${projectSlug}-${impulseSlug}-v${version}.pxe`;
}
