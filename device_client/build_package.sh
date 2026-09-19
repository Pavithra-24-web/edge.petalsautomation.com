#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Petal Edge — device-client release builder (contract §14.1 / §14.3)
#
# Stamps VERSION, tars the six runtime files FLAT (no wrapper directory) into
# petal-device-<version>.tar.gz, and publishes it via the admin upload endpoint.
# The server computes the sha256 and size — this script never sends a checksum.
#
#   ./build_package.sh v1.0.0 \
#       --host https://app.petaledge.ai \
#       --token "$PETAL_ADMIN_JWT"
#
# Or build-only (skip upload), e.g. for local inspection / CI artifact:
#
#   ./build_package.sh v1.0.0 --no-upload --out ./dist
#
# Options:
#   --host   <url>    backend base URL for upload (or PETAL_BACKEND_HOST)
#   --token  <jwt>    admin bearer token for upload (or PETAL_ADMIN_JWT)
#   --out    <dir>    output directory for the tarball (default: mktemp -d)
#   --no-upload       build the tarball but do not POST it
#
# Fails loudly on any step; never silently continues.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)

log()  { printf '\033[1;34m[build]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ ok  ]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[FAIL ]\033[0m %s\n' "$*" >&2; exit 1; }

# The six files that make up a release bundle (contract §14.1). VERSION is
# generated here; the other five are the existing device_client sources.
RUNTIME_FILES="client.py forwarder.py provision.py requirements.txt petal-device.service"

# ── Args ─────────────────────────────────────────────────────────────────────
VERSION="${1:-}"
[ -n "$VERSION" ] || die "usage: build_package.sh <vMAJOR.MINOR.PATCH> [--host URL --token JWT | --no-upload]"
shift || true

case "$VERSION" in
    v[0-9]*.[0-9]*.[0-9]*) : ;;
    *) die "version must look like vMAJOR.MINOR.PATCH (leading 'v' required), got: $VERSION" ;;
esac

HOST="${PETAL_BACKEND_HOST:-}"
TOKEN="${PETAL_ADMIN_JWT:-}"
OUT_DIR=""
DO_UPLOAD=1

while [ $# -gt 0 ]; do
    case "$1" in
        --host)      HOST="${2:?--host needs a value}"; shift 2 ;;
        --token)     TOKEN="${2:?--token needs a value}"; shift 2 ;;
        --out)       OUT_DIR="${2:?--out needs a value}"; shift 2 ;;
        --no-upload) DO_UPLOAD=0; shift ;;
        *) die "unknown argument: $1" ;;
    esac
done

# ── Preconditions ────────────────────────────────────────────────────────────
command -v tar  >/dev/null 2>&1 || die "tar not found"
for f in $RUNTIME_FILES; do
    [ -f "${SCRIPT_DIR}/${f}" ] || die "missing runtime file: ${SCRIPT_DIR}/${f}"
done

# ── 1. Stamp VERSION (single line) ───────────────────────────────────────────
printf '%s\n' "$VERSION" > "${SCRIPT_DIR}/VERSION"
ok "stamped VERSION = ${VERSION}"

# ── 2. Tar the six files FLAT (no wrapper dir, no --strip-components needed) ──
if [ -z "$OUT_DIR" ]; then
    OUT_DIR="$(mktemp -d)"
fi
mkdir -p "$OUT_DIR"
TARBALL="${OUT_DIR}/petal-device-${VERSION}.tar.gz"

# -C SCRIPT_DIR so members are stored as bare filenames (flat layout, §14.1).
log "packing ${TARBALL}"
tar -czf "$TARBALL" -C "$SCRIPT_DIR" $RUNTIME_FILES VERSION \
    || die "tar failed"

# Sanity: members must be flat (no directory components).
if tar -tzf "$TARBALL" | grep -q '/'; then
    die "tarball is not flat — members contain a directory component"
fi
ok "built $(basename "$TARBALL") ($(wc -c < "$TARBALL") bytes)"
tar -tzf "$TARBALL" | sed 's/^/       member: /'

# ── 3. Upload via POST /api/v1/device-client/upload (server computes sha256) ──
if [ "$DO_UPLOAD" = "0" ]; then
    ok "build-only (--no-upload); tarball at ${TARBALL}"
    exit 0
fi

[ -n "$HOST" ]  || die "upload needs a backend host (--host or PETAL_BACKEND_HOST)"
[ -n "$TOKEN" ] || die "upload needs an admin token (--token or PETAL_ADMIN_JWT)"
command -v curl >/dev/null 2>&1 || die "curl not found (needed for upload)"

UPLOAD_URL="${HOST%/}/api/v1/device-client/upload"
log "uploading to ${UPLOAD_URL}"

# multipart: exactly `version` + `file`. No sha256 — the server is the sole
# checksum authority (contract §14.3).
HTTP_CODE="$(curl -sS -o /tmp/petal_upload_resp.$$ -w '%{http_code}' \
    -X POST "$UPLOAD_URL" \
    -H "Authorization: Bearer ${TOKEN}" \
    -F "version=${VERSION}" \
    -F "file=@${TARBALL};type=application/gzip")" \
    || die "upload request failed"

RESP="$(cat /tmp/petal_upload_resp.$$ 2>/dev/null || true)"
rm -f "/tmp/petal_upload_resp.$$"

case "$HTTP_CODE" in
    201|200) ok "published ${VERSION}"; printf '%s\n' "$RESP" ;;
    409)     die "version ${VERSION} already published (immutable) — bump the version" ;;
    401|403) die "upload rejected (${HTTP_CODE}) — admin token missing/invalid" ;;
    *)       die "upload failed (HTTP ${HTTP_CODE}): ${RESP}" ;;
esac
