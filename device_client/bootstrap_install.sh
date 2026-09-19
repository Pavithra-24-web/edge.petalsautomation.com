#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Petal Edge — network bootstrap installer (contract §14.3 / §14.5)
#
# This is the script served by `GET /install.sh`. It is the zero-clone path:
#
#     curl -fsSL https://app.petaledge.ai/install.sh | bash
#
# Unlike the legacy device_client/install.sh (which copies a checkout), this
# FETCHES the published release over HTTP, verifies its sha256 against the
# manifest (the single source of truth — there is no .sha256 sidecar), then
# installs and starts the petal-device systemd service.
#
# The backend host is interpolated by the server when it serves this file (it
# substitutes the @@…@@ placeholder on the BACKEND_HOST line below). Running the
# raw file directly? Set PETAL_BACKEND_HOST to point it at your backend.
#
# Seven steps, in order (contract §14.5):
#   1. Detect Pi/Linux (arch + apt); fail loudly otherwise.
#   2. Fetch the latest manifest; read version, sha256, download_url.
#   3. Download the tarball to a temp file.
#   4. Verify sha256 vs the manifest → abort BEFORE touching /opt/petal on mismatch.
#   5. Extract flat into /opt/petal/device_client/.
#   6. Install deps + service (python3/venv/pip, petal user, venv, systemd unit).
#   7. Provision interactively as the petal user when no config.json exists.
#
# Fails loudly on any step; never silently continues.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Interpolated by the /install.sh route (it replaces the @@…@@ placeholder);
# env override wins for raw-file runs.
BACKEND_HOST="${PETAL_BACKEND_HOST:-@@PETAL_BACKEND_HOST@@}"

OPT_ROOT="/opt/petal"
VENV_DIR="${OPT_ROOT}/venv"
DEST="${OPT_ROOT}/device_client"
SERVICE_USER="petal"
UNIT_NAME="petal-device.service"
UNIT_DEST="/etc/systemd/system/${UNIT_NAME}"

# ── Output helpers ───────────────────────────────────────────────────────────
log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[  ok   ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[ warn  ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ FAIL  ]\033[0m %s\n' "$*" >&2; exit 1; }

# Guard: the @@…@@ placeholder must have been interpolated (served) or the host
# overridden via PETAL_BACKEND_HOST. A real host never contains '@@'.
case "$BACKEND_HOST" in
    *@@*)
        die "backend host not set — fetch this via the server (curl -fsSL <host>/install.sh | bash) or export PETAL_BACKEND_HOST" ;;
esac
BACKEND_HOST="${BACKEND_HOST%/}"
API_BASE="${BACKEND_HOST}/api/v1/device-client"
INSTALL_URL="${BACKEND_HOST}/install.sh"

# ── Privilege: this touches /opt, apt, useradd, systemd → needs root ─────────
# Make the documented `curl … | bash` work for a sudo-capable non-root user by
# re-running the served script under sudo (env carried through).
if [ "$(id -u)" != "0" ]; then
    if command -v sudo >/dev/null 2>&1; then
        log "re-running with sudo (root required for /opt/petal, apt, systemd)"
        exec sudo -E env "PETAL_BACKEND_HOST=${BACKEND_HOST}" bash -c "curl -fsSL '${INSTALL_URL}' | bash"
    fi
    die "must run as root and sudo is not available — re-run as root"
fi

command -v systemctl >/dev/null 2>&1 || SYSTEMCTL_MISSING=1
: "${SYSTEMCTL_MISSING:=0}"

# apt helper (reused from the legacy installer) — install a dpkg only if a probe
# command is missing, updating the apt cache at most once.
ensure_apt_pkg() {
    pkg="$1"; probe="$2"
    if eval "$probe" >/dev/null 2>&1; then
        return 0
    fi
    command -v apt-get >/dev/null 2>&1 \
        || die "$probe not available and apt-get is missing — install '$pkg' manually and re-run"
    if [ "${APT_UPDATED:-0}" != "1" ]; then
        log "apt-get update"
        apt-get update -y || die "apt-get update failed"
        APT_UPDATED=1
    fi
    log "installing $pkg"
    apt-get install -y "$pkg" || die "failed to install $pkg"
}

# ── 1. Detect Raspberry Pi / Linux (arch + apt) ──────────────────────────────
log "step 1/7 — detecting platform"
[ "$(uname -s)" = "Linux" ] \
    || die "unsupported OS '$(uname -s)' — the device-client installs on Debian / Raspberry Pi OS only"
ARCH="$(uname -m)"
case "$ARCH" in
    aarch64|arm64|armv7l|armv6l|x86_64|amd64) : ;;
    *) die "unsupported architecture '$ARCH' — expected arm/aarch64 (Raspberry Pi) or x86_64 (Debian)" ;;
esac
command -v apt-get >/dev/null 2>&1 \
    || die "apt-get not found — this installer targets Debian / Raspberry Pi OS"
ok "platform: Linux/${ARCH} with apt"

# curl + coreutils(sha256sum) are needed before the apt phase (steps 2 & 4).
ensure_apt_pkg curl      "command -v curl"
ensure_apt_pkg coreutils "command -v sha256sum"

# ── 2. Fetch the latest manifest ─────────────────────────────────────────────
log "step 2/7 — fetching latest manifest from ${API_BASE}/latest"
MANIFEST="$(curl -fsSL "${API_BASE}/latest")" \
    || die "could not fetch manifest from ${API_BASE}/latest"

# Minimal, dependency-free JSON field extraction (python3 isn't guaranteed yet).
_json_field() {
    printf '%s' "$MANIFEST" \
        | grep -o "\"$1\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" \
        | head -1 | sed 's/.*"\([^"]*\)"$/\1/' || true
}
VERSION="$(_json_field version)"
SHA256_EXPECTED="$(_json_field sha256)"
DOWNLOAD_URL="$(_json_field download_url)"
[ -n "$VERSION" ] && [ -n "$SHA256_EXPECTED" ] && [ -n "$DOWNLOAD_URL" ] \
    || die "manifest missing required fields (version/sha256/download_url)"
ok "latest version: ${VERSION}"

# ── 3. Download the tarball to a temp file ───────────────────────────────────
log "step 3/7 — downloading ${DOWNLOAD_URL}"
TMP_DL="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DL"; }
trap cleanup EXIT
TARBALL="${TMP_DL}/petal-device-${VERSION}.tar.gz"
curl -fsSL "$DOWNLOAD_URL" -o "$TARBALL" \
    || die "download failed: ${DOWNLOAD_URL}"
ok "downloaded $(wc -c < "$TARBALL") bytes"

# ── 4. Verify sha256 vs the manifest — abort BEFORE touching /opt/petal ──────
log "step 4/7 — verifying sha256"
SHA256_ACTUAL="$(sha256sum "$TARBALL" | awk '{print $1}')"
if [ "$SHA256_ACTUAL" != "$SHA256_EXPECTED" ]; then
    # /opt/petal has not been touched yet; the trap cleans up the temp file.
    die "checksum mismatch — expected ${SHA256_EXPECTED}, got ${SHA256_ACTUAL}. Aborting before touching ${OPT_ROOT}."
fi
ok "sha256 verified"

# ── 5. Extract flat into /opt/petal/device_client/ ───────────────────────────
# The tarball has flat members (§14.1) and never contains config.json, so an
# existing provisioned config is preserved untouched across re-installs.
log "step 5/7 — extracting ${VERSION} → ${DEST}"
mkdir -p "$DEST" || die "could not create ${DEST}"
tar -xzf "$TARBALL" -C "$DEST" || die "extraction failed"
[ -f "${DEST}/client.py" ] && [ -f "${DEST}/requirements.txt" ] \
    && [ -f "${DEST}/${UNIT_NAME}" ] \
    || die "extracted bundle is missing expected files"
ok "extracted device-client ${VERSION}"

# ── 6. Install deps + service ────────────────────────────────────────────────
log "step 6/7 — installing dependencies and service"

# 6a. Base packages: python3, venv, pip.
ensure_apt_pkg python3      "command -v python3"
ensure_apt_pkg python3-venv "python3 -c 'import venv, ensurepip'"
ensure_apt_pkg python3-pip  "command -v pip3 || python3 -m pip --version"
ok "python3: $(python3 --version 2>&1)"

# 6b. Service user.
if id "$SERVICE_USER" >/dev/null 2>&1; then
    ok "user '${SERVICE_USER}' already exists"
else
    log "creating system user '${SERVICE_USER}'"
    useradd --system --home-dir "$OPT_ROOT" --shell /usr/sbin/nologin "$SERVICE_USER" \
        || die "failed to create user '${SERVICE_USER}'"
    ok "created user '${SERVICE_USER}'"
fi

# 6c. Virtualenv + dependencies.
if [ -x "${VENV_DIR}/bin/python" ]; then
    ok "venv already present at ${VENV_DIR}"
else
    log "creating venv at ${VENV_DIR}"
    python3 -m venv "$VENV_DIR" || die "python3 -m venv failed (is python3-venv installed?)"
    ok "created venv"
fi
log "upgrading pip"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip >/dev/null || die "pip self-upgrade failed"
log "installing device-client dependencies (this can take a while on a Pi)"
"${VENV_DIR}/bin/python" -m pip install -r "${DEST}/requirements.txt" \
    || die "pip install -r requirements.txt failed"
ok "dependencies installed"

# 6d. Ownership + secret permissions (preserve/lock down an existing config).
chown -R "${SERVICE_USER}:${SERVICE_USER}" "$OPT_ROOT" || die "chown ${OPT_ROOT} failed"
if [ -f "${DEST}/config.json" ]; then
    chmod 600 "${DEST}/config.json" || die "chmod 600 config.json failed"
fi

# 6e. systemd unit (from the extracted bundle).
UNIT_SRC="${DEST}/${UNIT_NAME}"
if [ "$SYSTEMCTL_MISSING" = "1" ]; then
    warn "systemctl not found (not a systemd host?) — installing unit file only."
    cp "$UNIT_SRC" "$UNIT_DEST" || die "failed to copy unit to ${UNIT_DEST}"
    chmod 0644 "$UNIT_DEST" || die "failed to chmod unit file"
    warn "Enable it on a systemd host with:"
    warn "    sudo systemctl daemon-reload && sudo systemctl enable --now ${UNIT_NAME}"
    ok "unit file installed at ${UNIT_DEST}"
    exit 0
fi
log "installing systemd unit → ${UNIT_DEST}"
cp "$UNIT_SRC" "$UNIT_DEST" || die "failed to copy unit to ${UNIT_DEST}"
chmod 0644 "$UNIT_DEST" || die "failed to chmod unit file"
systemctl daemon-reload || die "systemctl daemon-reload failed"
log "enabling and starting ${UNIT_NAME}"
systemctl enable --now "$UNIT_NAME" >/dev/null 2>&1 \
    || die "systemctl enable --now failed"
ok "petal-device enabled and started"

# ── 7. Provision interactively when no config.json exists ────────────────────
log "step 7/7 — provisioning"
run_provision() {
    # Run provision.py as the service user with a real terminal on stdin. Under
    # `curl … | bash`, this script's own stdin is the pipe, so we attach /dev/tty.
    if command -v runuser >/dev/null 2>&1; then
        runuser -u "$SERVICE_USER" -- "${VENV_DIR}/bin/python" "${DEST}/provision.py" < /dev/tty
    else
        su -s /bin/sh "$SERVICE_USER" -c "'${VENV_DIR}/bin/python' '${DEST}/provision.py'" < /dev/tty
    fi
}

if [ -f "${DEST}/config.json" ]; then
    ok "config.json already present — reusing it"
    log "restarting ${UNIT_NAME} to pick up ${VERSION}"
    systemctl restart "$UNIT_NAME" || die "systemctl restart failed"
    ok "petal-device is running"
elif [ -e /dev/tty ] && [ -r /dev/tty ]; then
    log "no config.json — launching interactive provisioning as '${SERVICE_USER}'"
    run_provision || die "provisioning failed — re-run: curl -fsSL ${INSTALL_URL} | bash"
    [ -f "${DEST}/config.json" ] || die "provisioning finished but no config.json was written"
    chmod 600 "${DEST}/config.json" || true
    chown "${SERVICE_USER}:${SERVICE_USER}" "${DEST}/config.json" || true
    log "restarting ${UNIT_NAME} with fresh config"
    systemctl restart "$UNIT_NAME" || die "systemctl restart failed"
    ok "config.json written; petal-device is running"
else
    warn "no config.json and no interactive terminal — skipping provisioning."
    warn "Provision later, then the service will start automatically:"
    warn "    sudo -u ${SERVICE_USER} ${VENV_DIR}/bin/python ${DEST}/provision.py"
    warn "    sudo systemctl restart ${UNIT_NAME}"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
printf '\n'
ok "Install complete (${VERSION})."
printf '  Follow logs:   journalctl -u petal-device -f\n'
printf '  Status:        systemctl status petal-device\n'
printf '  Restart:       sudo systemctl restart petal-device\n'
