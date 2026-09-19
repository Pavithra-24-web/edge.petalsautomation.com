#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# Petal Edge — device-client installer (Debian / Raspberry Pi OS)
#
# One-shot, idempotent installer that provisions and runs device_client/client.py
# as a systemd service (petal-device) with auto-restart and start-on-boot.
#
#   sudo ./install.sh
#
# Steps:
#   1. ensure python3 + venv + pip (apt-get only if missing)
#   2. create the `petal` system user
#   3. create /opt/petal/venv and pip install -r requirements.txt
#   4. copy device_client/ → /opt/petal/device_client (preserving an existing config)
#   5. provision config.json interactively when absent (or print how to)
#   6. install + enable + start the petal-device systemd unit
#
# Fails loudly on any step; never silently continues.
# ─────────────────────────────────────────────────────────────────────────────
set -eu

OPT_ROOT="/opt/petal"
VENV_DIR="${OPT_ROOT}/venv"
DEST="${OPT_ROOT}/device_client"
SERVICE_USER="petal"
UNIT_NAME="petal-device.service"
UNIT_DEST="/etc/systemd/system/${UNIT_NAME}"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)

# ── Output helpers ───────────────────────────────────────────────────────────
log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[  ok   ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[ warn  ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ FAIL  ]\033[0m %s\n' "$*" >&2; exit 1; }

# ── Preconditions ────────────────────────────────────────────────────────────
[ "$(id -u)" = "0" ] || die "must run as root — try: sudo ./install.sh"

command -v systemctl >/dev/null 2>&1 || SYSTEMCTL_MISSING=1
: "${SYSTEMCTL_MISSING:=0}"

# ── 1. Base packages: python3, venv, pip ─────────────────────────────────────
ensure_apt_pkg() {
    # ensure_apt_pkg <dpkg-name> <probe-command>
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

log "checking base packages (python3, venv, pip)"
ensure_apt_pkg python3       "command -v python3"
ensure_apt_pkg python3-venv  "python3 -c 'import venv, ensurepip'"
ensure_apt_pkg python3-pip   "command -v pip3 || python3 -m pip --version"
ok "python3: $(python3 --version 2>&1)"

# ── 2. Service user ──────────────────────────────────────────────────────────
if id "$SERVICE_USER" >/dev/null 2>&1; then
    ok "user '$SERVICE_USER' already exists"
else
    log "creating system user '$SERVICE_USER'"
    useradd --system --home-dir "$OPT_ROOT" --shell /usr/sbin/nologin "$SERVICE_USER" \
        || die "failed to create user '$SERVICE_USER'"
    ok "created user '$SERVICE_USER'"
fi

# ── 3. Virtualenv + dependencies ─────────────────────────────────────────────
mkdir -p "$OPT_ROOT" || die "could not create $OPT_ROOT"

if [ -x "${VENV_DIR}/bin/python" ]; then
    ok "venv already present at $VENV_DIR"
else
    log "creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR" || die "python3 -m venv failed (is python3-venv installed?)"
    ok "created venv"
fi

REQ_SRC="${SCRIPT_DIR}/requirements.txt"
[ -f "$REQ_SRC" ] || die "requirements.txt not found next to install.sh ($REQ_SRC)"

log "upgrading pip"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip >/dev/null \
    || die "pip self-upgrade failed"
log "installing device-client dependencies (this can take a while on a Pi)"
"${VENV_DIR}/bin/python" -m pip install -r "$REQ_SRC" \
    || die "pip install -r requirements.txt failed"
ok "dependencies installed"

# ── 4. Copy source → /opt/petal/device_client (keep existing config.json) ─────
SRC_REAL=$(CDPATH= cd -- "$SCRIPT_DIR" && pwd -P)
DEST_REAL=$(mkdir -p "$DEST" && CDPATH= cd -- "$DEST" && pwd -P)

if [ "$SRC_REAL" = "$DEST_REAL" ]; then
    ok "source is already the install target — skipping copy"
else
    # Preserve a device's provisioned config across re-installs.
    SAVED_CFG=""
    if [ -f "${DEST}/config.json" ]; then
        SAVED_CFG=$(mktemp)
        cp -p "${DEST}/config.json" "$SAVED_CFG"
    fi

    log "copying device_client → $DEST"
    cp -a "${SRC_REAL}/." "${DEST}/" || die "copy to $DEST failed"

    # Never ship a developer checkout's config.json/.env unless it was already
    # the target's own provisioned file.
    if [ -n "$SAVED_CFG" ]; then
        cp -p "$SAVED_CFG" "${DEST}/config.json"
        rm -f "$SAVED_CFG"
    else
        rm -f "${DEST}/config.json"
    fi

    # Prune transient artefacts that may have ridden along.
    rm -rf "${DEST}/.pytest_cache" 2>/dev/null || true
    find "$DEST" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    ok "copied source"
fi

# The installer runs as root; hand the whole tree to the service user.
chown -R "${SERVICE_USER}:${SERVICE_USER}" "$OPT_ROOT" || die "chown $OPT_ROOT failed"

# ── 5. Configuration (provision.py writes config.json, chmod 600) ─────────────
if [ -f "${DEST}/config.json" ]; then
    ok "config.json already present — reusing it"
    CONFIGURED=1
elif [ -t 0 ]; then
    log "no config.json found — launching interactive provisioning"
    # Run as the service user so the written config is owned correctly.
    if command -v runuser >/dev/null 2>&1; then
        runuser -u "$SERVICE_USER" -- "${VENV_DIR}/bin/python" "${DEST}/provision.py" \
            || die "provisioning failed — re-run: sudo ./install.sh"
    else
        su -s /bin/sh "$SERVICE_USER" -c "'${VENV_DIR}/bin/python' '${DEST}/provision.py'" \
            || die "provisioning failed — re-run: sudo ./install.sh"
    fi
    [ -f "${DEST}/config.json" ] || die "provisioning finished but no config.json was written"
    ok "config.json written"
    CONFIGURED=1
else
    CONFIGURED=0
    warn "no config.json and no interactive terminal — skipping provisioning."
    warn "Provision later, then start the service:"
    warn "    sudo -u ${SERVICE_USER} ${VENV_DIR}/bin/python ${DEST}/provision.py"
    warn "    sudo systemctl start ${UNIT_NAME}"
fi

# ── 6. systemd unit ──────────────────────────────────────────────────────────
UNIT_SRC="${SCRIPT_DIR}/${UNIT_NAME}"
[ -f "$UNIT_SRC" ] || die "unit template not found ($UNIT_SRC)"

if [ "$SYSTEMCTL_MISSING" = "1" ]; then
    warn "systemctl not found (not a systemd host?) — installing unit file only."
    cp "$UNIT_SRC" "$UNIT_DEST" || die "failed to copy unit to $UNIT_DEST"
    chmod 0644 "$UNIT_DEST" || die "failed to chmod unit file"
    warn "Enable it on a systemd host with:"
    warn "    sudo systemctl daemon-reload && sudo systemctl enable --now ${UNIT_NAME}"
    ok "unit file installed at $UNIT_DEST"
    exit 0
fi

log "installing systemd unit → $UNIT_DEST"
cp "$UNIT_SRC" "$UNIT_DEST" || die "failed to copy unit to $UNIT_DEST"
chmod 0644 "$UNIT_DEST" || die "failed to chmod unit file"
systemctl daemon-reload || die "systemctl daemon-reload failed"

log "enabling ${UNIT_NAME} (start on boot)"
systemctl enable "$UNIT_NAME" >/dev/null 2>&1 || die "systemctl enable failed"

if [ "$CONFIGURED" = "1" ]; then
    log "starting ${UNIT_NAME}"
    systemctl restart "$UNIT_NAME" || die "systemctl start failed"
    ok "petal-device is enabled and running"
else
    warn "service enabled but NOT started (no config.json yet)."
fi

# ── Done ─────────────────────────────────────────────────────────────────────
printf '\n'
ok "Install complete."
printf '  Follow logs:   journalctl -u petal-device -f\n'
printf '  Status:        systemctl status petal-device\n'
printf '  Restart:       sudo systemctl restart petal-device\n'
