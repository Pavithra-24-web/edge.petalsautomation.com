#!/usr/bin/env python3
"""
Petal Edge — interactive device provisioner (the `edge-impulse-daemon` equivalent).

One command configures a Pi/host end to end, exactly like Edge Impulse's daemon:

    python provision.py

walks you through:
  1. backend host + login (email/password → JWT, held only in memory)
  2. pick one of your projects
  3. reuse or create a device key for that project
  4. write device_client/config.json (chmod 600) with host + api_key + device_id
     (+ hmac_key when a new key is created — forwarder.py needs it to sign uploads)
  5. print the exact run command

Then just run the reference client — it picks the config up automatically:

    python client.py

Security model (matches the workspace safety rules):
  • The password is typed into THIS terminal via getpass — never accepted as a
    CLI argument, never echoed, never written anywhere.
  • Only two secrets are persisted: nothing (the JWT stays in memory) and the
    device api_key (written to config.json, which is chmod 600).

Stdlib only (urllib/getpass) so it runs on a bare Pi without pip installs.

Endpoints used:
  POST /api/v1/auth/login                         → { access_token }
  GET  /api/v1/projects/                           → user's projects
  GET  /api/v1/devices/project/{project_id}/keys   → existing device keys (prefixes only)
  POST /api/v1/devices/project/{project_id}/keys   → create a key (returns api_key once)
"""
from __future__ import annotations

import getpass
import json
import os
import pathlib
import socket
import sys
import urllib.error
import urllib.request

CONFIG_PATH = pathlib.Path(__file__).parent / "config.json"
DEFAULT_HOST = "http://localhost:8010"
DEFAULT_DEVICE_TYPE = "python"
_TIMEOUT = 15  # seconds per HTTP call


# ─── Tiny stdlib HTTP helpers ─────────────────────────────────────────────────

class ApiError(Exception):
    """An HTTP call returned a non-2xx status; carries status + server detail."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def _request(method: str, url: str, *, token: str | None = None, body: dict | None = None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8") or "null"
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = exc.reason
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            detail = payload.get("detail", detail) if isinstance(payload, dict) else detail
        except Exception:
            pass
        raise ApiError(exc.code, str(detail)) from None
    except urllib.error.URLError as exc:
        raise ApiError(0, f"could not reach {url} ({exc.reason})") from None


def _get(url: str, token: str | None = None):
    return _request("GET", url, token=token)


def _post(url: str, body: dict, token: str | None = None):
    return _request("POST", url, token=token, body=body)


# ─── Prompt helpers ───────────────────────────────────────────────────────────

def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            value = input(f"{text}{suffix}: ").strip()
        except EOFError:
            print()
            _abort("input stream closed")
        if value:
            return value
        if default is not None:
            return default
        print("  (a value is required)")


def _abort(msg: str) -> None:
    print(f"\n✗ {msg}", file=sys.stderr)
    sys.exit(1)


def _normalize_host(host: str) -> str:
    host = host.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


# ─── Provisioning steps ───────────────────────────────────────────────────────

def _login(host: str) -> str:
    """Prompt for credentials and return a JWT. Password is never stored."""
    print("\n── Sign in ──────────────────────────────────────────────")
    email = _prompt("Email")
    # getpass reads directly from the tty; the password is not echoed and is
    # discarded as soon as the token comes back.
    password = getpass.getpass("Password (hidden): ")
    if not password:
        _abort("password is required")
    try:
        data = _post(f"{host}/api/v1/auth/login", {"email": email, "password": password})
    except ApiError as exc:
        if exc.status == 401:
            _abort("login failed: invalid email or password")
        if exc.status == 403:
            _abort(f"login failed: {exc.detail}")
        _abort(f"login failed: {exc.detail}")
    finally:
        del password  # drop the plaintext promptly, regardless of outcome
    token = (data or {}).get("access_token")
    if not token:
        _abort("login response did not include an access_token")
    print(f"✓ Signed in as {email}")
    return token


def _pick_project(host: str, token: str) -> dict:
    projects = _get(f"{host}/api/v1/projects/", token=token)
    if not isinstance(projects, list) or not projects:
        _abort("no projects found for this account — create one in the dashboard first")
    print("\n── Select a project ─────────────────────────────────────")
    for i, p in enumerate(projects, 1):
        print(f"  {i}. {p.get('name', '(unnamed)')}")
    while True:
        choice = _prompt(f"Project number (1–{len(projects)})", "1")
        if choice.isdigit() and 1 <= int(choice) <= len(projects):
            chosen = projects[int(choice) - 1]
            print(f"✓ Project: {chosen.get('name')}")
            return chosen
        print("  (enter a number from the list)")


def _resolve_device_key(
    host: str, token: str, project_id: str, default_name: str
) -> tuple[str, str | None]:
    """Return ``(api_key, hmac_key)`` — created fresh, or supplied by the user.

    Existing keys are shown for context, but the backend reveals the full
    secrets (api_key AND hmac_key) only once, at creation, so a reused key can
    only supply its ef_… api_key by paste — its hmac_key is unrecoverable and
    comes back as None (the forwarder then needs PETAL_HMAC_KEY, or create a new
    key). The default path creates a new key and captures both secrets.
    """
    keys = _get(f"{host}/api/v1/devices/project/{project_id}/keys", token=token)
    keys = keys if isinstance(keys, list) else []

    print("\n── Device key ───────────────────────────────────────────")
    print("  1. Create a new device key (recommended)")
    for i, k in enumerate(keys, start=2):
        prefix = k.get("api_key_prefix") or "????????"
        name = k.get("name") or "(unnamed)"
        created = (k.get("created_at") or "")[:10]
        state = "" if k.get("is_active", True) else "  [revoked]"
        print(f"  {i}. reuse '{name}'  ({prefix}…)  {created}{state}")

    while True:
        choice = _prompt(f"Choice (1–{len(keys) + 1})", "1")
        if not (choice.isdigit() and 1 <= int(choice) <= len(keys) + 1):
            print("  (enter a number from the list)")
            continue
        idx = int(choice)
        if idx == 1:
            return _create_key(host, token, project_id, default_name)
        # Reuse an existing key → the secret can't be read back, so ask for it.
        target = keys[idx - 2]
        print(
            f"\n  The full secret for '{target.get('name') or '(unnamed)'}' is only shown "
            "once, at creation,\n  so it can't be retrieved. Paste its ef_… api_key, "
            "or leave blank to create a new key."
        )
        pasted = input("  api_key: ").strip()
        if not pasted:
            return _create_key(host, token, project_id, default_name)
        if not pasted.startswith("ef_"):
            print("  (that doesn't look like an ef_… device key — try again)")
            continue
        print("✓ Using the provided device key")
        print(
            "  (its hmac_key can't be recovered — set PETAL_HMAC_KEY to sign uploads\n"
            "   with forwarder.py, or create a new key to capture both secrets.)"
        )
        return pasted, None


def _create_key(host: str, token: str, project_id: str, default_name: str) -> tuple[str, str | None]:
    name = _prompt("Name for the new key", default_name)
    data = _post(
        f"{host}/api/v1/devices/project/{project_id}/keys",
        {"name": name},
        token=token,
    )
    api_key = (data or {}).get("api_key")
    if not api_key:
        _abort("key creation response did not include an api_key")
    # hmac_key is returned once, at creation — capture it so forwarder.py can sign.
    hmac_key = (data or {}).get("hmac_key")
    if not hmac_key:
        print("  (warning: key response had no hmac_key — forwarder.py will need PETAL_HMAC_KEY)")
    print(f"✓ Created device key '{name}'  ({_mask(api_key)})")
    return api_key, hmac_key


def _mask(secret: str) -> str:
    if len(secret) <= 10:
        return "ef_…"
    return f"{secret[:6]}…{secret[-2:]}"


def _write_config(config: dict) -> None:
    # Write first, then tighten permissions to 0600 (owner read/write only) —
    # the file holds the device api_key. chmod is a POSIX no-op on Windows but
    # never errors there.
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError as exc:
        print(f"  (warning: could not chmod 600 {CONFIG_PATH}: {exc})")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("Petal Edge device provisioner")
    print("=" * 58)
    hostname = socket.gethostname() or "petal-device"

    host = _normalize_host(_prompt("Backend host", DEFAULT_HOST))
    token = _login(host)
    project = _pick_project(host, token)
    api_key, hmac_key = _resolve_device_key(host, token, project["id"], default_name=hostname)

    print("\n── Device identity ──────────────────────────────────────")
    device_id = _prompt("Device ID", hostname)
    device_type = _prompt("Device type", DEFAULT_DEVICE_TYPE)

    config = {
        "host": host,
        "api_key": api_key,
        "device_id": device_id,
        "device_type": device_type,
    }
    # hmac_key is only present when a new key was created (it's shown once). It's
    # required by forwarder.py to sign serial uploads; client.py ignores it.
    if hmac_key:
        config["hmac_key"] = hmac_key
    _write_config(config)

    print("\n" + "=" * 58)
    print(f"✓ Wrote {CONFIG_PATH}  (mode 600, no password stored)")
    print(f"  project    : {project.get('name')}")
    print(f"  device_id  : {device_id}")
    print(f"  api_key    : {_mask(api_key)}")
    if hmac_key:
        print(f"  hmac_key   : {_mask(hmac_key)}  (for forwarder.py)")
    print("\nStart the device — config.json is picked up automatically:\n")
    print(f"    python {pathlib.Path(__file__).parent / 'client.py'}")
    print("\n(An env var of the same PETAL_* name still overrides the file.)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
