#!/usr/bin/env python3
"""
Petal Edge — MCU serial data forwarder (the Edge Impulse `data-forwarder` equivalent).

Small MCUs (ESP32 / Arduino / STM32) can't run the WebSocket device client, but
they can stream newline-delimited CSV sensor rows over a serial/USB link. This
tool reads those rows, buffers a sample window, wraps it in the Edge-Impulse
ingestion envelope, signs it with HMAC-SHA256, and POSTs it to the backend
exactly like `client.py` does — so the samples land in the same training set.

    # one command after provisioning (config.json supplies host/api_key/hmac_key):
    python forwarder.py --port COM3 --axes accX,accY,accZ --label wave

    # offline sanity check — print the signed envelope instead of uploading:
    echo "0.1,0.2,9.8" | python forwarder.py --stdin --dry-run \
        --axes accX,accY,accZ --label wave --freq 62.5 --length-ms 1000

Wire format (matches backend/app/services/ingestion.py + ingestion_auth.py):
    {
      "protected": {"ver": "v1", "alg": "HS256", "iat": <unix_ts>},
      "signature": "<64-hex HMAC-SHA256>",   ← 64 zeros while signing, then filled
      "payload": {
        "device_name": "<device_id>",
        "device_type": "<board>",
        "interval_ms": <1000 / freq_hz>,
        "sensors":     [{"name": "accX", "units": "N/A"}, ...],
        "values":      [[0.1, 0.2, 9.8], ...],   ← one row per serial CSV line
        "label":       "<label>"
      }
    }

The HMAC is computed over the compact JSON body with the "signature" field set to
64 zeros; the real signature is then substituted textually (no re-serialisation,
no key reordering) so backend `build_hmac_body` reproduces the signed bytes and
verification passes. See backend/app/core/ingestion_auth.py.

Config precedence for every setting (same rule as client.py):
    CLI flag  >  environment variable  >  config.json  >  built-in default

Requires `pyserial` only for real serial reads. --dry-run and --stdin work with
the standard library alone (HTTP upload uses urllib, not aiohttp), so this file
can be exercised end to end without hardware or extra installs.

Endpoint used:
    POST /api/v1/ingestion/{category}/data   (category: training | testing | anomaly)
      headers: x-api-key (required), x-label (optional, mirrors payload.label)
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

logger = logging.getLogger("petal.forwarder")

# ── Provisioned config file (device_client/config.json) ───────────────────────
# Written by provision.py. Precedence for every setting it carries is
# CLI flag > env var > config.json > built-in default, so an explicit flag or
# env var always wins and the placeholder defaults apply only when nothing else
# is present. Missing/malformed file is a no-op (falls back to env/defaults).
_CONFIG_PATH = pathlib.Path(__file__).parent / "config.json"

# Short human keys written by provision.py → the PETAL_* names used below.
_CONFIG_KEY_ALIASES = {
    "host":             "PETAL_HOST",
    "api_key":          "PETAL_API_KEY",
    "hmac_key":         "PETAL_HMAC_KEY",
    "device_id":        "PETAL_DEVICE_ID",
    "device_type":      "PETAL_DEVICE_TYPE",
    "serial_port":      "PETAL_SERIAL_PORT",
    "baud":             "PETAL_BAUD",
    "sensor_axes":      "PETAL_SENSOR_AXES",
    "sample_length_ms": "PETAL_SAMPLE_LENGTH_MS",
    "frequency_hz":     "PETAL_FREQ_HZ",
    "label":            "PETAL_LABEL",
    "category":         "PETAL_CATEGORY",
}


def _load_config_file() -> Dict[str, str]:
    """Load device_client/config.json into a {PETAL_*: value} dict.

    Returns {} when the file is absent, unreadable, or malformed — a bad file
    must never stop the forwarder from starting. Both the short keys emitted by
    provision.py and raw PETAL_* keys are accepted.
    """
    if not _CONFIG_PATH.exists():
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as exc:
        logger.warning("[config] ignoring %s (could not parse: %s)", _CONFIG_PATH, exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning("[config] ignoring %s (expected a JSON object)", _CONFIG_PATH)
        return {}
    out: Dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            continue
        env_key = _CONFIG_KEY_ALIASES.get(key) or (key if key.startswith("PETAL_") else None)
        if env_key:
            out[env_key] = str(value)
    return out


_FILE_CONFIG: Dict[str, str] = _load_config_file()


def _cfg(key: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a setting as: environment > config.json > built-in default."""
    if key in os.environ:
        return os.environ[key]
    if key in _FILE_CONFIG:
        return _FILE_CONFIG[key]
    return default


# ── Sensor axes ───────────────────────────────────────────────────────────────

def parse_axes(raw: str) -> List[Dict[str, str]]:
    """Turn "accX,accY,accZ" (or "accX:m/s2,accY:m/s2") into EI sensor dicts.

    Each axis is ``{"name": <name>, "units": <units>}``; units default to "N/A"
    when not given via the ``name:units`` form.
    """
    sensors: List[Dict[str, str]] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            name, _, units = token.partition(":")
            name, units = name.strip(), units.strip() or "N/A"
        else:
            name, units = token, "N/A"
        sensors.append({"name": name, "units": units})
    if not sensors:
        raise ValueError("no sensor axes configured — pass --axes accX,accY,accZ")
    return sensors


def parse_csv_row(line: str, n_axes: int) -> Optional[List[float]]:
    """Parse one CSV sensor row into a list of floats, or None to skip it.

    Rows that are blank, non-numeric (e.g. a firmware boot banner), or have the
    wrong number of columns are skipped with a debug log rather than aborting the
    stream — MCUs commonly emit stray text before real data starts.
    """
    line = line.strip()
    if not line:
        return None
    cols = [c.strip() for c in line.split(",")]
    if len(cols) != n_axes:
        logger.debug("[serial] skip row (%d cols, expected %d): %r", len(cols), n_axes, line)
        return None
    try:
        return [float(c) for c in cols]
    except ValueError:
        logger.debug("[serial] skip non-numeric row: %r", line)
        return None


# ── Envelope building + signing ───────────────────────────────────────────────

def build_payload(
    *,
    device_name: str,
    device_type: str,
    interval_ms: float,
    sensors: List[Dict[str, str]],
    rows: List[List[float]],
    label: Optional[str],
) -> Dict:
    """Assemble the inner Edge-Impulse ingestion payload from a sample window."""
    payload: Dict = {
        "device_name": device_name,
        "device_type": device_type,
        "interval_ms": interval_ms,
        "sensors":     sensors,
        "values":      rows,
    }
    if label:
        payload["label"] = label
    return payload


def build_signed_envelope(payload: Dict, hmac_key_hex: str) -> Tuple[bytes, str]:
    """Wrap ``payload`` in the EI envelope and sign it. Returns (body_bytes, sig).

    The HMAC is computed over the compact JSON with the "signature" field set to
    64 zeros; the real signature then replaces those zeros in-place. Because the
    only byte difference between the signed form and the sent form is the
    signature value, backend ``build_hmac_body`` (which zeroes the field again)
    reproduces the exact signed bytes and verification succeeds.
    """
    zeros = "0" * 64
    outer = {
        "protected": {"ver": "v1", "alg": "HS256", "iat": int(time.time())},
        "signature": zeros,
        "payload":   payload,
    }
    signed_bytes = json.dumps(outer, separators=(",", ":")).encode()
    real_sig = hmac.new(bytes.fromhex(hmac_key_hex), signed_bytes, hashlib.sha256).hexdigest()
    # Re-emit with the real signature. Dict-unpack keeps insertion order and only
    # updates the value of the existing "signature" key, so the byte layout is
    # identical to `signed_bytes` apart from that field — exactly what the server
    # re-derives when it zeroes the signature to verify.
    final = json.dumps({**outer, "signature": real_sig}, separators=(",", ":")).encode()
    return final, real_sig


# ── HTTP upload (stdlib) ──────────────────────────────────────────────────────

def post_sample(
    *,
    host: str,
    api_key: str,
    category: str,
    body: bytes,
    label: Optional[str],
    timeout: int = 30,
) -> Dict:
    """POST a signed envelope to the ingestion endpoint; return the parsed JSON.

    Raises RuntimeError on any non-2xx response (message includes server detail).
    """
    url = f"{host.rstrip('/')}/api/v1/ingestion/{category}/data"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("x-api-key", api_key)
    if label:
        req.add_header("x-label", label)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8") or "null"
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = exc.reason
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            detail = payload.get("detail", detail) if isinstance(payload, dict) else detail
        except Exception:
            pass
        raise RuntimeError(f"upload HTTP {exc.code}: {str(detail)[:160]}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach {url} ({exc.reason})") from None


def _log_upload_result(data: Dict) -> None:
    """Log id / dedup / filename / hash, mirroring client._upload_sample."""
    tag = "dedup" if data.get("duplicate") else "new"
    logger.info(
        "[sample] uploaded(%s)  id=%s  filename=%s  label_id=%s  hash=%s",
        tag,
        data.get("id", "?"),
        data.get("filename", "?"),
        data.get("label_id"),
        (data.get("payload_hash") or "")[:12],
    )


# ── Row sources ───────────────────────────────────────────────────────────────

def iter_serial_rows(port: str, baud: int, n_axes: int) -> Iterator[List[float]]:
    """Yield parsed float rows read line-by-line from a serial port.

    `serial` (pyserial) is imported here so that --dry-run/--stdin and the test
    suite never require it. Accepts a pyserial URL (e.g. "loop://") as `port`.
    """
    try:
        import serial  # type: ignore
    except ImportError:
        raise SystemExit(
            "pyserial is required to read from a serial port — `pip install pyserial`, "
            "or use --stdin for offline testing."
        )
    opener = getattr(serial, "serial_for_url", None)
    ser = opener(port, baudrate=baud, timeout=1) if (opener and "://" in port) \
        else serial.Serial(port, baudrate=baud, timeout=1)
    logger.info("[serial] reading %s @ %d baud", port, baud)
    try:
        with ser:
            while True:
                raw = ser.readline()
                if not raw:
                    continue  # read timeout — keep waiting for data
                try:
                    line = raw.decode("utf-8", errors="replace")
                except Exception:
                    continue
                row = parse_csv_row(line, n_axes)
                if row is not None:
                    yield row
    except KeyboardInterrupt:
        return


def iter_stdin_rows(n_axes: int) -> Iterator[List[float]]:
    """Yield parsed float rows read from stdin (offline testing without pyserial)."""
    for line in sys.stdin:
        row = parse_csv_row(line, n_axes)
        if row is not None:
            yield row


# ── Sample assembly loop ──────────────────────────────────────────────────────

def _collect_window(rows: Iterable[List[float]], n_rows: int) -> Iterator[List[List[float]]]:
    """Group a flat stream of rows into fixed-size sample windows of `n_rows`."""
    buf: List[List[float]] = []
    for row in rows:
        buf.append(row)
        if len(buf) >= n_rows:
            yield buf
            buf = []
    # Emit a trailing partial window so piped/finite sources aren't dropped.
    if buf:
        yield buf


def run_forwarder(
    *,
    rows: Iterable[List[float]],
    host: str,
    api_key: str,
    hmac_key: str,
    category: str,
    device_name: str,
    device_type: str,
    sensors: List[Dict[str, str]],
    interval_ms: float,
    n_rows: int,
    label: Optional[str],
    dry_run: bool,
    max_samples: Optional[int] = None,
) -> int:
    """Drive the read→buffer→sign→upload loop. Returns the number of samples sent."""
    sent = 0
    for window in _collect_window(rows, n_rows):
        payload = build_payload(
            device_name=device_name,
            device_type=device_type,
            interval_ms=interval_ms,
            sensors=sensors,
            rows=window,
            label=label,
        )
        body, sig = build_signed_envelope(payload, hmac_key)

        if dry_run:
            logger.info(
                "[dry-run] sample #%d  rows=%d  sig=%s…  %d bytes",
                sent + 1, len(window), sig[:12], len(body),
            )
            sys.stdout.write(body.decode("utf-8") + "\n")
            sys.stdout.flush()
        else:
            data = post_sample(
                host=host, api_key=api_key, category=category, body=body, label=label,
            )
            _log_upload_result(data)

        sent += 1
        if max_samples is not None and sent >= max_samples:
            break
    return sent


# ── CLI ───────────────────────────────────────────────────────────────────────

def _first(*values: Optional[str]) -> Optional[str]:
    """Return the first non-None/non-empty value (CLI > env/config > default)."""
    for v in values:
        if v is not None and str(v).strip() != "":
            return v
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forwarder.py",
        description="Read serial CSV sensor rows and upload signed Edge-Impulse samples.",
    )
    p.add_argument("--host", help="Backend base URL (default: config.json / http://localhost:8010)")
    p.add_argument("--port", help="Serial port or pyserial URL (e.g. COM3, /dev/ttyUSB0, loop://)")
    p.add_argument("--baud", type=int, help="Serial baud rate (default 115200)")
    p.add_argument("--axes", help="Comma-separated sensor axis names, e.g. accX,accY,accZ")
    p.add_argument("--label", help="Label applied to every uploaded sample")
    p.add_argument("--category", choices=["training", "testing", "anomaly"],
                   help="Ingestion split (default training)")
    p.add_argument("--freq", type=float, help="Sampling frequency in Hz (default 62.5)")
    p.add_argument("--length-ms", type=int, dest="length_ms",
                   help="Target sample window length in ms (default 1000)")
    p.add_argument("--rows", type=int,
                   help="Rows per sample window (overrides freq×length computation)")
    p.add_argument("--count", type=int,
                   help="Stop after uploading this many samples (default: run forever)")
    p.add_argument("--stdin", action="store_true",
                   help="Read CSV rows from stdin instead of a serial port (offline testing)")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Print the signed envelope instead of uploading")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
    )

    host        = _first(args.host, _cfg("PETAL_HOST"), "http://localhost:8010")
    api_key     = _first(_cfg("PETAL_API_KEY"), "ef_changeme")
    hmac_key    = _cfg("PETAL_HMAC_KEY")
    device_name = _first(_cfg("PETAL_DEVICE_ID"), "mcu-forwarder-001")
    device_type = _first(_cfg("PETAL_DEVICE_TYPE"), "mcu")
    category    = _first(args.category, _cfg("PETAL_CATEGORY"), "training")
    label       = _first(args.label, _cfg("PETAL_LABEL"))
    axes_raw    = _first(args.axes, _cfg("PETAL_SENSOR_AXES"))
    baud        = int(_first(str(args.baud) if args.baud else None, _cfg("PETAL_BAUD"), "115200"))
    freq_hz     = float(_first(str(args.freq) if args.freq else None, _cfg("PETAL_FREQ_HZ"), "62.5"))
    length_ms   = int(_first(str(args.length_ms) if args.length_ms else None,
                             _cfg("PETAL_SAMPLE_LENGTH_MS"), "1000"))
    port        = _first(args.port, _cfg("PETAL_SERIAL_PORT"))

    # ── Validate required config ──────────────────────────────────────────────
    if not axes_raw:
        logger.error("no sensor axes configured — pass --axes accX,accY,accZ (or set PETAL_SENSOR_AXES)")
        return 2
    try:
        sensors = parse_axes(axes_raw)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    if freq_hz <= 0:
        logger.error("frequency must be > 0 Hz (got %s)", freq_hz)
        return 2
    interval_ms = round(1000.0 / freq_hz, 4)
    n_rows = args.rows if args.rows else max(1, round(length_ms / 1000.0 * freq_hz))

    if not hmac_key:
        logger.error(
            "no HMAC key available — the forwarder signs every upload. Run provision.py "
            "(writes hmac_key into config.json) or set PETAL_HMAC_KEY."
        )
        return 2
    if not args.dry_run and (not api_key or api_key == "ef_changeme"):
        logger.error(
            "no device api_key — run provision.py or set PETAL_API_KEY (uploads need x-api-key)."
        )
        return 2

    # ── Row source ────────────────────────────────────────────────────────────
    n_axes = len(sensors)
    if args.stdin:
        rows: Iterable[List[float]] = iter_stdin_rows(n_axes)
    elif port:
        rows = iter_serial_rows(port, baud, n_axes)
    else:
        logger.error("no serial --port configured (or use --stdin) — nothing to read")
        return 2

    logger.info(
        "[forwarder] host=%s  category=%s  device=%s/%s  axes=%s  freq=%.3fHz  "
        "interval=%.4fms  window=%d rows  label=%s  %s",
        host, category, device_name, device_type, [s["name"] for s in sensors],
        freq_hz, interval_ms, n_rows, label or "(none)",
        "DRY-RUN" if args.dry_run else "upload",
    )

    try:
        sent = run_forwarder(
            rows=rows,
            host=host,
            api_key=api_key,
            hmac_key=hmac_key,
            category=category,
            device_name=device_name,
            device_type=device_type,
            sensors=sensors,
            interval_ms=interval_ms,
            n_rows=n_rows,
            label=label,
            dry_run=args.dry_run,
            max_samples=args.count,
        )
    except KeyboardInterrupt:
        logger.info("[forwarder] interrupted")
        return 130
    except RuntimeError as exc:
        logger.error("[forwarder] upload failed: %s", exc)
        return 1

    logger.info("[forwarder] done — %d sample(s) %s", sent,
                "built" if args.dry_run else "uploaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
