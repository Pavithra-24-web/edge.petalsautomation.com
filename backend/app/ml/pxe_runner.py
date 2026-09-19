"""
PXE host runtime — unpacks a PXE1 binary, launches runner.py as an OS process,
and speaks the Edge Impulse .eim stdio-JSONL protocol over stdin/stdout pipes.

Supports PXE format v1 (32-byte header, 5 sections) and v2 (36-byte header,
6 sections including postprocess_config.json).  Old v1 packages without
postprocess_config.json load unchanged.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

_PXE_MAGIC          = b"PXE1"
_PXE_HEADER_V1      = struct.Struct("<4sIIIIIII")   # 32 bytes, 5 sections
_PXE_HEADER_V2      = struct.Struct("<4sIIIIIIII")  # 36 bytes, 6 sections


def _validate_postprocess_config_dict(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("postprocess_config.json must contain a JSON object")
    if not isinstance(cfg.get("enabled"), bool):
        raise ValueError("postprocess_config.enabled must be a bool")
    th = cfg.get("threshold")
    if not isinstance(th, (int, float)) or not (0.0 <= float(th) <= 1.0):
        raise ValueError(f"postprocess_config.threshold must be float in [0,1], got {th!r}")
    cf = cfg.get("class_filter")
    if not isinstance(cf, list) or not all(isinstance(x, str) for x in cf):
        raise ValueError("postprocess_config.class_filter must be a list of strings")
    if not isinstance(cfg.get("tracking_enabled"), bool):
        raise ValueError("postprocess_config.tracking_enabled must be a bool")
    kg = cfg.get("keep_grace")
    if not isinstance(kg, int) or kg < 0:
        raise ValueError(f"postprocess_config.keep_grace must be int >= 0, got {kg!r}")
    mo = cfg.get("max_observations")
    if not isinstance(mo, int) or mo < 1:
        raise ValueError(f"postprocess_config.max_observations must be int >= 1, got {mo!r}")


def _load_postprocess_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        cfg = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"Malformed .pxe postprocess_config.json: {exc}") from exc
    _validate_postprocess_config_dict(cfg)
    return cfg


def _unpack_pxe(data: bytes, dest: Path) -> dict:
    """Parse a PXE1 v1 or v2 binary and write all sections to dest directory.

    v1 (format_version=1): 5 sections, no postprocess_config.json
    v2 (format_version=2): 6 sections, includes postprocess_config.json

    Returns the parsed manifest dict.
    """
    min_hdr = _PXE_HEADER_V1.size
    if len(data) < min_hdr:
        raise ValueError(f".pxe too small ({len(data)} bytes)")

    # Peek at version field (offset 4, uint32 LE)
    ver = struct.unpack_from("<I", data, 4)[0]

    if ver == 1:
        hdr_struct = _PXE_HEADER_V1
        if len(data) < hdr_struct.size:
            raise ValueError(f".pxe v1 too small ({len(data)} bytes)")
        magic, _ver, runner_sz, model_sz, manifest_sz, labels_sz, dsp_sz, _flags = \
            hdr_struct.unpack_from(data)
        postprocess_sz = 0
    elif ver == 2:
        hdr_struct = _PXE_HEADER_V2
        if len(data) < hdr_struct.size:
            raise ValueError(f".pxe v2 too small ({len(data)} bytes)")
        magic, _ver, runner_sz, model_sz, manifest_sz, labels_sz, dsp_sz, postprocess_sz, _flags = \
            hdr_struct.unpack_from(data)
    else:
        raise ValueError(f"Unsupported .pxe format version {ver}")

    if magic != _PXE_MAGIC:
        raise ValueError(f"Bad .pxe magic {magic!r}; expected {_PXE_MAGIC!r}")

    expected_size = (
        hdr_struct.size
        + runner_sz
        + model_sz
        + manifest_sz
        + labels_sz
        + dsp_sz
        + postprocess_sz
    )
    if len(data) != expected_size:
        raise ValueError(
            f"Malformed .pxe package: header declares {expected_size} bytes, got {len(data)}"
        )

    offset = hdr_struct.size
    for name, size in (
        ("runner.py",       runner_sz),
        ("model.tflite",    model_sz),
        ("manifest.json",   manifest_sz),
        ("labels.txt",      labels_sz),
        ("dsp_config.json", dsp_sz),
    ):
        (dest / name).write_bytes(data[offset: offset + size])
        offset += size

    if postprocess_sz:
        (dest / "postprocess_config.json").write_bytes(data[offset: offset + postprocess_sz])

    return json.loads((dest / "manifest.json").read_text())


class PxeProcess:
    """Manages one .pxe subprocess instance.

    Unpacks PXE1 binary to a temp directory, launches runner.py as a child
    process, completes the EI hello handshake, then serves classify calls
    over stdin/stdout JSON lines.

    Use as a context manager — cleanup is guaranteed on exit:
        with PxeProcess(pxe_bytes) as proc:
            result = proc.classify(features)
    """

    def __init__(self, pxe_bytes: bytes) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="pxe_")
        self._dir    = Path(self._tmpdir.name)
        try:
            self.manifest = _unpack_pxe(pxe_bytes, self._dir)
            # Load postprocess_config.json if present (v2 packages).
            pp_path = self._dir / "postprocess_config.json"
            self.postprocess_config: dict = _load_postprocess_config(pp_path)
            self._proc = subprocess.Popen(
                [sys.executable, str(self._dir / "runner.py")],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self._dir),
            )
            hello_line = self._proc.stdout.readline()
            if not hello_line:
                stderr = self._proc.stderr.read(2048).decode(errors="replace")
                raise RuntimeError(f".pxe runner did not send hello: {stderr}")
            self.hello = json.loads(hello_line)
        except Exception:
            self._cleanup()
            raise

    def classify(self, features: list) -> dict:
        """Send a classify request; return the parsed result dict.
        Blocks until the runner responds. Raises if the process has exited.
        """
        if self._proc.poll() is not None:
            raise RuntimeError(".pxe runner process exited unexpectedly")
        self._proc.stdin.write((json.dumps({"classify": features}) + "\n").encode())
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError(".pxe runner closed stdout during classify")
        return json.loads(line)

    def _cleanup(self) -> None:
        proc = getattr(self, "_proc", None)
        try:
            if proc is not None and proc.poll() is None:
                proc.stdin.write(b'{"stop": 1}\n')
                proc.stdin.flush()
                proc.wait(timeout=3)
        except Exception:
            try:
                if proc is not None:
                    proc.kill()
            except Exception:
                pass
        finally:
            self._tmpdir.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._cleanup()


def run_pxe_inference(pxe_bytes: bytes, features: list) -> dict:
    """One-shot convenience: unpack → launch → classify → stop → cleanup.

    Returns the raw classify result dict from the runner, plus private keys:
      ``_pxe_manifest``         — manifest dict from the PXE binary
      ``_pxe_postprocess_config`` — postprocess_config dict (empty for v1 packages)
    Callers should pop both private keys before returning the result to clients.
    """
    with PxeProcess(pxe_bytes) as proc:
        result = proc.classify(features)
        result["_pxe_manifest"]          = proc.manifest
        result["_pxe_postprocess_config"] = proc.postprocess_config
        return result
