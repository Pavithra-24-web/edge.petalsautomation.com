"""PXE runner subprocess lifecycle and inference tick mixin."""
from __future__ import annotations

import json
import logging
import pathlib
import subprocess
import sys
import types
from typing import Any, Optional

logger = logging.getLogger("unoq")

try:
    import numpy as _np
    _NP_AVAILABLE = True
except ImportError:
    _np = None  # type: ignore[assignment]
    _NP_AVAILABLE = False

try:
    from tflite_runtime.interpreter import Interpreter as _TFLiteInterpreter  # type: ignore[import]
    _TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as _tf  # type: ignore[import]
        _TFLiteInterpreter = _tf.lite.Interpreter
        _TFLITE_AVAILABLE = True
    except ImportError:
        _TFLiteInterpreter = None  # type: ignore[assignment]
        _TFLITE_AVAILABLE = False


class _InferenceEngineMixin:
    """Mixin providing PXE runner management and inference tick execution for UnoQRuntime."""

    _pkg_dir:    Optional[pathlib.Path]
    _pkg_format: str
    _pxe_proc:   Optional[subprocess.Popen]
    _infer_mod:  Optional[types.ModuleType]
    _interpreter: Any
    _infer_tick:  int
    _cv2_warned:  bool

    def _launch_pxe_runner(self) -> None:
        """Launch runner.py from the installed .pxe package as a subprocess.
        Completes the EI hello handshake. Reuses an already-running process.
        """
        if self._pxe_proc is not None and self._pxe_proc.poll() is None:
            logger.debug("[pxe] runner already running, reusing pid=%d", self._pxe_proc.pid)
            return
        if self._pkg_dir is None:
            logger.error("[pxe] cannot launch runner — no package dir")
            return
        runner = self._pkg_dir / "runner.py"
        if not runner.exists():
            logger.error("[pxe] runner.py not found in %s", self._pkg_dir)
            return
        try:
            proc = subprocess.Popen(
                [sys.executable, str(runner)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self._pkg_dir),
            )
            hello_line = proc.stdout.readline()
            if not hello_line:
                stderr = proc.stderr.read(512).decode(errors="replace")
                logger.error("[pxe] runner failed to start: %s", stderr)
                proc.kill()
                return
            hello = json.loads(hello_line)
            self._pxe_proc = proc
            mp = hello.get("model_parameters", {})
            logger.info("[pxe] runner started pid=%d  model_type=%s  labels=%s",
                        proc.pid, mp.get("model_type"), mp.get("labels"))
        except Exception as exc:
            logger.error("[pxe] failed to launch runner: %s", exc)

    def _stop_pxe_runner(self) -> None:
        """Gracefully stop the .pxe runner subprocess."""
        if self._pxe_proc is None:
            return
        try:
            if self._pxe_proc.poll() is None:
                self._pxe_proc.stdin.write(b'{"stop": 1}\n')
                self._pxe_proc.stdin.flush()
                self._pxe_proc.wait(timeout=3)
        except Exception:
            try:
                self._pxe_proc.kill()
            except Exception:
                pass
        finally:
            self._pxe_proc = None
        logger.info("[pxe] runner stopped")

    def _ensure_interpreter(self) -> bool:
        """Lazy-init the TFLite interpreter from the active package."""
        if self._interpreter is not None:
            return True
        if not _TFLITE_AVAILABLE or not _NP_AVAILABLE:
            return False
        if self._pkg_dir is None or self._infer_mod is None:
            return False
        model_path = self._pkg_dir / "model.tflite"
        if not model_path.exists():
            logger.error("[infer] model.tflite not found in %s", self._pkg_dir)
            return False
        try:
            self._interpreter = self._infer_mod.load_interpreter(str(model_path))
            logger.info("[infer] interpreter loaded from %s", model_path)
            return True
        except Exception as exc:
            logger.error("[infer] interpreter load failed: %s", exc)
            return False

    def _run_pxe_inference_tick(self) -> Optional[dict]:
        """One inference tick via the .pxe subprocess runner (stdio-JSONL)."""
        if self._pxe_proc is None or self._pxe_proc.poll() is not None:
            logger.warning("[pxe] runner exited — attempting restart")
            self._launch_pxe_runner()
            if self._pxe_proc is None:
                return None
        manifest_path = self._pkg_dir / "manifest.json"  # type: ignore[operator]
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            logger.error("[pxe] manifest read failed: %s", exc)
            return None
        input_shape = manifest.get("input_shape", [])
        features_arr = (
            # For .pxe image models: send raw pixel data so the embedded runner
            # applies its own DSP pipeline (resize mode, normalization, channel
            # order).  Pre-resizing here would double-apply those transforms and
            # corrupt FOMO spatial semantics.
            self._raw_frame_pixels_for_pxe(manifest)
            if len(input_shape) >= 3
            else self._features_from_sensor(input_shape)
        )
        if features_arr is None:
            return None
        try:
            features = (
                features_arr.tolist()
                if len(input_shape) >= 3
                else features_arr.flatten().tolist()
            )
            self._pxe_proc.stdin.write((json.dumps({"classify": features}) + "\n").encode())
            self._pxe_proc.stdin.flush()
            line = self._pxe_proc.stdout.readline()
            if not line:
                logger.error("[pxe] runner stdout closed unexpectedly")
                self._pxe_proc = None
                return None
            normalized = self._normalize_pxe_result(json.loads(line), manifest)
            return self._apply_pxe_tracking(normalized)
        except Exception as exc:
            logger.error("[pxe] classify failed: %s", exc)
            return None

    def _run_inference_tick(self) -> Optional[dict]:
        """Run one inference tick synchronously (called from executor)."""
        if self._pkg_format == "pxe":
            return self._run_pxe_inference_tick()
        # .pe — existing in-process path below, untouched
        if not self._ensure_interpreter():
            return None
        if self._infer_mod is None or not _NP_AVAILABLE:
            return None

        manifest_path = self._pkg_dir / "manifest.json"  # type: ignore[operator]
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            logger.error("[infer] manifest read failed: %s", exc)
            return None

        input_shape = manifest.get("input_shape", [])
        is_image    = len(input_shape) >= 3

        if is_image:
            features = self._features_from_image(manifest, input_shape)
        else:
            features = self._features_from_sensor(input_shape)

        if features is None:
            return None

        try:
            result = self._infer_mod.run_inference(self._interpreter, features)
            return result
        except Exception as exc:
            logger.error("[infer] run_inference failed: %s", exc)
            return None
