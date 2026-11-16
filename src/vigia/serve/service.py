"""FastAPI alert service (plan Phase 6). Needs the ``serve`` extra.

The service keeps, per camera, one ``vigia.eval.alerting.AlertPolicy`` (the same class the
evaluation harness uses) and one frame buffer of the last ``scorer.context_frames`` frames.

Endpoints:

- ``POST /cameras/{camera_id}/frames``: multipart image + ``timestamp`` (ISO 8601 or epoch
  seconds). Decodes the image to RGB uint8, appends it to the camera's buffer, scores the
  buffer with the ``OnnxScorer`` and keeps the score of the newest frame, steps the
  camera's policy with ``(timestamp_s, score)``, and returns ``{"score": float,
  "alert": bool}``. A timestamp not later than the camera's previous one is rejected
  with 409.
- ``GET /cameras/{camera_id}/alerts``: the alerts fired for that camera since start-up,
  as ``[{"timestamp": ..., "score": ...}, ...]``; 404 for an unknown camera.
- ``GET /healthz``: ``{"status": "ok", "model": scorer.name}`` once the model is loaded.

State is in memory only; restarting the service resets every policy and buffer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app(config: Mapping[str, Any]) -> FastAPI:
    """Build the FastAPI app described in the module docstring.

    ``config`` is a resolved config with a ``serve`` section (ONNX path, alert policy
    parameters as in ``configs/eval``). FastAPI and ONNX Runtime are imported here.
    """
    raise NotImplementedError("Phase 6: alert service (create_app)")
