import json
import logging
import traceback
from pathlib import Path

import numpy as np
import onnxruntime as ort

from inference.device_utils import resolve_onnx_providers

logger = logging.getLogger(__name__)


class GameOnnxModel:
    def __init__(self, model_dir: Path, *, requested_device: str | None = None):
        self.model_dir = Path(model_dir)
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"GAME ONNX config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.samplerate = int(self.config["samplerate"])
        self.timestep = float(self.config["timestep"])
        self.loop = bool(self.config.get("loop", True))
        self.languages = dict(self.config.get("languages") or {})
        self.requested_device = str(requested_device)
        self.provider_name, providers = self._resolve_providers(self.requested_device)
        self._sessions = self._load_sessions(providers)

    @staticmethod
    def _resolve_providers(device: str) -> tuple[str, list[str]]:
        return resolve_onnx_providers(device, label="GAME ONNX")

    def _load_sessions(self, providers: list[str]) -> dict[str, ort.InferenceSession]:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False

        filenames = {
            "encoder": "encoder.onnx",
            "segmenter": "segmenter.onnx",
            "estimator": "estimator.onnx",
            "dur2bd": "dur2bd.onnx",
            "bd2dur": "bd2dur.onnx",
        }
        sessions = {}
        for key, filename in filenames.items():
            path = self.model_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Required GAME ONNX file not found: {path}")
            logger.info(f"Loading GAME sub-model '{key}' from '{path}'...")
            sessions[key] = ort.InferenceSession(str(path), sess_options=sess_options, providers=providers)
        return sessions

    def shutdown(self) -> None:
        self._sessions.clear()

    release = shutdown

    def infer_batch(
        self,
        *,
        waveforms: np.ndarray,
        durations: np.ndarray,
        known_durations: np.ndarray | None,
        boundary_threshold: float,
        boundary_radius: int,
        score_threshold: float,
        language: int = 0,
        ts: list[float] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        ts = list(ts or [])
        batch_size = int(waveforms.shape[0])

        try:
            enc_out = self._sessions["encoder"].run(
                None,
                {
                    "waveform": waveforms.astype(np.float32, copy=False),
                    "duration": durations.astype(np.float32, copy=False),
                },
            )
            x_seg = np.asarray(enc_out[0], dtype=np.float32)
            x_est = np.asarray(enc_out[1], dtype=np.float32)
            mask_t = np.asarray(enc_out[2], dtype=bool)
        except Exception as e:
            logger.error(f"GAME sub-model 'encoder' failed: {repr(e)}\n{traceback.format_exc()}")
            raise

        if known_durations is not None and known_durations.size > 0:
            known_boundaries = np.zeros_like(mask_t, dtype=bool)
            for idx in range(batch_size):
                valid = known_durations[idx][known_durations[idx] > 0]
                if len(valid) == 0:
                    continue
                try:
                    kb = self._sessions["dur2bd"].run(
                        None,
                        {
                            "durations": valid[np.newaxis, :].astype(np.float32, copy=False),
                            "maskT": mask_t[idx : idx + 1].astype(bool, copy=False),
                        },
                    )[0]
                    known_boundaries[idx : idx + 1] = np.asarray(kb, dtype=bool)
                except Exception as e:
                    logger.error(f"GAME sub-model 'dur2bd' failed: {repr(e)}\n{traceback.format_exc()}")
                    raise
        else:
            known_boundaries = np.zeros_like(mask_t, dtype=bool)

        boundaries = known_boundaries.copy()
        lang_arr = np.full((batch_size,), int(language), dtype=np.int64)

        try:
            if self.loop and ts:
                for t in ts:
                    boundaries = self._sessions["segmenter"].run(
                        None,
                        {
                            "x_seg": x_seg.astype(np.float32, copy=False),
                            "language": lang_arr,
                            "known_boundaries": known_boundaries.astype(bool, copy=False),
                            "prev_boundaries": boundaries.astype(bool, copy=False),
                            "t": np.full((batch_size,), float(t), dtype=np.float32),
                            "maskT": mask_t.astype(bool, copy=False),
                            "threshold": np.array(boundary_threshold, dtype=np.float32),
                            "radius": np.array(boundary_radius, dtype=np.int64),
                        },
                    )[0]
                    boundaries = np.asarray(boundaries, dtype=bool)
            else:
                boundaries = self._sessions["segmenter"].run(
                    None,
                    {
                        "x_seg": x_seg.astype(np.float32, copy=False),
                        "language": lang_arr,
                        "known_boundaries": known_boundaries.astype(bool, copy=False),
                        "prev_boundaries": boundaries.astype(bool, copy=False),
                        "t": np.zeros((batch_size,), dtype=np.float32),
                        "maskT": mask_t.astype(bool, copy=False),
                        "threshold": np.array(boundary_threshold, dtype=np.float32),
                        "radius": np.array(boundary_radius, dtype=np.int64),
                    },
                )[0]
                boundaries = np.asarray(boundaries, dtype=bool)
        except Exception as e:
            logger.error(f"GAME sub-model 'segmenter' failed: {repr(e)}\n{traceback.format_exc()}")
            raise

        try:
            durations_out, mask_n = self._sessions["bd2dur"].run(
                None,
                {
                    "boundaries": boundaries.astype(bool, copy=False),
                    "maskT": mask_t.astype(bool, copy=False),
                },
            )
            durations_out = np.asarray(durations_out, dtype=np.float32)
            mask_n = np.asarray(mask_n, dtype=bool)
        except Exception as e:
            logger.error(f"GAME sub-model 'bd2dur' failed: {repr(e)}\n{traceback.format_exc()}")
            raise

        try:
            presence, scores = self._sessions["estimator"].run(
                None,
                {
                    "x_est": x_est.astype(np.float32, copy=False),
                    "boundaries": boundaries.astype(bool, copy=False),
                    "maskT": mask_t.astype(bool, copy=False),
                    "maskN": mask_n.astype(bool, copy=False),
                    "threshold": np.array(score_threshold, dtype=np.float32),
                },
            )
            presence = np.asarray(presence, dtype=bool)
            scores = np.asarray(scores, dtype=np.float32)
        except Exception as e:
            logger.error(f"GAME sub-model 'estimator' failed: {repr(e)}\n{traceback.format_exc()}")
            raise

        results = []
        for idx in range(batch_size):
            valid = mask_n[idx].astype(bool)
            results.append(
                (
                    durations_out[idx][valid],
                    presence[idx][valid],
                    scores[idx][valid],
                )
            )
        return results
