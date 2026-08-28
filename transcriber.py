import logging
import os
import time
from typing import Any

import av
import gigaam

logger = logging.getLogger(__name__)

_model: Any | None = None


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def max_audio_duration_sec() -> float:
    return float(_env("MAX_AUDIO_DURATION_SEC", "25"))


def get_model() -> Any:
    global _model
    if _model is None:
        model_name = _env("GIGAAM_MODEL", "v3_e2e_rnnt")
        download_root = _env("GIGAAM_MODEL_PATH", "/models")
        device = _env("DEVICE", "cuda")
        fp16 = _env("FP16_ENCODER", "true").lower() in ("1", "true", "yes", "on")
        logger.info(
            "Loading GigaAM %s (device=%s, fp16_encoder=%s, cache=%s)",
            model_name,
            device,
            fp16,
            download_root,
        )
        _model = gigaam.load_model(
            model_name,
            device=device,
            download_root=download_root,
            fp16_encoder=fp16,
        )
        logger.info("GigaAM loaded on %s", next(_model.parameters()).device)
    return _model


def get_audio_duration(path: str) -> float:
    with av.open(path) as container:
        if container.duration is not None:
            return container.duration / 1_000_000.0
        total = 0.0
        stream = container.streams.audio[0]
        for packet in container.demux(stream):
            for frame in packet.decode():
                total += float(frame.samples) / float(frame.rate)
        return total


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    text = getattr(result, "text", None)
    if text is not None:
        return str(text).strip()
    return str(result).strip()


def transcribe_wav(path: str) -> dict[str, Any]:
    model = get_model()
    started = time.perf_counter()
    result = model.transcribe(path)
    processing_time = time.perf_counter() - started
    duration = get_audio_duration(path)
    return {
        "text": _result_text(result),
        "duration": duration,
        "processing_time": processing_time,
        "language": "ru",
    }


def warmup() -> None:
    import tempfile
    import wave

    if _env("WARMUP_ON_START", "true").lower() not in ("1", "true", "yes", "on"):
        logger.info("Warmup disabled")
        return

    logger.info("Running GigaAM warmup")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        warmup_path = tmp.name
    try:
        with wave.open(warmup_path, "w") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 16000)
        result = transcribe_wav(warmup_path)
        logger.info(
            "Warmup complete (processing_time=%.3fs)",
            result["processing_time"],
        )
    finally:
        if os.path.exists(warmup_path):
            os.unlink(warmup_path)
