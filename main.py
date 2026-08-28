import logging
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
import uvicorn

from transcriber import (
    get_audio_duration,
    get_model,
    max_audio_duration_sec,
    transcribe_wav,
    warmup,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    warmup()
    yield


app = FastAPI(title="GigaAM ASR Service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/transcribe-chunk")
async def transcribe_chunk(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty file")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    suffix = ".wav"
    if file.filename and "." in file.filename:
        suffix = os.path.splitext(file.filename)[1] or suffix

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        duration = get_audio_duration(tmp_path)
        max_duration = max_audio_duration_sec()
        if duration > max_duration:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long (max {max_duration:.0f}s, got {duration:.1f}s)",
            )

        result = transcribe_wav(tmp_path)
        return {
            "text": result["text"],
            "duration": result["duration"],
            "processing_time": result["processing_time"],
            "language": result.get("language"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5002"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
