# gigaam-asr

Docker-сервис русской речи на [GigaAM](https://github.com/salute-developers/GigaAM) `v3_e2e_rnnt`.  
HTTP-контракт: `POST /api/v1/transcribe-chunk` (WAV, чанки до 25 с).

Проверено на **NVIDIA GTX 1080 8 GB** (CUDA 12.4, PyTorch 2.6). Веса **не входят** в репозиторий — скачиваются при первом старте.

## Что внутри

| | |
|---|---|
| Образ | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` + пакет `gigaam` |
| API | `GET /health`, `POST /api/v1/transcribe-chunk` |
| Порт | `5002` |
| VRAM | ~2.3 GB |
| Язык | русский e2e RNNT |

`transformers` и `pyannote.audio` для коротких чанков не нужны.

## Требования

- Docker с NVIDIA Container Toolkit
- GPU с CUDA 12.x (или `DEVICE=cpu`, медленно)
- Свободный TCP **5002**

## Быстрый старт

```bash
git clone https://github.com/greebo3661/gigaam-asr.git
cd gigaam-asr
cp .env.example .env          # при необходимости
docker compose up -d --build
docker compose logs -f gigaam-asr
```

Первый запуск качает `v3_e2e_rnnt.ckpt` и tokenizer в `./model/` (том). Это может занять несколько минут.

```bash
curl -s http://127.0.0.1:5002/health
curl -s -X POST http://127.0.0.1:5002/api/v1/transcribe-chunk -F "file=@chunk-16k.wav"
```

Приёмка (PowerShell):

```powershell
pwsh -NoProfile -File scripts/run-acceptance-tests.ps1
```

## Переменные

См. [.env.example](.env.example).

| Env | Default | Смысл |
|-----|---------|--------|
| `GIGAAM_MODEL` | `v3_e2e_rnnt` | имя модели или путь к `.ckpt` |
| `GIGAAM_MODEL_PATH` | `/models` | кэш весов (volume `./model`) |
| `DEVICE` | `cuda` | `cuda` / `cpu` |
| `FP16_ENCODER` | `true` | FP16 encoder на GPU |
| `PORT` | `5002` | HTTP |
| `MAX_AUDIO_DURATION_SEC` | `25` | лимит длины чанка |

## Веса и лицензии

Веса GigaAM качает сам пакет с CDN Sber. Условия модели — у [salute-developers/GigaAM](https://github.com/salute-developers/GigaAM) и [ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3).  
Код обёртки в этом репозитории — [MIT](LICENSE).

Подробнее: [MODELS.md](MODELS.md).

## Совместимость портов

Другой ASR на том же хосте (например Whisper) тоже часто слушает **5002**. Одновременно не поднимать.

## Ссылки

- [GigaAM](https://github.com/salute-developers/GigaAM)
- [ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3)
- [TZ.txt](TZ.txt) — исходное ТЗ лабораторной приёмки (хосты в тексте — пример)
- [GIGAAM-V3-MODELS-RESEARCH.md](GIGAAM-V3-MODELS-RESEARCH.md) — сравнение карточек на Hugging Face
