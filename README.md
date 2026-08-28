# GigaAM ASR Service (Docker + GTX 1080)

HTTP-сервис на официальном пакете **[GigaAM](https://github.com/salute-developers/GigaAM)** (`v3_e2e_rnnt`). Контракт API — как у Linux gateway (`POST /api/v1/transcribe-chunk`).

## Почему такой стек

По [pyproject.toml](https://github.com/salute-developers/GigaAM/blob/main/pyproject.toml) Sber:

| Компонент | Для чанков ≤25 с | Для longform |
|-----------|------------------|--------------|
| `pip install gigaam` + `[torch]` | **да** | — |
| `transformers` + HF snapshot | опционально | — |
| `pyannote.audio` | **не нужен** | только `[longform]` |

На **Ubuntu 22.04 + CUDA 12.4 + GTX 1080** в PyTorch index `cu124` есть **torch 2.6.0** (2.8 для cu124 нет). Это совместимо с `torch>=2.6` из GigaAM.

Веса: при первом старте в `./model/` скачиваются `v3_e2e_rnnt.ckpt` и tokenizer с CDN Sber. Снимок HuggingFace (`pytorch_model.bin`, `modeling_gigaam.py`) **не используется** этим сервисом — можно оставить как архив или удалить.

## Быстрый старт

```powershell
# Порт 5002: остановите whisper-asr если он занят
cd D:\docker\whisper; docker compose down

cd D:\docker\gigaam
docker compose build --no-cache
docker compose up -d
docker compose logs -f gigaam-asr
```

```powershell
curl.exe -s http://127.0.0.1:5002/health
curl.exe -s -X POST http://127.0.0.1:5002/api/v1/transcribe-chunk -F "file=@test-chunk-16k.wav"
pwsh -NoProfile -File scripts\run-acceptance-tests.ps1
```

Firewall (admin): `scripts\add-firewall-rule.ps1`

## Переменные

| Env | Default | Описание |
|-----|---------|----------|
| `GIGAAM_MODEL` | `v3_e2e_rnnt` | Имя модели или путь к `.ckpt` |
| `GIGAAM_MODEL_PATH` | `/models` | Кэш весов (volume `./model`) |
| `DEVICE` | `cuda` | `cuda` / `cpu` |
| `FP16_ENCODER` | `true` | FP16 encoder на GPU (рекомендуется Sber) |
| `PORT` | `5002` | HTTP |

## Ссылки

- [ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3) — HF-карточка
- [TZ.txt](TZ.txt) — приёмка
- [GIGAAM-V3-MODELS-RESEARCH.md](GIGAAM-V3-MODELS-RESEARCH.md) — сравнение 17 HF-репо
