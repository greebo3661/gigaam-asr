# Исследование моделей GigaAM-v3 на Hugging Face

**Дата:** 2026-06-04  
**Источник списка:** [поиск `gigaam-v3` на HF](https://huggingface.co/models?search=gigaam-v3) (17 репозиториев)  
**Родная модель:** [ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3)  
**Контекст деплоя:** Windows Server 2022, Docker, **NVIDIA GTX 1080 8 GB**, контракт API как у GigaAM (`POST /api/v1/transcribe-chunk`, 16 kHz mono WAV) — см. [VOICE-BOT-HANDOFF.md](../VOICE-BOT-HANDOFF.md), [whisper/TZ.txt](../whisper/TZ.txt).

---

## 1. Краткий вывод (TL;DR)

| Вопрос | Ответ |
|--------|--------|
| **В чём разница?** | Один и тот же семейный Conformer ~220M, но **5 вариантов весов** (ssl/ctc/rnnt/e2e_ctc/e2e_rnnt) × **разные рантаймы** (PyTorch, ONNX, OpenVINO, MLX, Sherpa) × **декодеры** (greedy, beam+KenLM, RNN-T joint). |
| **Что лучше по качеству?** | Для русского ASR: **`e2e_rnnt`** (ниже WER, пунктуация/нормализация) > **`rnnt`** > **`ctc`**; **`ctc + ngram LM`** ([waveletdeboshir](https://huggingface.co/waveletdeboshir/gigaam-v3-ctc-with-lm)) — компромисс без полного e2e. |
| **Что на GTX 1080 в Docker на Windows?** | **Практично:** [ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3) (`revision=ctc` или `e2e_ctc`) в PyTorch **FP32** (~1–2 GB VRAM) или [istupakov/gigaam-v3-onnx](https://huggingface.co/istupakov/gigaam-v3-onnx) + **onnxruntime-gpu** (CTC быстрее RNNT). **Не подходит:** все **MLX** (только Apple Silicon), **OpenVINO/NPU** (Intel), зеркала без отличий. |
| **Текущий прод у вас** | faster-whisper medium (~3.2 GB VRAM, int8_float32) — **проще и проверен**; GigaAM даёт **лучший WER на русском** (особенно callcenter/атипичная речь), но тяжелее по интеграции (torch 2.8, `trust_remote_code`, RNNT latency). |

---

## 2. Базовая архитектура (родная модель)

[GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3) — Conformer encoder (~220–240M параметров), обучен на русской речи (HuBERT-CTC pretrain, 700k часов для `ssl`).

### 2.1 Варианты весов (revisions) — не путать с HF-репозиториями

| Revision | Назначение | Выход | WER (средний по бенчмаркам Sber) |
|----------|------------|-------|----------------------------------|
| `ssl` | Энкодер, не ASR «из коробки» | — | — |
| `ctc` | ASR, CTC greedy | нижний регистр, без пунктуации | **9.2%** |
| `rnnt` | ASR, RNN-T | нижний регистр | **8.4%** |
| `e2e_ctc` | CTC + SentencePiece | пунктуация, нормализация | — |
| `e2e_rnnt` | RNN-T + SP | лучшее качество e2e | лучший среди линейки |

Официальная таблица WER (%): Open 3.0/2.6, Golos Farfield 4.5/3.9, Natural 7.8/6.9, Disordered 20.6/19.2, **Callcenter 10.3/9.5** (CTC/RNNT) vs Whisper 23.9.

Рекомендуемый стек upstream: `torch==2.8.0`, `torchaudio==2.8.0`, `transformers==4.57.1`, `trust_remote_code=True`.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("ai-sage/GigaAM-v3", revision="e2e_rnnt", trust_remote_code=True)
text = model.transcribe("example.wav")
```

---

## 3. Каталог всех 17 репозиториев

| # | Repo | Тип | Runtime | Вариант GigaAM | Загрузки/мес | Под Windows+1080 |
|---|------|-----|---------|----------------|--------------|------------------|
| 1 | [ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3) | **Оригинал** | PyTorch + HF custom code | все 5 revisions | 67.7k | **Да** (основной выбор) |
| 2 | [istupakov/gigaam-v3-onnx](https://huggingface.co/istupakov/gigaam-v3-onnx) | Квант/экспорт | `onnx-asr` (CPU/GPU hub) | ctc, rnnt, e2e_ctc, e2e_rnnt | 1.1k | **Да** (ONNX Runtime CUDA) |
| 3 | [thanrl/GigaAM-v3](https://huggingface.co/thanrl/GigaAM-v3) | Зеркало | PyTorch | копия карточки ai-sage | 4 | Да, но **бессмысленно** |
| 4 | [vhadzhykhanov/GigaAM-v3-fastapi-supported](https://huggingface.co/vhadzhykhanov/GigaAM-v3-fastapi-supported) | Обёртка? | PyTorch | без README | 0 | Неизвестно — **нет документации** |
| 5 | [waveletdeboshir/gigaam-v3-ctc-with-lm](https://huggingface.co/waveletdeboshir/gigaam-v3-ctc-with-lm) | Декодер+LM | PyTorch + kenlm + pyctcdecode | CTC + ngram beam | 566 | **Да**, если нужны таймстемпы/LM |
| 6 | [kiriyk/GigaAM-v3-onnx-rnnt-e2e](https://huggingface.co/kiriyk/GigaAM-v3-onnx-rnnt-e2e) | ONNX | onnx | e2e_rnnt | 1 | Да (эксперимент, 0 DL) |
| 7 | [al-bo/gigaam-v3-ctc-mlx](https://huggingface.co/al-bo/gigaam-v3-ctc-mlx) | Порт | **MLX** | CTC | 48 | **Нет** (Apple only) |
| 8 | [al-bo/gigaam-v3-rnnt-mlx](https://huggingface.co/al-bo/gigaam-v3-rnnt-mlx) | Порт | **MLX** | RNNT | 235 | **Нет** |
| 9 | [cnonim/gigaam-v3-e2e-rnnt-onnx](https://huggingface.co/cnonim/gigaam-v3-e2e-rnnt-onnx) | ONNX | onnxruntime | e2e_rnnt + INT8 | 0 | Да (ручная сборка пайплайна) |
| 10 | [Smirnov75/GigaAM-v3-sherpa-onnx](https://huggingface.co/Smirnov75/GigaAM-v3-sherpa-onnx) | ONNX | **sherpa-onnx** | ctc/rnnt/e2e_* | 41 | Да (C++/WS, другой API) |
| 11 | [aystream/GigaAM-v3-e2e-ctc-mlx](https://huggingface.co/aystream/GigaAM-v3-e2e-ctc-mlx) | Порт | MLX | e2e_ctc | 54 | **Нет** |
| 12 | [aystream/GigaAM-v3-e2e-rnnt-mlx](https://huggingface.co/aystream/GigaAM-v3-e2e-rnnt-mlx) | Порт | MLX | e2e_rnnt | 90 | **Нет** |
| 13 | [vpermilp/GigaAM-v3](https://huggingface.co/vpermilp/GigaAM-v3) | Зеркало | PyTorch + safetensors | как ai-sage | 42 | Да, но **дубль** |
| 14 | [Andrewsab/gigaam-v3-e2e-rnnt-ov](https://huggingface.co/Andrewsab/gigaam-v3-e2e-rnnt-ov) | OpenVINO IR | CPU / **Intel Arc iGPU** | e2e_rnnt FP16 | 3 | **Нет на NVIDIA** |
| 15 | [Andrewsab/gigaam-v3-e2e-rnnt-ov-npu](https://huggingface.co/Andrewsab/gigaam-v3-e2e-rnnt-ov-npu) | OpenVINO | **Intel NPU** | e2e_rnnt calibrated | 0 | **Нет** |
| 16 | [kruatech/gigaam-v3-mlx](https://huggingface.co/kruatech/gigaam-v3-mlx) | Бандл | MLX Swift (iOS/macOS) | e2e_rnnt | 0 | **Нет** |
| 17 | [VoiceScribe/gigaam-v3-e2e-rnnt-mlx](https://huggingface.co/VoiceScribe/gigaam-v3-e2e-rnnt-mlx) | Порт | MLX | e2e_rnnt | 124 | **Нет** |

---

## 4. Детальный разбор по группам

### 4.1 Оригинал и зеркала (PyTorch)

**ai-sage/GigaAM-v3** — единственный источник истины: все метрики, `model.transcribe()`, обновления revisions.

- **thanrl/GigaAM-v3**, **vpermilp/GigaAM-v3** — те же README/метрики; мало загрузок → риск устаревания, нет причин не использовать ai-sage.
- **vhadzhykhanov/GigaAM-v3-fastapi-supported** — теги `gigaam`, `custom_code`, **пустая model card**, 0 downloads → предположительно эксперимент под FastAPI; для прод не брать без аудита файлов.

### 4.2 CTC + языковая модель (лучший «классический» CTC)

**waveletdeboshir/gigaam-v3-ctc-with-lm** — неофициальная обёртка Transformers:

- База: GigaAM-v3 CTC + ngram LM из [bond005/wav2vec2-large-ru-golos-with-lm](https://huggingface.co/bond005/wav2vec2-large-ru-golos-with-lm).
- Декодирование: `pyctcdecode` + `kenlm`, beam_width/alpha/beta настраиваются.
- **Плюсы:** word-level timestamps (`MODEL_STRIDE = 40` ms), потенциально лучше greedy CTC на шумных данных.
- **Минусы:** beam 64 медленнее greedy; зависимости `kenlm` (сборка на Windows в Docker возможна, но капризна); не e2e-пунктуация.

### 4.3 ONNX-экосистема

| Repo | Инструмент | Как использовать |
|------|------------|------------------|
| **istupakov/gigaam-v3-onnx** | [onnx-asr](https://github.com/istupakov/onnx-asr) | `pip install onnx-asr[cpu,hub]` → `load_model("gigaam-v3-ctc")` и т.д. |
| **kiriyk/GigaAM-v3-onnx-rnnt-e2e** | сырой ONNX | только e2e_rnnt, минимальный README |
| **cnonim/gigaam-v3-e2e-rnnt-onnx** | onnxruntime | encoder/decoder/joint + **INT8** варианты, tokenizer.json |
| **Smirnov75/GigaAM-v3-sherpa-onnx** | sherpa-onnx | WebSocket server, NeMo-CTC или RNNT triple; **другой протокол**, не ваш `/transcribe-chunk` |

**onnx-asr** на Windows + CUDA: реалистичный путь для **низкой латентности** без полного PyTorch stack (проверить `onnxruntime-gpu` vs 1080 Pascal).

Сравнение из aystream (Apple M2, **не ваша платформа**, но порядок величин): MLX CTC 180× RT > PyTorch MPS RNNT 26× > **ONNX CPU CTC 12×**.

### 4.4 MLX (7 из 17 репозиториев — только Apple Silicon)

Все **al-bo/***, **aystream/***, **kruatech/***, **VoiceScribe/*** — MLX / Swift, метрики на M1–M4:

- CTC быстрее RNNT (al-bo: 139× vs 48× RT на M4).
- aystream e2e: CTC ~330× vs RNNT ~77× на 20 с chunk (M2 Max).
- kruatech — нативный iOS/macOS бандл, ~1.1 GB RAM, не Python.

**На GTX 1080 / Windows Docker: исключить полностью.**

### 4.5 OpenVINO (Intel)

**Andrewsab/gigaam-v3-e2e-rnnt-ov** — FP16 IR, ~425 MB encoder; лучшая цель **Intel Arc iGPU** (~520× RTFx encoder), CPU ~34×. На **NVIDIA не используется** как замена CUDA.

**Andrewsab/gigaam-v3-e2e-rnnt-ov-npu** — гибрид INT8/FP16 под Intel NPU, static shape 30 s, первая компиляция ~3.5 мин (с тюнингом свойств). Для Voice Scribe на Windows с NPU — не для 1080.

### 4.6 End-to-end vs «сырой» ASR

| Критерий | `ctc` / `rnnt` | `e2e_ctc` / `e2e_rnnt` |
|----------|----------------|-------------------------|
| Пунктуация | нет | да |
| Нормализация чисел/аббревиатур | нет | да |
| WER (официально) | RNNT лучше CTC | e2e_rnnt обычно лучший выбор для бота |
| Латентность | CTC < RNNT | RNNT joint — bottleneck (до ~50% времени) |
| VRAM (оценка FP32) | ~0.8–1.5 GB | ~1.0–2.0 GB (зависит от revision) |

Для **voice_bot** (ответ TTS + LLM) e2e-текст **предпочтителен** — меньше постобработки.

---

## 5. GTX 1080 8 GB + Windows Server 2022 + Docker

### 5.1 Ограничения Pascal (из вашего whisper-стека)

- **float16 в ctranslate2 не поддерживается** на 1080 → для Whisper выбран `int8_float32`.
- **PyTorch CUDA** на 1080: FP32 и FP16 (tensor cores нет, FP16 иногда медленнее/нестабильнее) — для GigaAM разумно **FP32** или **FP16 только после бенчмарка**.
- Одновременно: Asterisk Docker + whisper-asr ~3.2 GB → **свободно ~4–5 GB**; GigaAM ~1–2 GB **может уместиться**, но два тяжёлых GPU-сервиса на одной 1080 — риск OOM.

### 5.2 Оценка VRAM и RTF (порядок величин)

| Решение | VRAM (оценка) | RTF / 4 с chunk | Заметки |
|---------|---------------|-----------------|---------|
| faster-whisper medium int8 | ~3.2 GB | проверено у вас ~1.08 s / 4 s | уже в прод |
| GigaAM-v3 `ctc` PyTorch FP32 | ~1.0–1.5 GB | ожид. 0.3–0.8 s (зависит от torch) | проще RNNT |
| GigaAM-v3 `e2e_rnnt` PyTorch | ~1.5–2.5 GB | выше из-за joint loop | лучше WER |
| onnx-asr `gigaam-v3-ctc` GPU | ~0.5–1.5 GB | потенциально лучший RTF | меньше deps |
| waveletdeboshir + beam 64 | +CPU RAM для LM | decode CPU-heavy | latency spikes |

**Рекомендация для замены Whisper на GigaAM на том же хосте:**

1. **Пилот:** `ai-sage/GigaAM-v3` + `revision=e2e_ctc` (баланс качество/скорость) или `ctc` (макс. скорость).
2. **Если мало VRAM:** `istupakov/gigaam-v3-onnx` + `gigaam-v3-e2e-ctc` или `gigaam-v3-ctc` на ORT-GPU.
3. **Макс. качество (callcenter):** `e2e_rnnt` — замерить `processing_time` на 2–4 с chunks; если > SLA gateway — откат на e2e_ctc.
4. **Не использовать на 1080:** MLX, OpenVINO, Sherpa без адаптера API.

### 5.3 Docker-образ (концурентный дизайн)

```
nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
  → python 3.11
  → torch 2.8.0 + torchaudio 2.8.0 (cu124)
  → transformers 4.57.1
  → fastapi + uvicorn (контракт как whisper-asr)
  → HF_HOME=/models (volume)
  → revision=e2e_ctc по умолчанию
```

Альтернатива легче по зависимостям:

```
onnxruntime-gpu + onnx-asr[hub]  → model gigaam-v3-e2e-ctc
```

Healthcheck: `GET /health`; warmup при старте — один `transcribe` на 1 с silence.

---

## 6. Что лучше — матрица решений

| Сценарий | Лучший выбор |
|----------|----------------|
| Макс. WER на русском, callcenter | `ai-sage/GigaAM-v3` **`e2e_rnnt`** |
| Бот: пунктуация + скорость | **`e2e_ctc`** (ai-sage или onnx-asr) |
| Минимальная латентность chunk 2–4 с | **`ctc`** ONNX или PyTorch |
| Word timestamps для диаризации | **waveletdeboshir/gigaam-v3-ctc-with-lm** |
| Прод без PyTorch / меньший образ | **istupakov/gigaam-v3-onnx** |
| Streaming WebSocket, C++ | **Smirnov75/GigaAM-v3-sherpa-onnx** (новый сервис) |
| iPhone / Mac offline | **kruatech/gigaam-v3-mlx** |
| Intel laptop с NPU | **Andrewsab** ov + ov-npu |
| Уже работает, минимум риска | **оставить faster-whisper** до завершения A/B |

---

## 7. Дополнительный анализ (вопросы, которые вы не задавали)

### 7.1 Замена Whisper vs дополнение

- Whisper medium: сильный **мультиязычный** baseline, слабее на **русском callcenter** (WER ~24% в таблице GigaAM vs ~10% GigaAM).
- GigaAM не даёт встроенного **language detect** как whisper — у вас `language=ru` фиксирован → ок.
- **Silero VAD** из whisper-стека можно оставить: резать тишину до GigaAM → меньше hallucinations и GPU time.

### 7.2 Риски `trust_remote_code`

Официальная модель тянет custom Python с Hub — политика безопасности: пиннинг revision, checksum, свой Docker build без `--trust-remote-code` в проде невозможен без форка кода → **зафиксировать commit HF** и сканировать обновления.

### 7.3 Лицензия и прод

MIT на ai-sage и большинстве портов — коммерческое использование допустимо; проверить **голосовые данные** на обучение fine-tune отдельно.

### 7.4 Зеркала и supply chain

thanrl, vpermilp — дубли с малым числом загрузок: риск **не тех весов**. Всегда `ai-sage/GigaAM-v3` + явный `revision`.

### 7.5 Совместимость с Linux gateway

Контракт уже под GigaAM ([TZ](../whisper/TZ.txt)): миграция whisper → gigaam — **смена только Windows-сервиса**, gateway (`GigaamClient`) без изменений если сохранить JSON и 16 kHz WAV.

### 7.6 A/B метрики для приёмки

- WER/CER на 50–100 записях callcenter (ваш домен).
- p95 `processing_time` на chunk 4 s (SLA < 1.5 s?).
- VRAM peak под нагрузкой 2 параллельных звонков (если планируется).
- Empty text rate на тишине (с VAD).

### 7.7 Почему 7/17 репозиториев — «шум» для вашей задачи

Почти **41%** списка — MLX/OpenVINO/зеркала; реальная развилка для Windows GPU: **3 ветки** (PyTorch ai-sage, onnx-asr, sherpa).

### 7.8 Hugging Face plugin

С плагином HF: `hf download ai-sage/GigaAM-v3 --revision e2e_ctc` в volume `d:\docker\gigaam\model\` — без ручного копирования.

---

## 8. Рекомендуемый план внедрения (если решите мигрировать с Whisper)

1. Скачать `ai-sage/GigaAM-v3` (`e2e_ctc`) в `d:\docker\gigaam\model\`.
2. Новый сервис `gigaam-asr` на порту **5002** (или 5003 + переключение gateway) — копия API из whisper.
3. Прогон `run-acceptance-tests.ps1` аналог с эталонными WAV.
4. A/B с whisper на 3003 в тестовом режиме (`ASR_URL` env).
5. При успехе — отключить whisper GPU или вынести на второй хост.

---

## 9. Ссылки

- [Поиск моделей gigaam-v3](https://huggingface.co/models?search=gigaam-v3)
- [ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3)
- [istupakov/gigaam-v3-onnx](https://huggingface.co/istupakov/gigaam-v3-onnx)
- [waveletdeboshir/gigaam-v3-ctc-with-lm](https://huggingface.co/waveletdeboshir/gigaam-v3-ctc-with-lm)
- [Smirnov75/GigaAM-v3-sherpa-onnx](https://huggingface.co/Smirnov75/GigaAM-v3-sherpa-onnx)
- [GigaAM GitHub](https://github.com/salute-developers/GigaAM)
- [Paper arXiv:2506.01192](https://arxiv.org/abs/2506.01192)

---

*Отчёт подготовлен по model cards и HF API на 2026-06-04. Числа RTF/VRAM на 1080 требуют локального бенчмарка — в таблице 5.2 указаны оценки.*
