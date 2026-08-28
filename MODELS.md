# Веса GigaAM

В git нет `.ckpt` и `.bin`. Каталог `model/` на диске — кэш Docker-тома.

## Что качает сервис

При старте пакет `gigaam` кладёт в `GIGAAM_MODEL_PATH` (по умолчанию `./model` → `/models`):

- `v3_e2e_rnnt.ckpt`
- `v3_e2e_rnnt_tokenizer.model`

Источник: CDN Sber / репозиторий [salute-developers/GigaAM](https://github.com/salute-developers/GigaAM).  
Карточка: [ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3).

## Повторная загрузка

Удалите файлы в `model/` (кроме заметок) и перезапустите контейнер.

Снимок Hugging Face Transformers (`pytorch_model.bin`) этому сервису не нужен.
