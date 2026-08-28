# GigaAM ASR — результат приёмки (TZ §6)

**Дата:** 2026-06-04 13:23:07  
**Хост:** Windows Server 2022, 192.168.148.109  
**URL ASR:** http://192.168.148.109:5002  
**Модель:** ai-sage/GigaAM-v3 e2e_rnnt (d:\docker\gigaam\model)  
**Контейнер:** gigaam-asr  

---

## Чеклист

| # | Пункт | Статус | Детали |
|---|-------|--------|--------|| 1 | nvidia-smi GTX 1080 | **PASS** | /=========================================+========================+======================/ / /   0  NVIDIA GeForce ... |
| 2 | VRAM GigaAM loaded | **PASS** | 2334 MiB, 8192 MiB |
| 3 | GET /health localhost | **PASS** | {"status":"ok"} |
| 4 | GET /health 192.168.148.109 | **PASS** | {"status":"ok"} |
| 5 | test.mp3 source | **PASS** | duration_sec=6964.845688 path=d:\docker\test.mp3 |
| 6 | POST transcribe-chunk WAV 4s ru | **PASS** | {"text":"╨б╨╝╨╛╤В╤А╨╕. ╨Я╨╛ ╤В╨╡╨║╤Г╤Й╨╡╨╣ ╤А╨╡╨░╨╗╨╕╨╖╨░╤Ж╨╕╨╕","duration":4.0,"processing_time":0.5088917000011861,... |
| 7 | POST empty file -> 400 | **PASS** | http_code=400 |
| 8 | Firewall 5002 from 192.168.149.0/24 | **PASS** | TCP 5002 allow from 192.168.149.0/24 |
| 9 | ASR 192.168.148.109:5002 TCP | **PASS** | TcpTestSucceeded=True |

---

## Итог: 9 / 9

- Чанк: 	est-chunk-16k.wav (4 с, 16 kHz mono)
- Текст: «╨б╨╝╨╛╤В╤А╨╕. ╨Я╨╛ ╤В╨╡╨║╤Г╤Й╨╡╨╣ ╤А╨╡╨░╨╗╨╕╨╖╨░╤Ж╨╕╨╕»
- processing_time: 0.508891700001186 с

### Linux gateway

```env
ASR_BASE_URL=http://192.168.148.109:5002
APP_MODE=voice_bot
```
