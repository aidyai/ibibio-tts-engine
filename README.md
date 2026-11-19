# Ibibio TTS Engine

**Production text to speech for Ibibio, a Nigerian language with almost no speech technology. One bilingual model speaks Ibibio and English, streams audio as it is generated, and runs on an H100 with FP8 TensorRT-LLM.**

It is the voice layer of [USEM AI](https://github.com/aidyai), the bilingual English-Ibibio AI platform.

## Hear it

Raw output from the live endpoint, no editing. Click ▶, then press play on the page that opens.

1. *Ided owowaan ado akappa iwad*  
   [▶ Ama](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/ama_short.wav) · [▶ Iniabasi](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/iniabasi_short.wav)

2. *Akpọ ifʌkkọ ado anie ewaana ye ọkpọ esịt ke nkọk itak isịn*  
   [▶ Ama](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/ama_medium.wav) · [▶ Iniabasi](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/iniabasi_medium.wav)

3. *Sake ado ukọd isọñ idʌñ Japanese enamma eto ibu-ibu adesi*  
   [▶ Ama](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/ama_long.wav) · [▶ Iniabasi](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/iniabasi_long.wav)

4. *Akaanawaan ado amaase ntǝkayịn daña ebre mbre, atie ndobo-ndobo ke ñkpọitie inemesịt.*  
   [▶ Ama](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/ama_longer.wav) · [▶ Iniabasi](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/iniabasi_longer.wav)

5. *Kwa ndubi, eka ado ama edikood mme ñwed nnọ nditọ ammọ. But grandma is soo nice and I love her for that*  
   [▶ Ama](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/ama_mixed.wav) · [▶ Iniabasi](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/iniabasi_mixed.wav)

Sentence 5 switches from Ibibio to English mid-sentence.

## Results

Measured November 2026 on an NVIDIA H100 (Hopper) cloud GPU.

| Metric | Result |
|---|---|
| Languages | Ibibio and English, one model, 5 voices |
| Naturalness (MOS), Ibibio | **2.95 / 5** from 4 native speakers (English not rated) |
| Cold start | **about 30 s** (33.6 s on the first deploy, 29.4 s after scale to zero) |
| Cold start, engine and decoder load | 30.1 s of the 33.6 s |
| First audio chunk, warm | **0.18 s** (same on 9 consecutive requests) |
| First audio chunk, first request after load | 0.49 s to 0.53 s |
| Warm, full file (`/synthesize`) | 8.4 s of audio in 3.5 s |
| Warm, streaming, same sentence | 8.4 s of audio in 2.73 s (about 0.33x real time) |
| Warm, streaming, 10 requests, 2 voices | 3 to 9.6 s of audio in 0.9 to 2.5 s (about **0.26x real time**) |
| Output | 24 kHz, 16-bit mono WAV |
| Training data | about 80 hours of English-Ibibio speech |

All warm requests were uncached. Timings are measured inside the container.

## Streaming

`POST /synthesize/stream` sends a WAV header and then raw audio as soon as the first 8 frames are decoded, so playback starts in under a fifth of a second while the rest is still being generated. `POST /synthesize` returns the whole file.

```bash
curl -X POST https://YOUR-ENDPOINT/synthesize/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Ided owowaan ado akappa iwad", "voice": "ama"}' --output out.wav
```

## How it is built

- Orpheus TTS fine-tuned with LoRA on English-Ibibio speech
- FP8 quantization, compiled to a TensorRT-LLM engine with paged KV cache and chunked context
- SNAC decodes audio tokens to sound in 8-frame chunks
- Scales to zero when idle, caches repeated phrases

## Repository

| | |
|---|---|
| `deploy/serve.py` | The whole service: quantize, build engine, serve |
| `benchmarks/latency.py` | Timing script behind the numbers |
| `benchmarks/samples.py` | Generates the clips in `docs/samples/` |

Model weights and training data are not included.

## Limits

- No auth and open CORS in this example. Add a key check before exposing it.
- The MOS panel is small (4 raters). Treat it as an indication.
- No load test yet. Each container handles 2 requests at once.

MIT licensed. Built on [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS), [SNAC](https://github.com/hubertsiuzdak/snac) and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).
