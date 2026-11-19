# Ibibio TTS Engine

<sub>Bilingual English-Ibibio text to speech, in production at USEM AI. Fine-tuned Orpheus, FP8 on TensorRT-LLM, H100, streaming output.</sub>

### 🔊 Hear it

<sub>Raw output, no editing. Click ▶, then press play on the page that opens.</sub>

| Say | Voice 1 | Voice 2 |
|---|---|---|
| Ided owowaan ado akappa iwad | [▶ Ama](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/ama_short.wav) | [▶ Iniabasi](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/iniabasi_short.wav) |
| Akpọ ifʌkkọ ado anie ewaana ye ọkpọ esịt ke nkọk itak isịn | [▶ Ama](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/ama_medium.wav) | [▶ Iniabasi](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/iniabasi_medium.wav) |
| Sake ado ukọd isọñ idʌñ Japanese enamma eto ibu-ibu adesi | [▶ Ama](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/ama_long.wav) | [▶ Iniabasi](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/iniabasi_long.wav) |
| Akaanawaan ado amaase ntǝkayịn daña ebre mbre, atie ndobo-ndobo ke ñkpọitie inemesịt. | [▶ Ama](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/ama_longer.wav) | [▶ Iniabasi](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/iniabasi_longer.wav) |
| Kwa ndubi, eka ado ama edikood mme ñwed nnọ nditọ ammọ. But grandma is soo nice and I love her for that | [▶ Ama](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/ama_mixed.wav) | [▶ Iniabasi](https://github.com/aidyai/ibibio-tts-engine/blob/main/docs/samples/iniabasi_mixed.wav) |

The last row is one sentence that switches from Ibibio to English.

### Numbers

<sub>Measured November 2026 on an NVIDIA H100 (Hopper) cloud GPU.</sub>

| | |
|---|---|
| Languages | Ibibio and English, one model |
| Naturalness (MOS), Ibibio | 2.95 / 5, from 4 native speakers. English not rated |
| First audio chunk (warm) | 0.18 s |
| Generation speed (warm) | 3 to 9.6 s of audio in 0.9 to 2.5 s, about 0.26x real time |
| Cold start | about 30 s (scales to zero) |

### How it works

<sub>

- Orpheus TTS fine-tuned with LoRA on about 80 hours of English-Ibibio speech
- FP8 quantization, compiled to a TensorRT-LLM engine
- SNAC decodes audio tokens to 24 kHz sound in small chunks, so playback starts before the sentence is done
- `POST /synthesize` returns a WAV, `POST /synthesize/stream` streams audio
- Files: `deploy/serve.py` (the service), `benchmarks/latency.py` (timing), `benchmarks/samples.py` (makes the clips)
- Weights and training data are not included

</sub>

### Limits

<sub>

- No auth and open CORS in this example. Add a key check before exposing it.
- MOS panel is small (4 raters). Treat it as an indication.
- No load test yet. Each container handles 2 requests at once.

MIT licensed. Built on [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS), [SNAC](https://github.com/hubertsiuzdak/snac) and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).

</sub>
