# Bilingual English-Ibibio TTS Engine

Text-to-speech that speaks **both Ibibio and English**. Ibibio is a Nigerian language with almost no digital resources, so the model is built to handle it alongside English in the same voice. A fine-tuned [Orpheus](https://github.com/canopyai/Orpheus-TTS) model, quantized to FP8 and served with TensorRT-LLM on a production-grade NVIDIA H100 (Hopper) cloud GPU, with streaming audio.

This repository holds the serving code and the measured results. It is the production TTS path of **USEM AI**, the first bilingual English-Ibibio AI platform (translation, transcription and audio generation APIs), where it was built and is used.

## Why this exists

Most speech technology is built for a handful of languages. Ibibio has millions of speakers in southern Nigeria and, when we started, no digital speech corpus at all. Without data there is no model, and without a model there is no voice for the language in any app. Ibibio speakers also move between Ibibio and English all the time, so the voice has to handle both.

So the work started with data, not code: about **80 hours of English-Ibibio speech** and **205,000 parallel sentence pairs**, collected and curated from scratch. The TTS model is a LoRA fine-tune of Orpheus on that bilingual speech, so one model reads Ibibio and English. This repo is the part that makes it usable: a fast, cheap serving layer.

## What is in here

| Path | What it is |
|---|---|
| `deploy/serve.py` | The whole service: FP8 quantization, TensorRT-LLM engine build, SNAC audio decoder, HTTP API with streaming, on a serverless cloud GPU |
| `benchmarks/samples.py` | Generates the audio samples in `docs/samples/` from a live endpoint |
| `docs/` | Sample page and audio files |
| `benchmarks/latency.py` | The test script used for the numbers below. Standard library only |

The model weights and the training data are **not** in this repository.

## How it works

```
text ──> Orpheus (fine-tuned, FP8, TensorRT-LLM) ──> audio tokens ──> SNAC decoder ──> 24 kHz audio
                          │                                                 │
                    paged KV cache                                  decoded in 8-frame chunks
                                                                    and streamed as they appear
```

- **FP8 on H100.** The model is quantized to FP8 and compiled into a TensorRT-LLM engine (paged KV cache, chunked context).
- **Streaming.** Audio tokens are decoded in small chunks and sent as they are produced, so the caller hears audio before the sentence is finished.
- **Scale to zero.** No GPU runs when nobody is calling it (`min_containers=0`).
- **Cache.** Repeated phrases are answered from an in-memory cache.
- **API.** `POST /synthesize` returns a full WAV; `POST /synthesize/stream` streams 16-bit, 24 kHz mono audio after a WAV header.

## Results

Measured in November 2026 on a production-grade NVIDIA H100 (Hopper) cloud GPU, with the FP8 engine from this repo.

| Metric | Result | How it was measured |
|---|---|---|
| Languages | **Ibibio and English** | One bilingual model |
| Naturalness (MOS), Ibibio | **2.95 / 5** | Mean opinion score from 4 native Ibibio speakers rating the model's Ibibio speech on a 1 to 5 scale. English was not rated |
| Cold start | **about 29 s** | First request after the container had scaled to zero: 29.4 s, of which 27.5 s was starting the container and loading the engine and decoder |
| Streaming, first audio chunk (warm) | **0.18 s** | Measured inside the container from the start of generation. Same value on 9 consecutive warm requests |
| Streaming, whole sentence (warm) | **3 to 9.6 s of audio in 0.9 to 2.5 s** (about 0.26x real time) | 10 requests, 5 Ibibio and mixed sentences, 2 voices, 3 to 9.6 s of audio each |

Client-side time to first audio (over the network) is reported by `benchmarks/latency.py`; the table from that run goes here.

## Listen

Raw output from the live endpoint, two voices, five sentences, no editing:

**[Play the samples](https://aidyai.github.io/ibibio-tts-engine/)**

The files are also in [`docs/samples/`](docs/samples) (`ama_*.wav`, `iniabasi_*.wav`) and were produced with `benchmarks/samples.py`.

## Run it yourself

You need an account on a serverless GPU cloud with H100 access, a Hugging Face token that can read your model, and your own fine-tuned Orpheus checkpoint.

```bash
pip install modal
modal token set --token-id <id> --token-secret <secret>
modal secret create huggingface-secret HF_TOKEN=<your read token>

# 1. set MODEL_REPO at the top of deploy/serve.py, then:
modal run deploy/serve.py::quantize        # once
modal run deploy/serve.py::build_engine    # once
modal deploy deploy/serve.py
```

Then measure it:

```bash
python benchmarks/latency.py https://<your-endpoint>            # warm: p50 / p95 table
python benchmarks/latency.py https://<your-endpoint> --cold     # one request after 6+ minutes idle
```

## Known limits

- **Cold start is about 30 s.** The first request after idle waits for the GPU to load. `/warm` only wakes the web layer, not the GPU.
- **No authentication and open CORS** in this example. Add a key check before exposing it publicly.
- **MOS 2.95 is honest, not great, and the panel is small.** It comes from 4 native Ibibio speakers rating Ibibio speech only, so treat it as an indication, not a precise score. English output has not been rated yet. More data and a larger listening test, covering both languages, are the next step.
- **One request class.** Each GPU container handles 2 requests at once; there is no load-test result in this repo yet.
- The in-memory cache is per container.

## Credits

Built on [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS), [SNAC](https://github.com/hubertsiuzdak/snac) and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).

## License

Code is MIT (see `LICENSE`). Model weights and training data are not included and are not covered by this license.

Author: [Idara Samuel Osu](https://github.com/aidyai)
