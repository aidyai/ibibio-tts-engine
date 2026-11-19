# orpheus tts on an fp8 tensorrt-llm engine, served with a streaming http api
# build the engine once (quantize, build_engine), then deploy

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import struct
import time

import modal

CUDA_VERSION     = "12.4.1"
TORCH_VERSION    = "2.4.0"
TRTLLM_VERSION   = "0.12.0"
TRANSFORMERS_VER = "transformers>=4.45.0,<4.48.0"
STARLETTE_VER    = "1.0.0"
UVICORN_VER      = "0.34.0"
PYTHON_VERSION   = "3.10"

MODEL_REPO   = "your-org/your-bilingual-orpheus-model"   # EDIT: your fine-tuned Orpheus checkpoint on the Hugging Face Hub

TRTLLM_CKPT_DIR = "/model-weights/orpheus-trtllm-ckpt"
ENGINE_DIR      = "/model-weights/orpheus-trtllm-engine"

TRTLLM_SRC        = "/opt/trtllm-src"
QUANTIZE_SCRIPT   = f"{TRTLLM_SRC}/examples/quantization/quantize.py"

# snac frames decoded per chunk
SNAC_CHUNK_FRAMES = 8

# seconds
SYNTHESIS_TIMEOUT_S = 120

# audio cache entries per container
AUDIO_CACHE_SIZE = 256

tts_image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu22.04",
        add_python=PYTHON_VERSION,
    )
    .apt_install(
        "git", "wget", "libsndfile1", "ffmpeg",
        "build-essential", "cmake",
        "openmpi-bin", "libopenmpi-dev",
    )
    .pip_install(
        f"torch=={TORCH_VERSION}+cu124",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        f"tensorrt-llm=={TRTLLM_VERSION}",
        "pynvml<12",
        "cuda-python==12.4.0",
        extra_options=(
            "--extra-index-url https://pypi.nvidia.com "
            "--extra-index-url https://download.pytorch.org/whl/cu124"
        ),
    )
    .run_commands(
        f"git clone --depth 1 --branch v{TRTLLM_VERSION} "
        f"https://github.com/NVIDIA/TensorRT-LLM.git {TRTLLM_SRC}"
    )
    .run_commands(
        f"pip install -r {TRTLLM_SRC}/examples/quantization/requirements.txt || true"
    )
    .pip_install(
        TRANSFORMERS_VER,
        "huggingface_hub[hf_transfer]",
        "hf_transfer",
        "snac",
        "soundfile",
        "numpy>=1.26.0",
        "datasets",
    )
    .pip_install(
        f"starlette=={STARLETTE_VER}",
        f"uvicorn[standard]=={UVICORN_VER}",
        "python-multipart>=0.0.20",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

volume    = modal.Volume.from_name("orpheus-tts-vol", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


def _get_hf_token() -> str:
    import os
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_API_TOKEN")
    )
    if not token:
        raise RuntimeError(
            "No HuggingFace token found. "
            "Make sure your secret contains HF_TOKEN."
        )
    return token


app = modal.App(
    name="orpheus-tts",
    image=tts_image,
    volumes={"/model-weights": volume},
    secrets=[hf_secret],
)


@app.function(gpu="H100", timeout=7200, volumes={"/model-weights": volume}, secrets=[hf_secret])
def quantize():
    # fp8 runs natively on h100 (sm90). bfloat16 is only the calibration dtype
    import os
    os.makedirs(TRTLLM_CKPT_DIR, exist_ok=True)

    hf_token = _get_hf_token()
    os.environ["HF_TOKEN"] = hf_token

    if not os.path.isfile(QUANTIZE_SCRIPT):
        raise FileNotFoundError(
            f"quantize.py not found at {QUANTIZE_SCRIPT}\n"
            "The git clone step in the image build may have failed."
        )

    _run([
        "python", QUANTIZE_SCRIPT,
        "--model_dir",  MODEL_REPO,
        "--dtype",      "bfloat16",
        "--qformat",    "fp8",
        "--calib_size", "512",
        "--output_dir", TRTLLM_CKPT_DIR,
        "--tp_size",    "1",
    ])

    volume.commit()
    print(f">>> FP8 TRT-LLM checkpoint saved to {TRTLLM_CKPT_DIR}")


@app.function(gpu="H100", timeout=7200, volumes={"/model-weights": volume}, secrets=[hf_secret])
def build_engine():
    # plugin flags take float16/bfloat16, fp8 comes from the quantized checkpoint
    import os
    os.makedirs(ENGINE_DIR, exist_ok=True)

    _run([
        "trtllm-build",
        "--checkpoint_dir",         TRTLLM_CKPT_DIR,
        "--output_dir",             ENGINE_DIR,
        "--gemm_plugin",            "fp8",        # fp8 is valid for gemm
        "--gpt_attention_plugin",   "bfloat16",  # matches the checkpoint dtype
        "--max_batch_size",         "64",
        "--max_input_len",          "1024",
        "--max_seq_len",            "5120",
        "--max_num_tokens",         "16384",
        "--paged_kv_cache",         "enable",
        "--remove_input_padding",   "enable",
        "--use_paged_context_fmha", "enable",
        "--context_fmha",           "enable",
        "--multiple_profiles",      "enable",
        "--workers",                "1",
    ])

    volume.commit()
    print(f">>> TRT-LLM engine written to {ENGINE_DIR}")


def _run(cmd: list[str]):
    import subprocess
    print(f">>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, capture_output=False, check=True)


class _AudioCache:
    def __init__(self, maxsize: int = AUDIO_CACHE_SIZE):
        self._cache: dict[str, bytes] = {}
        self._maxsize = maxsize

    @staticmethod
    def key(text: str, voice: str) -> str:
        return hashlib.sha256(f"{voice}:{text}".encode()).hexdigest()

    def get(self, k: str) -> bytes | None:
        return self._cache.get(k)

    def set(self, k: str, data: bytes) -> None:
        if len(self._cache) >= self._maxsize:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[k] = data


@app.cls(
    gpu="H100",
    scaledown_window=300,
    min_containers=0,
    volumes={"/model-weights": volume},
    secrets=[hf_secret],
    timeout=300,
)
@modal.concurrent(max_inputs=2)
class OrpheusTTS:

    @modal.enter()
    def load(self):
        import torch
        from tensorrt_llm.runtime import ModelRunnerCpp
        from snac import SNAC
        from transformers import AutoTokenizer

        self.device = torch.device("cuda")
        self.torch  = torch
        self._cache = _AudioCache(maxsize=AUDIO_CACHE_SIZE)

        hf_token = _get_hf_token()

        print(">>> Loading tokenizer ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_REPO,
            token=hf_token,
        )

        print(">>> Loading TRT-LLM engine ...")
        self.runner = ModelRunnerCpp.from_dir(
            engine_dir=ENGINE_DIR,
            rank=0,
            max_output_len=4096,
            kv_cache_free_gpu_memory_fraction=0.5,
            enable_chunked_context=True,
        )

        print(">>> Loading SNAC decoder onto GPU ...")
        self.snac = SNAC.from_pretrained(
            "hubertsiuzdak/snac_24khz",
            token=hf_token,
        ).to(self.device).eval()

        print(">>> OrpheusTTS ready.")

    @modal.method()
    def synthesize(self, text: str, voice: str = "iniabasi") -> bytes:
        import torch

        if len(text) > 2000:
            raise ValueError("Text exceeds 2000 character limit.")

        cache_key = self._cache.key(text, voice)
        cached = self._cache.get(cache_key)
        if cached is not None:
            print(f">>> Cache hit for key {cache_key[:8]}…")
            return cached

        prompt = f"{voice}: {text}"
        SOH, EOT, EOH = 128259, 128009, 128260

        text_ids  = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = torch.cat([
            torch.tensor([[SOH]]),
            text_ids,
            torch.tensor([[EOT, EOH]]),
        ], dim=1).to(self.device)

        with torch.inference_mode():
            output_ids = self.runner.generate(
                batch_input_ids=[input_ids[0]],
                max_new_tokens=4000,
                end_id=128262,
                pad_id=self.tokenizer.pad_token_id,
                temperature=0.6,
                repetition_penalty=1.1,
                do_sample=True,
            )[0][0]

        generated    = output_ids[input_ids.shape[1]:].tolist()
        audio_tokens = [t for t in generated if 128266 <= t < 128266 + 7 * 4096]

        if not audio_tokens:
            raise RuntimeError("No audio tokens generated.")

        audio = self._decode_snac(audio_tokens)
        wav   = _audio_to_wav(audio.cpu().numpy(), sample_rate=24000)

        self._cache.set(cache_key, wav)
        return wav

    def _decode_snac(self, audio_tokens: list[int]):
        import torch

        layer_1, layer_2, layer_3 = [], [], []
        for i in range((len(audio_tokens) + 1) // 7):
            try:
                v0 = max(0, min(4095, audio_tokens[7*i]   - 128266))
                v1 = max(0, min(4095, audio_tokens[7*i+1] - 128266 - 4096))
                v2 = max(0, min(4095, audio_tokens[7*i+2] - 128266 - 8192))
                v3 = max(0, min(4095, audio_tokens[7*i+3] - 128266 - 12288))
                v4 = max(0, min(4095, audio_tokens[7*i+4] - 128266 - 16384))
                v5 = max(0, min(4095, audio_tokens[7*i+5] - 128266 - 20480))
                v6 = max(0, min(4095, audio_tokens[7*i+6] - 128266 - 24576))
                layer_1.append(v0)
                layer_2.append(v1)
                layer_3.append(v2)
                layer_3.append(v3)
                layer_2.append(v4)
                layer_3.append(v5)
                layer_3.append(v6)
            except IndexError:
                break

        codes = [
            torch.tensor(layer_1, dtype=torch.long, device=self.device).unsqueeze(0),
            torch.tensor(layer_2, dtype=torch.long, device=self.device).unsqueeze(0),
            torch.tensor(layer_3, dtype=torch.long, device=self.device).unsqueeze(0),
        ]

        with torch.inference_mode():
            audio = self.snac.decode(codes)

        return audio.detach().squeeze()

    @modal.method()
    def synthesize_stream(self, text: str, voice: str = "iniabasi"):
        # yields a wav header first, then raw int16 24khz mono pcm
        import torch

        if len(text) > 2000:
            raise ValueError("Text exceeds 2000 character limit.")

        prompt    = f"{voice}: {text}"
        SOH, EOT, EOH = 128259, 128009, 128260

        text_ids  = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = torch.cat([
            torch.tensor([[SOH]]),
            text_ids,
            torch.tensor([[EOT, EOH]]),
        ], dim=1).to(self.device)

        yield _streaming_wav_header(sample_rate=24000)

        END_ID = 128262
        PAD_ID = self.tokenizer.pad_token_id
        n_prompt     = input_ids.shape[1]
        pos          = n_prompt          # next position to read in the runner's output row
        token_buffer: list[int] = []
        step         = 0
        n_audio      = 0
        t0           = time.perf_counter()
        first_chunk  = True

        with torch.inference_mode():
            for output in self.runner.generate(
                batch_input_ids=[input_ids[0]],
                max_new_tokens=4000,
                end_id=END_ID,
                pad_id=PAD_ID,
                temperature=0.6,
                repetition_penalty=1.1,
                do_sample=True,
                streaming=True,
            ):
                row = output[0][0]
                if step < 2:
                    print(f">>> stream step {step}: type={type(output).__name__} "
                          f"row_len={row.shape[0]} prompt_len={n_prompt}")
                step += 1

                # the row is padded at the end, read only the new part
                # and drop the padding so pos only moves over real tokens
                new = row[pos:].tolist()
                while new and new[-1] in (END_ID, PAD_ID):
                    new.pop()
                pos += len(new)

                for t in new:
                    if 128266 <= t < 128266 + 7 * 4096:
                        token_buffer.append(t)
                        n_audio += 1

                while len(token_buffer) >= 7 * SNAC_CHUNK_FRAMES:
                    chunk        = token_buffer[:7 * SNAC_CHUNK_FRAMES]
                    token_buffer = token_buffer[7 * SNAC_CHUNK_FRAMES:]
                    if first_chunk:
                        print(f">>> first audio chunk after {time.perf_counter() - t0:.2f}s (step {step})")
                        first_chunk = False
                    yield self._decode_snac_chunk(chunk)

        remaining_frames = len(token_buffer) // 7
        if remaining_frames > 0:
            chunk = token_buffer[:7 * remaining_frames]
            yield self._decode_snac_chunk(chunk)
        print(f">>> stream done: {step} steps, {n_audio} audio tokens, {time.perf_counter() - t0:.2f}s")

    def _decode_snac_chunk(self, tokens: list[int]) -> bytes:
        import torch
        import numpy as np

        n_frames = len(tokens) // 7
        layer_1, layer_2, layer_3 = [], [], []

        for i in range(n_frames):
            v0 = max(0, min(4095, tokens[7*i]   - 128266))
            v1 = max(0, min(4095, tokens[7*i+1] - 128266 - 4096))
            v2 = max(0, min(4095, tokens[7*i+2] - 128266 - 8192))
            v3 = max(0, min(4095, tokens[7*i+3] - 128266 - 12288))
            v4 = max(0, min(4095, tokens[7*i+4] - 128266 - 16384))
            v5 = max(0, min(4095, tokens[7*i+5] - 128266 - 20480))
            v6 = max(0, min(4095, tokens[7*i+6] - 128266 - 24576))
            layer_1.append(v0)
            layer_2.extend([v1, v4])
            layer_3.extend([v2, v3, v5, v6])

        codes = [
            torch.tensor([layer_1], dtype=torch.long, device=self.device),
            torch.tensor([layer_2], dtype=torch.long, device=self.device),
            torch.tensor([layer_3], dtype=torch.long, device=self.device),
        ]

        with torch.inference_mode():
            audio = self.snac.decode(codes).detach().squeeze().cpu().numpy()

        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16).tobytes()


def _err(code: str, message: str, *, retryable: bool = False, status: int = 400):
    from starlette.responses import JSONResponse
    return JSONResponse(
        {"error": {"code": code, "message": message, "retryable": retryable}},
        status_code=status,
    )


def _make_asgi_app():
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.requests import Request
    from starlette.responses import Response, JSONResponse, StreamingResponse
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware

    SUPPORTED_VOICES = ["abas", "eskor", "iniabasi", "ama", "elise"]
    MAX_TEXT_LEN     = 2000
    tts              = OrpheusTTS()

    async def _parse_json(request: Request):
        if "application/json" not in request.headers.get("content-type", ""):
            return None, _err("UNSUPPORTED_MEDIA_TYPE",
                               "Content-Type must be application/json",
                               status=415)
        try:
            body = await request.body()
            if len(body) > 16_384:
                return None, _err("PAYLOAD_TOO_LARGE", "Request body too large",
                                   status=413)
            return json.loads(body), None
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, _err("INVALID_JSON", "Body is not valid JSON", status=400)

    def _validate(data: dict, voices: list[str], max_len: int):
        text  = data.get("text", "").strip()
        voice = data.get("voice", "iniabasi").strip().lower()
        if not text:
            return None, None, _err("MISSING_TEXT", "text field is required", status=422)
        if len(text) > max_len:
            return None, None, _err("TEXT_TOO_LONG",
                                     f"text exceeds {max_len} characters", status=422)
        if voice not in voices:
            return None, None, _err("INVALID_VOICE",
                                     f"voice must be one of: {', '.join(voices)}", status=422)
        return text, voice, None

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "timestamp": time.time()})

    async def warm(request: Request) -> JSONResponse:
        return JSONResponse({"status": "warm", "timestamp": time.time()})

    async def voices(request: Request) -> JSONResponse:
        return JSONResponse({"voices": SUPPORTED_VOICES})

    async def synthesize(request: Request) -> Response:
        idempotency_key = request.headers.get("Idempotency-Key", "")
        data, err = await _parse_json(request)
        if err:
            return err
        text, voice, err = _validate(data, SUPPORTED_VOICES, MAX_TEXT_LEN)
        if err:
            return err
        try:
            wav_bytes = await asyncio.wait_for(
                tts.synthesize.remote.aio(text, voice),
                timeout=SYNTHESIS_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            return _err("SYNTHESIS_TIMEOUT",
                         f"Synthesis exceeded {SYNTHESIS_TIMEOUT_S}s limit",
                         retryable=True, status=504)
        except ValueError as exc:
            return _err("INVALID_INPUT", str(exc), status=422)
        except Exception as exc:
            return _err("SYNTHESIS_FAILED", str(exc), retryable=True, status=500)

        headers = {
            "Content-Disposition": 'attachment; filename="speech.wav"',
            "X-Voice":             voice,
            "X-Text-Length":       str(len(text)),
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return Response(content=wav_bytes, media_type="audio/wav", headers=headers)

    async def synthesize_stream(request: Request) -> Response:
        idempotency_key = request.headers.get("Idempotency-Key", "")
        data, err = await _parse_json(request)
        if err:
            return err
        text, voice, err = _validate(data, SUPPORTED_VOICES, MAX_TEXT_LEN)
        if err:
            return err

        async def audio_generator():
            try:
                async for pcm_chunk in tts.synthesize_stream.remote_gen.aio(text, voice):
                    yield pcm_chunk
            except Exception as exc:
                print(f"!!! Stream error: {exc}")

        headers = {
            "X-Voice":                voice,
            "X-Content-Type-Options": "nosniff",
            "Cache-Control":          "no-cache",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return StreamingResponse(audio_generator(), media_type="audio/wav", headers=headers)

    routes = [
        Route("/health",            health,            methods=["GET"]),
        Route("/warm",              warm,              methods=["GET"]),
        Route("/voices",            voices,            methods=["GET"]),
        Route("/synthesize",        synthesize,        methods=["POST"]),
        Route("/synthesize/stream", synthesize_stream, methods=["POST"]),
    ]
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "Authorization", "Idempotency-Key"],
        ),
    ]
    return Starlette(routes=routes, middleware=middleware)


@app.function(scaledown_window=60, secrets=[hf_secret])
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="orpheus-tts")
def web_app():
    return _make_asgi_app()


def _audio_to_wav(audio_np, sample_rate: int = 24000) -> bytes:
    import numpy as np
    audio_np  = np.clip(audio_np, -1.0, 1.0)
    pcm       = (audio_np * 32767).astype(np.int16)
    n_samples = len(pcm)
    buf       = io.BytesIO()
    data_size = n_samples * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()


def _streaming_wav_header(sample_rate: int = 24000) -> bytes:
    h  = b"RIFF"
    h += struct.pack("<I", 0xFFFFFFFF)
    h += b"WAVE"
    h += b"fmt "
    h += struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    h += b"data"
    h += struct.pack("<I", 0xFFFFFFFF)
    return h