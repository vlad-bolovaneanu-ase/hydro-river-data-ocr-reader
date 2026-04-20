#!/usr/bin/env python3
"""Serve DeepSeek OCR2 via a small FastAPI + vLLM service."""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageOps
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry

from deepseek_ocr2_service_config import (
    CROP_MODE,
    DEFAULT_IGNORE_EOS,
    DEFAULT_MODE,
    DTYPE,
    ENFORCE_EAGER,
    FREE_OCR_PROMPT,
    GROUNDING_PROMPT,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MODEL_PATH,
    SERVER_HOST,
    SERVER_PORT,
    TENSOR_PARALLEL_SIZE,
    DEEPSEEK_PROJECT_DIR,
)


@dataclass
class ServerSettings:
    project_dir: str
    model_path: str
    host: str
    port: int
    dtype: str
    max_model_len: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    enforce_eager: bool
    crop_mode: bool
    default_mode: str
    default_ignore_eos: bool
    cuda_visible_devices: Optional[str]


class GenerateRequest(BaseModel):
    image_path: str
    mode: Optional[str] = None
    prompt: Optional[str] = None
    max_tokens: int = 8192
    temperature: float = 0.0
    ignore_eos: Optional[bool] = None
    request_id: Optional[str] = None


class GenerateResponse(BaseModel):
    request_id: str
    mode: str
    ignore_eos: bool
    text: str
    elapsed_ms: int


def prompt_for_mode(mode: str) -> str:
    if mode == "free_ocr":
        return FREE_OCR_PROMPT
    if mode == "grounding":
        return GROUNDING_PROMPT
    raise ValueError(f"Unsupported mode: {mode}")


def _configure_env(cuda_visible_devices: Optional[str]) -> None:
    os.environ.setdefault("VLLM_USE_V1", "0")
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    ptxas = "/usr/local/cuda-11.8/bin/ptxas"
    if torch.version.cuda == "11.8" and Path(ptxas).exists():
        os.environ.setdefault("TRITON_PTXAS_PATH", ptxas)


def _ensure_project_importable(project_dir: str) -> None:
    p = str(Path(project_dir).expanduser().resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_deepseek_components(project_dir: str) -> tuple[Any, Any]:
    _ensure_project_importable(project_dir)
    from deepseek_ocr2 import DeepseekOCR2ForCausalLM  # pylint: disable=import-outside-toplevel
    from process.image_process import DeepseekOCR2Processor  # pylint: disable=import-outside-toplevel

    return DeepseekOCR2ForCausalLM, DeepseekOCR2Processor


def _load_image_rgb(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def _build_engine(settings: ServerSettings) -> tuple[AsyncLLMEngine, Any]:
    model_cls, processor_cls = _load_deepseek_components(settings.project_dir)
    try:
        ModelRegistry.register_model("DeepseekOCR2ForCausalLM", model_cls)
    except Exception:
        # Already registered in this process.
        pass

    engine_args = AsyncEngineArgs(
        model=settings.model_path,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        dtype=settings.dtype,
        max_model_len=settings.max_model_len,
        enforce_eager=settings.enforce_eager,
        trust_remote_code=True,
        tensor_parallel_size=settings.tensor_parallel_size,
        gpu_memory_utilization=settings.gpu_memory_utilization,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    processor = processor_cls()
    return engine, processor


def create_app(settings: ServerSettings) -> FastAPI:
    app = FastAPI(title="DeepSeek OCR2 vLLM Service")
    app.state.settings = settings
    app.state.engine = None
    app.state.processor = None

    @app.on_event("startup")
    async def _startup() -> None:
        _configure_env(app.state.settings.cuda_visible_devices)
        engine, processor = _build_engine(app.state.settings)
        app.state.engine = engine
        app.state.processor = processor

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest) -> GenerateResponse:
        image_path = Path(req.image_path).expanduser().resolve()
        if not image_path.exists():
            raise HTTPException(status_code=400, detail=f"Image not found: {image_path}")

        mode = req.mode or app.state.settings.default_mode
        try:
            prompt = req.prompt if req.prompt is not None else prompt_for_mode(mode)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        ignore_eos = app.state.settings.default_ignore_eos if req.ignore_eos is None else req.ignore_eos
        image = _load_image_rgb(image_path)

        if "<image>" in prompt:
            image_features = app.state.processor.tokenize_with_images(
                images=[image],
                bos=True,
                eos=True,
                cropping=app.state.settings.crop_mode,
            )
            payload = {
                "prompt": prompt,
                "multi_modal_data": {"image": image_features},
            }
        else:
            payload = {"prompt": prompt}

        sampling_params = SamplingParams(
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            skip_special_tokens=False,
            ignore_eos=ignore_eos,
        )

        request_id = req.request_id or f"req-{uuid.uuid4().hex}"
        start = time.time()
        final_text = ""

        async for request_output in app.state.engine.generate(payload, sampling_params, request_id):
            if request_output.outputs:
                final_text = request_output.outputs[0].text

        elapsed_ms = int((time.time() - start) * 1000)
        return GenerateResponse(
            request_id=request_id,
            mode=mode,
            ignore_eos=ignore_eos,
            text=final_text,
            elapsed_ms=elapsed_ms,
        )

    return app


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve DeepSeek OCR2 via vLLM")
    p.add_argument("--project-dir", default=DEEPSEEK_PROJECT_DIR)
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--host", default=SERVER_HOST)
    p.add_argument("--port", type=int, default=SERVER_PORT)
    p.add_argument("--dtype", default=DTYPE)
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    p.add_argument("--tensor-parallel-size", type=int, default=TENSOR_PARALLEL_SIZE)
    p.add_argument("--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION)
    p.add_argument("--enforce-eager", action="store_true", default=ENFORCE_EAGER)
    p.add_argument("--crop-mode", action="store_true", default=CROP_MODE)
    p.add_argument("--default-mode", choices=["free_ocr", "grounding"], default=DEFAULT_MODE)
    p.add_argument("--default-ignore-eos", action="store_true", default=DEFAULT_IGNORE_EOS)
    p.add_argument("--cuda-visible-devices", default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    settings = ServerSettings(
        project_dir=args.project_dir,
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        crop_mode=args.crop_mode,
        default_mode=args.default_mode,
        default_ignore_eos=args.default_ignore_eos,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
