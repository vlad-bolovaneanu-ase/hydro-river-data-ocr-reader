#!/usr/bin/env python3
"""Shared defaults for DeepSeek OCR2 server/client scripts."""

from pathlib import Path

# DeepSeek OCR2 source tree (contains deepseek_ocr2.py and process/).
DEEPSEEK_PROJECT_DIR = "/home/jovyan/libs/DeepSeek-OCR-2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm"

# Model and generation defaults.
MODEL_PATH = "deepseek-ai/DeepSeek-OCR-2"
FREE_OCR_PROMPT = "<image>\nFree OCR."
GROUNDING_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
DEFAULT_MODE = "free_ocr"
DEFAULT_IGNORE_EOS = False
CROP_MODE = True
DTYPE = "bfloat16"
MAX_MODEL_LEN = 8192
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.75
ENFORCE_EAGER = False

# vLLM service endpoint.
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8008
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# Client store layout.
STORE_DIR = str(Path.cwd() / "deepseek_ocr_store")
IMAGES_SUBDIR = "images"
GENERATIONS_SUBDIR = "generations"

# Network defaults.
HTTP_TIMEOUT_SECONDS = 120

# Hydro bulletin defaults.
DEFAULT_ROOT_URL = "https://www.hidro.ro/bulletin_type/prognoza-hidrologica-pentru-rauri/"
