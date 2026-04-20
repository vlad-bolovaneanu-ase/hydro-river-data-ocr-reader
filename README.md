# Download and parse Romanian hydro river levels

This repository offers two approaches:

1. A python script based on `pytesseract`, which runs on CPU and has been tested on only 1 image.

2. If a suitable GPU is available (at least 16 GB of RAM with quantization; over 20 GB recommended for the full model), `DeepSeek-OCR2` can be used with vLLM for more accurate results.

## Python script

- Install [tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html) on your platform of choice. Also make sure to install the Romanian extension, e.g. on Debian-based distributions `sudo apt-get install tesseract-ron`.

- Create a python environment. Anaconda and its smaller alternatives, such as miniconda, are recommended. Then run: 

```bash
conda create --file tesseract_conda_env.yaml
```

- Run `hydro_parser_tesseract script with one image to test. Example below:

```bash
python hydro_parser_tesseract.py \
    --buleltin-url "https://www.hidro.ro/bulletin/prognoza-hidrologica-pentru-rauri-in-intervalul-19-04-2026-ora-07-00-20-04-2026-ora-07-00/" \
    --rivers Arges Vedea Dambovita \
    -o results.xlsx
```

More pages can be automatically parsed and all images extracted:

```bash
python hydro_parser_tesseract.py \
    --root-url "https://www.hidro.ro/bulletin_type/prognoza-hidrologica-pentru-rauri/"
    --rivers Arges Vedea Dambovita \
    --max-pages 3 \
    -o results.xlsx
```

## DeepSeek-OCR

A good starting point is a [Google Colab](https://colab.research.google.com/), which offers NVIDIA T4 GPU's for free. In this case, you will need to use a quantized model version for inference. 

The tutorial below is designed for GPU's with at least 20 GB RAM and compute capability >= 7.5 which ideally support [flash attention](https://github.com/dao-ailab/flash-attention) (Ampere or newer).

- Follow instructions from the [official DeepSeek guide](https://github.com/deepseek-ai/DeepSeek-OCR-2#vllm-inference). An already tested environment can be installed with 

```bash
conda create --file deepseek_conda_env.yaml
```

thereby skipping most steps. The repository must still be cloned. If their test sample succedes, the next step can begin. 

- Download all images in a store. A more robust method which handles repeated download requests with backoff retries.

```bash
python hydro_image_finder_downloader.py \
    --root-url "https://www.hidro.ro/bulletin_type/prognoza-hidrologica-pentru-rauri/" \
    --base-delay 0.6 \
    --max-delay 20 \
    --max-retries 15
```

- Start the model vLLM server. Replace `your_lib_dir` with the path where the the DeepSeek repo was cloned.

```bash
python deepseek_ocr2_vllm_server.py \
    --project-dir <your_lib_dir>/DeepSeek-OCR-2/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm \
    --host 127.0.0.1 \
    --port 8008 \
    --default-mode grounding
```

- Start the client script, which will feed each image to the server and save results incrementally in multiple csv files:

```bash
python deepseek_ocr2_client.py \
    --image-manifest deepseek_ocr_store/image_manifest.json \
    --store-dir deepseek_ocr_store \
    --server-url http://127.0.0.1:8008 \
    --mode grounding \
    --workers 1
```

