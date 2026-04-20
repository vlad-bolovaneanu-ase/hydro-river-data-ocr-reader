
#!/usr/bin/env python3
"""
Parser pentru buletinele INHGA / hidro.ro:
- citește pagini de buletin sau imagini locale
- descarcă imaginile relevante din buletin
- aplică OCR pe tabelul hidrologic
- filtrează doar râurile dorite
- exportă CSV / XLSX

IMPORTANT
---------
1) Scriptul este gândit pentru formatul tabelului din:
   "Prognoza hidrologică pentru râuri"
   unde imaginea conține tabelul mare cu 50 de rânduri.
2) OCR-ul folosește pytesseract, deci ai nevoie și de binarul Tesseract instalat.
3) Layout-ul este tratat ca "template consistent". Dacă site-ul schimbă structura
   imaginii sau a tabelului, pot fi necesare ajustări ale pragurilor/curățării.

Exemple:
---------
1) Procesează o singură imagine locală:
   python hidro_parser.py --image pg2_2-4.jpg --rivers Arges Vedea Dambovita -o iesire.csv

2) Procesează un singur buletin:
   python hidro_parser.py --bulletin-url "https://www.hidro.ro/bulletin/..." -o iesire.xlsx

3) Procesează lista de buletine din root:
   python hidro_parser.py --root-url "https://www.hidro.ro/bulletin_type/prognoza-hidrologica-pentru-rauri/" \
       --max-pages 3 --rivers Arges Vedea Dambovita -o rauri.xlsx

4) Dacă tesseract nu e în PATH:
    python hidro_parser.py --image pg2_2-4.jpg --tesseract-cmd "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import io
import math
import os
import re
import shlex
import subprocess
import sys
import time
import hashlib
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]
import numpy as np
import pandas as pd
try:
    import pytesseract
except ImportError:
    pytesseract = None  # type: ignore[assignment]
import requests
from bs4 import BeautifulSoup
from PIL import Image

DEFAULT_ROOT_URL = "https://www.hidro.ro/bulletin_type/prognoza-hidrologica-pentru-rauri/"
DEFAULT_RIVERS = ["Arges", "Vedea", "Dambovita"]
DEFAULT_OCR_TEXT_LANG = "ron"
DEFAULT_OCR_NUM_LANG = "eng"
DEFAULT_BACKEND = "tesseract"

# Limbile OCR pot fi suprascrise din CLI.
OCR_TEXT_LANG = DEFAULT_OCR_TEXT_LANG
OCR_NUM_LANG = DEFAULT_OCR_NUM_LANG

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"
    )
}

# Coloanele utile din tabelul mare.
OUTPUT_COLUMNS = [
    "data",
    "bulletin_url",
    "image_url",
    "raul",
    "statia",
    "jud",
    "CA_cm",
    "CI_cm",
    "CP_cm",
    "Qmed_apr_m3_s",
    "diag_data",
    "diag_H_cm",
    "diag_Q_m3_s",
    "diag_dH_cm",
    "diag_G_P",
    "prog_data",
    "prog_H_cm",
    "prog_Q_m3_s",
    "prog_dH_cm",
    "prog_G_P",
]

# Normalizare OCR pentru râuri țintă.
RIVER_CANONICAL_MAP = {
    "arges": "Arges",
    "argeş": "Arges",
    "argeș": "Arges",
    "arges,": "Arges",
    "vedea": "Vedea",
    "dambovita": "Dambovita",
    "dîmbovita": "Dambovita",
    "dimbovita": "Dambovita",
    "dâmbovita": "Dambovita",
    "dâmbovița": "Dambovita",
    "dambovița": "Dambovita",
}

@dataclasses.dataclass
class BulletinImage:
    bulletin_url: str
    bulletin_date: str
    image_url: str


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def ensure_tesseract_dependencies() -> None:
    if cv2 is None:
        raise RuntimeError("Lipsește pachetul opencv-python (cv2). Instalează-l sau folosește --backend deepseek.")
    if pytesseract is None:
        raise RuntimeError("Lipsește pachetul pytesseract. Instalează-l sau folosește --backend deepseek.")


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def strip_diacritics_for_match(s: str) -> str:
    repl = str.maketrans({
        "ă": "a", "â": "a", "î": "i", "ș": "s", "ş": "s", "ț": "t", "ţ": "t",
        "Ă": "A", "Â": "A", "Î": "I", "Ș": "S", "Ş": "S", "Ț": "T", "Ţ": "T",
    })
    return s.translate(repl)


def canonical_river_name(s: str) -> str:
    raw = normalize_spaces(s)
    key = strip_diacritics_for_match(raw).lower()
    key = key.replace("â", "a").replace("ă", "a").replace("î", "i")
    key = re.sub(r"[^a-z0-9 ]+", "", key).strip()
    key_compact = key.replace(" ", "")

    # OCR confusion frecvent: "arzes" in loc de "arges".
    if key == "arzes":
        key = "arges"

    if "dambovita" in key_compact or "dimbovita" in key_compact:
        return "Dambovita"
    if "vedea" in key_compact:
        return "Vedea"
    if "arges" in key_compact or "arzes" in key_compact:
        return "Arges"

    return RIVER_CANONICAL_MAP.get(key, raw)


def normalize_river_for_match(s: str) -> str:
    s = strip_diacritics_for_match(normalize_spaces(s)).lower()
    s = s.replace("0", "o").replace("1", "l")
    s = re.sub(r"[^a-z]", "", s)
    return s


def river_match_score(text: str, target: str) -> float:
    t = normalize_river_for_match(text)
    u = normalize_river_for_match(target)
    if not t or not u:
        return 0.0
    if t == u:
        return 1.0

    score = SequenceMatcher(None, t, u).ratio()
    if u in t or (len(t) >= 4 and t in u):
        score = max(score, 0.92)
    return score


def clean_station_text(s: str) -> str:
    s = normalize_spaces(s)
    s = s.replace("|", "I")
    return s


def clean_county_text(s: str) -> str:
    s = normalize_spaces(s).upper()
    s = re.sub(r"[^A-Z]", "", s)
    return s


def clean_num(s: str) -> Optional[str]:
    s = s.strip()
    if not s:
        return None
    # În OCR apar frecvent caractere greșite.
    s = s.replace("O", "0").replace("o", "0")
    s = s.replace("l", "1").replace("I", "1")
    s = s.replace("—", "-").replace("–", "-")
    s = s.replace(" ", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9\.\-]", "", s)
    if s in {"", "-", ".", "-.", ".-"}:
        return None
    return s


def num_or_none(s: str) -> Optional[float]:
    s2 = clean_num(s)
    if s2 is None:
        return None
    try:
        return float(s2)
    except ValueError:
        return None


def ocr_cell(
    image: np.ndarray,
    lang: str = "eng",
    psm: int = 7,
    whitelist: Optional[str] = None,
    digits_only: bool = False,
) -> str:
    ensure_tesseract_dependencies()
    config = f"--oem 3 --psm {psm}"
    if whitelist:
        config += f' -c tessedit_char_whitelist="{whitelist}"'
    elif digits_only:
        config += ' -c tessedit_char_whitelist="0123456789-., "'
    pil = Image.fromarray(image)
    txt = pytesseract.image_to_string(pil, lang=lang, config=config)
    return normalize_spaces(txt)


def ocr_num_from_region(table_bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> Optional[float]:
    h = table_bgr.shape[0]

    # Încercăm mai multe ferestre/padding-uri pentru a reduce rateul pe celulele înguste.
    attempts = [
        (0, 0, 2, 7, False),
        (-1, 1, 1, 7, False),
        (-2, 2, 1, 6, False),
        (-2, 2, 0, 7, False),
        (-3, 3, 0, 6, False),
        (-2, 2, 1, 7, True),
    ]

    for dy0, dy1, pad, psm, use_gray in attempts:
        yy0 = max(0, y0 + dy0)
        yy1 = min(h, y1 + dy1)
        if yy1 <= yy0:
            continue

        if use_gray:
            cell = table_bgr[yy0:yy1, max(0, x0 + 1):min(table_bgr.shape[1], x1 - 1)]
            if cell.size == 0:
                continue
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            gray = cv2.bilateralFilter(gray, 5, 35, 35)
            txt = ocr_cell(gray, lang=OCR_NUM_LANG, psm=psm, digits_only=True)
        else:
            txt = ocr_cell(
                crop_cell(table_bgr, x0, yy0, x1, yy1, pad=pad),
                lang=OCR_NUM_LANG,
                psm=psm,
                digits_only=True,
            )

        val = num_or_none(txt)
        if val is not None:
            return val

    return None


def download_bytes(url: str, session: Optional[requests.Session] = None, timeout: int = 30) -> bytes:
    sess = session or requests.Session()
    r = sess.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.content


def parse_date_from_bulletin_url(url: str) -> Optional[str]:
    # ...intervalul-16-04-2026-ora...
    m = re.search(r"intervalul-(\d{2})-(\d{2})-(\d{4})-ora", url)
    if not m:
        return None
    dd, mm, yyyy = m.groups()
    return f"{yyyy}-{mm}-{dd}"


def has_resized_suffix(url: str) -> bool:
    """Detect WordPress-like resized image suffix, e.g. foo-640x890.jpg."""
    path = requests.compat.urlparse(url).path.lower()
    return re.search(r"-\d{2,5}x\d{2,5}\.[a-z0-9]+$", path) is not None


def original_variant_url(url: str) -> str:
    """Map resized variant URL to original URL by removing -<w>x<h> suffix."""
    parsed = requests.compat.urlparse(url)
    new_path = re.sub(r"-\d{2,5}x\d{2,5}(\.[a-z0-9]+)$", r"\1", parsed.path, flags=re.IGNORECASE)
    if new_path == parsed.path:
        return url
    return requests.compat.urlunparse((
        parsed.scheme,
        parsed.netloc,
        new_path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))


def safe_stem_from_source(image_source: str) -> str:
    candidate = image_source.split("?")[0].split("#")[0]
    name = Path(candidate).name or "image"
    stem = Path(name).stem or "image"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    if not stem:
        stem = "image"
    return stem


def ensure_local_image(image_source: str, download_dir: Optional[Path], session: Optional[requests.Session] = None) -> Path:
    if not re.match(r"^https?://", image_source):
        p = Path(image_source).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Imaginea locală nu există: {p}")
        return p

    if download_dir is None:
        raise ValueError("download_dir este necesar pentru surse imagine URL în backend deepseek")

    download_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem_from_source(image_source)
    h = hashlib.md5(image_source.encode("utf-8")).hexdigest()[:8]
    ext = Path(image_source.split("?")[0]).suffix.lower() or ".jpg"
    local_path = download_dir / f"{stem}-{h}{ext}"
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    data = download_bytes(image_source, session=session)
    local_path.write_bytes(data)
    return local_path


def deepseek_output_dir_for_image(local_image_path: Path, deepseek_outputs_dir: Path) -> Path:
    return deepseek_outputs_dir / f"{local_image_path.stem}_explained.md"


def extract_first_html_table(markdown_text: str) -> str:
    m = re.search(r"(<table\b[\s\S]*?</table>)", markdown_text, flags=re.IGNORECASE)
    if not m:
        raise RuntimeError("Nu am găsit niciun <table>...</table> în fișierul DeepSeek result.mmd")
    return m.group(1)


def parse_deepseek_table_to_dataframe(
    markdown_text: str,
    bulletin_date: str,
    bulletin_url: str,
    image_url: str,
    target_rivers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    table_html = extract_first_html_table(markdown_text)
    soup = BeautifulSoup(table_html, "html.parser")

    targets = {canonical_river_name(x) for x in (target_rivers or [])}
    rows = []

    for tr in soup.find_all("tr"):
        cells = [normalize_spaces(td.get_text(" ", strip=True)) for td in tr.find_all("td")]
        if len(cells) < 8:
            continue
        if not re.fullmatch(r"\d+", cells[0]):
            continue

        river = canonical_river_name(cells[1]) if len(cells) > 1 else ""
        if targets and river not in targets:
            continue

        rec: dict[str, object] = {
            "data": bulletin_date,
            "bulletin_url": bulletin_url,
            "image_url": image_url,
            "raul": river,
            "statia": clean_station_text(cells[2] if len(cells) > 2 else ""),
            "jud": clean_county_text(cells[3] if len(cells) > 3 else ""),
            "CA_cm": num_or_none(cells[4] if len(cells) > 4 else ""),
            "CI_cm": num_or_none(cells[5] if len(cells) > 5 else ""),
            "CP_cm": num_or_none(cells[6] if len(cells) > 6 else ""),
            "Qmed_apr_m3_s": num_or_none(cells[7] if len(cells) > 7 else ""),
            "diag_data": bulletin_date,
            "diag_H_cm": num_or_none(cells[8] if len(cells) > 8 else ""),
            "diag_Q_m3_s": num_or_none(cells[9] if len(cells) > 9 else ""),
            "diag_dH_cm": num_or_none(cells[10] if len(cells) > 10 else ""),
            "diag_G_P": cells[11] if len(cells) > 11 else "",
            "prog_data": "",
            "prog_H_cm": num_or_none(cells[12] if len(cells) > 12 else ""),
            "prog_Q_m3_s": num_or_none(cells[13] if len(cells) > 13 else ""),
            # În unele ieșiri DeepSeek coloanele +dH și G apar unite; păstrăm ce există.
            "prog_dH_cm": num_or_none(cells[14] if len(cells) > 14 else ""),
            "prog_G_P": cells[15] if len(cells) > 15 else "",
        }

        if not rec["raul"] or not rec["statia"]:
            continue
        rows.append(rec)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col.endswith("_G_P") or col in {"data", "bulletin_url", "image_url", "raul", "statia", "jud", "diag_data", "prog_data"} else np.nan
    return df[OUTPUT_COLUMNS]


def maybe_run_deepseek_ocr(command_template: Optional[str], image_path: Path, out_dir: Path) -> None:
    if not command_template:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = command_template.format(image=str(image_path), outdir=str(out_dir))
    log(f"[DEEPSEEK] Rulez: {cmd}")
    subprocess.run(shlex.split(cmd), check=True)


def parse_single_image_deepseek(
    image_source: str,
    bulletin_date: str,
    bulletin_url: str,
    image_url: str,
    rivers: Optional[Sequence[str]],
    deepseek_outputs_dir: Path,
    deepseek_command_template: Optional[str],
    download_dir: Optional[Path],
    session: Optional[requests.Session],
) -> pd.DataFrame:
    local_image_path = ensure_local_image(image_source, download_dir=download_dir, session=session)
    out_dir = deepseek_output_dir_for_image(local_image_path, deepseek_outputs_dir)
    result_mmd = out_dir / "result.mmd"

    if not result_mmd.exists():
        maybe_run_deepseek_ocr(deepseek_command_template, local_image_path, out_dir)

    if not result_mmd.exists():
        raise RuntimeError(
            "Lipsește result.mmd pentru DeepSeek. "
            f"Așteptat la: {result_mmd}. "
            "Folosește --deepseek-command-template sau generează output-ul în prealabil."
        )

    markdown_text = result_mmd.read_text(encoding="utf-8", errors="replace")
    return parse_deepseek_table_to_dataframe(
        markdown_text=markdown_text,
        bulletin_date=bulletin_date,
        bulletin_url=bulletin_url,
        image_url=image_url or image_source,
        target_rivers=rivers,
    )


def get_soup(url: str, session: Optional[requests.Session] = None) -> BeautifulSoup:
    html = download_bytes(url, session=session)
    return BeautifulSoup(html, "html.parser")


def absolute_url(base: str, maybe_relative: str) -> str:
    return requests.compat.urljoin(base, maybe_relative)


def find_pagination_links(soup: BeautifulSoup, page_url: str) -> List[str]:
    urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "bulletin_type/prognoza-hidrologica-pentru-rauri" in href:
            urls.add(absolute_url(page_url, href))
    return sorted(urls)


def list_bulletin_urls(root_url: str, max_pages: int = 1, session: Optional[requests.Session] = None) -> List[str]:
    sess = session or requests.Session()
    to_visit = [root_url]
    visited_pages = set()
    bulletin_urls = set()

    while to_visit and len(visited_pages) < max_pages:
        page = to_visit.pop(0)
        if page in visited_pages:
            continue
        visited_pages.add(page)
        log(f"[LIST] {page}")
        soup = get_soup(page, session=sess)

        for a in soup.find_all("a", href=True):
            href = absolute_url(page, a["href"])
            if "/bulletin/prognoza-hidrologica-pentru-rauri" in href:
                bulletin_urls.add(href)

        for p in find_pagination_links(soup, page):
            if p not in visited_pages and p not in to_visit:
                to_visit.append(p)

        time.sleep(0.4)

    return sorted(bulletin_urls)


def find_main_table_image(bulletin_url: str, session: Optional[requests.Session] = None) -> Optional[BulletinImage]:
    sess = session or requests.Session()
    soup = get_soup(bulletin_url, session=sess)

    image_candidates = []

    def add_candidate(src_url: str, base_score: float) -> None:
        s = base_score
        # Prefer original images; resized variants are often blurrier for OCR.
        if has_resized_suffix(src_url):
            s -= 8
            original = original_variant_url(src_url)
            if original != src_url:
                image_candidates.append((s + 10, original))
        else:
            s += 4
        image_candidates.append((s, src_url))

    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        src = absolute_url(bulletin_url, src)
        alt = (img.get("alt") or "").lower()
        width = int(img.get("width") or 0)
        height = int(img.get("height") or 0)

        # Preferăm imaginea mare din tabel, nu harta mică.
        score = 0
        if "wp-content/uploads" in src:
            score += 10
        if width >= 600 or height >= 800:
            score += 10
        if any(token in src.lower() for token in ["pg2", "640x890", "640x904", "3-10", "4-"]):
            score += 5
        # imaginea mare e de obicei portret
        if height > width:
            score += 5

        add_candidate(src, score)

        srcset = img.get("srcset")
        if srcset:
            for part in srcset.split(","):
                p = part.strip().split(" ")[0]
                if not p:
                    continue
                p = absolute_url(bulletin_url, p)
                score2 = score
                add_candidate(p, score2)

    if not image_candidates:
        return None

    # Eliminăm duplicatele păstrând scorul maxim.
    best = {}
    for score, src in image_candidates:
        best[src] = max(score, best.get(src, -10))

    # Prefer shorter URL when scores tie; originals are usually shorter than resized variants.
    ranked = sorted(best.items(), key=lambda x: (-x[1], len(x[0])))
    image_url = ranked[0][0]
    bulletin_date = parse_date_from_bulletin_url(bulletin_url) or ""
    return BulletinImage(bulletin_url=bulletin_url, bulletin_date=bulletin_date, image_url=image_url)


def read_image_any(source: str) -> np.ndarray:
    ensure_tesseract_dependencies()
    if re.match(r"^https?://", source):
        data = download_bytes(source)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Nu am putut citi imaginea: {source}")
    return img


def preprocess_for_lines(img_bgr: np.ndarray, scale: float = 2.0) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # prag adaptiv, text/table dark on white
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 15
    )
    return bw


def find_table_bbox(img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """Detectează conturul principal al tabelului."""
    bw = preprocess_for_lines(img_bgr, scale=2.0)
    h, w = bw.shape

    # Căutăm zone mari, dreptunghiulare, cu multe linii.
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1

    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < 0.15 * w * h:
            continue
        aspect = hh / max(ww, 1)
        score = area
        # tabelul este portret și mare
        if 1.0 < aspect < 2.5:
            score *= 1.2
        if y > 0.2 * h:
            score *= 0.9
        if score > best_score:
            best_score = score
            best = (x, y, ww, hh)

    if best is None:
        raise RuntimeError("Nu am putut detecta bounding box-ul tabelului.")
    x, y, ww, hh = best
    # Mărim puțin marginile
    pad = 6
    x = max(0, x - pad)
    y = max(0, y - pad)
    ww = min(img_bgr.shape[1] - x, ww + 2 * pad)
    hh = min(img_bgr.shape[0] - y, hh + 2 * pad)
    return x, y, ww, hh


def detect_vertical_boundaries(table_bgr: np.ndarray) -> List[int]:
    gray = cv2.cvtColor(table_bgr, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    bw = cv2.adaptiveThreshold(
        up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 15
    )

    kernel_h = max(20, up.shape[0] // 12)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical_kernel)

    proj = vertical.sum(axis=0)
    threshold = 0.25 * proj.max()
    xs = np.where(proj > threshold)[0]

    if len(xs) == 0:
        raise RuntimeError("Nu am detectat liniile verticale ale tabelului.")

    groups = []
    start = xs[0]
    prev = xs[0]
    for x in xs[1:]:
        if x - prev > 6:
            groups.append((start, prev))
            start = x
        prev = x
    groups.append((start, prev))

    centers = [int((a + b) / 2) for a, b in groups]
    centers = [int(c / 2) for c in centers]  # scale back

    # Uneori apar multe linii subțiri duble; le comprimăm.
    merged = []
    for c in centers:
        if not merged or abs(c - merged[-1]) > 8:
            merged.append(c)
        else:
            merged[-1] = int((merged[-1] + c) / 2)

    return merged


def detect_header_bottom(table_bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(table_bgr, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    bw = cv2.adaptiveThreshold(
        up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 15
    )

    kernel_w = max(50, up.shape[1] // 8)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel)

    proj = horizontal.sum(axis=1)
    thresh = 0.35 * proj.max()
    ys = np.where(proj > thresh)[0]
    if len(ys) == 0:
        # fallback aproximativ: ~15% din înălțime e header
        return int(table_bgr.shape[0] * 0.15)

    groups = []
    start = ys[0]
    prev = ys[0]
    for y in ys[1:]:
        if y - prev > 6:
            groups.append((start, prev))
            start = y
        prev = y
    groups.append((start, prev))

    centers = [int((a + b) / 2) for a, b in groups]
    centers = [int(c / 2) for c in centers]

    # Liniile de header apar sus; ne interesează ultima linie groasă înainte de date.
    top_lines = [c for c in centers if c < table_bgr.shape[0] * 0.35]
    if not top_lines:
        return int(table_bgr.shape[0] * 0.15)
    return max(top_lines) + 2


def crop_cell(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, pad: int = 2) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, x0 + pad)
    y0 = max(0, y0 + pad)
    x1 = min(w, x1 - pad)
    y1 = min(h, y1 - pad)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((10, 10), dtype=np.uint8)
    cell = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def crop_text_cell(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, pad: int = 2) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, x0 + pad)
    y0 = max(0, y0 + pad)
    x1 = min(w, x1 - pad)
    y1 = min(h, y1 - pad)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((10, 10), dtype=np.uint8)
    cell = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    # Pentru text, păstrăm grayscale; pragul binar agresiv pierde litere subțiri.
    gray = cv2.bilateralFilter(gray, 5, 35, 35)
    return gray


def extract_record_from_window(
    table_bgr: np.ndarray,
    cmap: dict[str, tuple[int, int]],
    bulletin_date: str,
    bulletin_url: str,
    image_url: str,
    y0: int,
    y1: int,
    forced_river: Optional[str] = None,
) -> dict:
    h, w = table_bgr.shape[:2]

    def ocr_text_col(col: str, psm: int = 7, whitelist: Optional[str] = None) -> str:
        x0, x1 = cmap[col]
        xx0 = max(0, x0 + 3)
        xx1 = min(w, x1 - 3)
        yy0 = max(0, y0 + 1)
        yy1 = min(h, y1 - 1)
        if xx1 <= xx0 or yy1 <= yy0:
            return ""

        cell = table_bgr[yy0:yy1, xx0:xx1]
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 5, 35, 35)
        return ocr_cell(gray, lang=OCR_TEXT_LANG, psm=psm, whitelist=whitelist)

    rec: dict[str, object] = {
        "data": bulletin_date,
        "bulletin_url": bulletin_url,
        "image_url": image_url,
        "diag_data": bulletin_date,
        "prog_data": "",
    }

    rec["raul"] = forced_river or canonical_river_name(
        ocr_text_col("raul", psm=7)
    )
    rec["statia"] = clean_station_text(
        ocr_text_col("statia", psm=7)
    )
    rec["jud"] = clean_county_text(
        ocr_text_col("jud", psm=7, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )

    for col in [
        "CA_cm", "CI_cm", "CP_cm", "Qmed_apr_m3_s",
        "diag_H_cm", "diag_Q_m3_s", "diag_dH_cm",
        "prog_H_cm", "prog_Q_m3_s", "prog_dH_cm",
    ]:
        rec[col] = ocr_num_from_region(table_bgr, cmap[col][0], y0, cmap[col][1], y1)

    rec["diag_G_P"] = ocr_text_col("diag_G_P", psm=7)
    rec["prog_G_P"] = ocr_text_col("prog_G_P", psm=7)
    return rec


def extract_targeted_rows(
    table_bgr: np.ndarray,
    bulletin_date: str,
    bulletin_url: str,
    image_url: str,
    cmap: dict[str, tuple[int, int]],
    targets: Sequence[str],
) -> pd.DataFrame:
    table_h = table_bgr.shape[0]
    y_start = int(table_h * 0.05)
    y_end = int(table_h * 0.92)
    candidate_heights = sorted(set([13, 14, 15, 16]))

    best: dict[str, tuple[float, int, int, str]] = {
        t: (0.0, -1, -1, "") for t in targets
    }

    river_x0, river_x1 = cmap["raul"]

    def river_scan_text(y0: int, y1: int) -> str:
        x0 = max(0, river_x0 + 3)
        x1 = min(table_bgr.shape[1], river_x1 - 3)
        yy0 = max(0, y0 + 2)
        yy1 = min(table_bgr.shape[0], y1 - 2)
        if x1 <= x0 or yy1 <= yy0:
            return ""

        cell = table_bgr[yy0:yy1, x0:x1]
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 5, 35, 35)
        return ocr_cell(gray, lang=OCR_TEXT_LANG, psm=7)

    def update_best(step: int, around: Optional[dict[str, int]] = None) -> None:
        for row_h in candidate_heights:
            if around:
                ranges = [
                    (max(y_start, y_mid - 24), min(y_end - row_h, y_mid + 24))
                    for y_mid in around.values()
                    if y_mid >= 0
                ]
                if not ranges:
                    ranges = [(y_start, y_end - row_h)]
            else:
                ranges = [(y_start, y_end - row_h)]

            for lo, hi in ranges:
                if hi <= lo:
                    continue
                for y0 in range(lo, hi, step):
                    y1 = y0 + row_h
                    river_txt = river_scan_text(y0, y1)
                    if not river_txt:
                        continue
                    for target in targets:
                        score = river_match_score(river_txt, target)
                        if score > best[target][0]:
                            best[target] = (score, y0, y1, river_txt)

    # Scanare coarse, apoi rafinare locală în jurul candidaților buni.
    update_best(step=8, around=None)
    around = {k: v[1] for k, v in best.items()}
    update_best(step=1, around=around)

    rows = []
    for target in targets:
        score, y0, y1, raw_txt = best[target]
        if score < 0.58:
            log(f"[WARN] Nu am identificat sigur raul '{target}' (best={score:.2f}, raw='{raw_txt}')")
            continue

        # Ajustăm local poziția pe Y pentru a prinde mai bine textul stației.
        best_local = (-1.0, y0, y1)
        for dy in range(-5, 6):
            yy0 = max(0, y0 + dy)
            yy1 = min(table_h, y1 + dy)
            if yy1 - yy0 < 8:
                continue

            river_local = river_scan_text(yy0, yy1)
            river_local_score = river_match_score(river_local, target)
            station_local = ocr_cell(crop_text_cell(table_bgr, *cmap["statia"], yy0, yy1), lang=OCR_TEXT_LANG, psm=7)
            station_letters = len(re.sub(r"[^A-Za-z]", "", station_local))
            local_score = 2.0 * river_local_score + min(station_letters, 14) / 14.0
            if local_score > best_local[0]:
                best_local = (local_score, yy0, yy1)

        y0, y1 = best_local[1], best_local[2]

        rec = extract_record_from_window(
            table_bgr=table_bgr,
            cmap=cmap,
            bulletin_date=bulletin_date,
            bulletin_url=bulletin_url,
            image_url=image_url,
            y0=max(0, y0 - 2),
            y1=min(table_h, y1 + 2),
            forced_river=target,
        )
        rows.append(rec)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[OUTPUT_COLUMNS]
    return df


def column_index_map(verticals: List[int]) -> dict[str, tuple[int, int]]:
    """
    Așteptăm aproximativ aceste delimitări:
    [left, nr, rau, statie, jud, ca, ci, cp, qmed, diag_h, diag_q, diag_dh, diag_gp,
     prog_h, prog_q, prog_dh, prog_gp, right]

    Dacă detectăm mai puține/multe, folosim relații aproximative.
    """
    if len(verticals) < 16:
        raise RuntimeError(f"Prea puține linii verticale detectate: {len(verticals)}")

    # Folosim primele 18 semnificative.
    v = verticals[:18] if len(verticals) >= 18 else verticals

    def seg(i: int) -> tuple[int, int]:
        return (v[i], v[i + 1])

    # Mapează segmentele standard.
    # Funcționează bine pe layout-ul observat.
    out = {
        "nr": seg(0),
        "raul": seg(1),
        "statia": seg(2),
        "jud": seg(3),
        "CA_cm": seg(4),
        "CI_cm": seg(5),
        "CP_cm": seg(6),
        "Qmed_apr_m3_s": seg(7),
        "diag_H_cm": seg(8),
        "diag_Q_m3_s": seg(9),
        "diag_dH_cm": seg(10),
        "diag_G_P": seg(11),
        "prog_H_cm": seg(12),
        "prog_Q_m3_s": seg(13),
        "prog_dH_cm": seg(14),
        "prog_G_P": seg(15),
    }
    return out


def extract_rows_from_table(
    table_bgr: np.ndarray,
    bulletin_date: str,
    bulletin_url: str,
    image_url: str,
    target_rivers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    verticals = detect_vertical_boundaries(table_bgr)
    header_bottom = detect_header_bottom(table_bgr)
    cmap = column_index_map(verticals)

    table_h, table_w = table_bgr.shape[:2]

    # Zona de date: de sub header până înainte de legendă.
    # Ultima linie a tabelului e marginea de jos.
    data_y0 = header_bottom
    data_y1 = table_h - max(18, table_h // 30)  # elimină legenda + margine
    n_rows = 50
    row_h = (data_y1 - data_y0) / n_rows

    targets_list = [canonical_river_name(x) for x in (target_rivers or [])]
    targets = set(targets_list)

    # Fallback robust pentru cazurile când segmentarea pe 50 de rânduri nu se aliniază corect.
    if targets_list:
        df_targeted = extract_targeted_rows(
            table_bgr=table_bgr,
            bulletin_date=bulletin_date,
            bulletin_url=bulletin_url,
            image_url=image_url,
            cmap=cmap,
            targets=targets_list,
        )
        if not df_targeted.empty:
            found = set(df_targeted["raul"].tolist())
            missing = sorted(set(targets_list) - found)
            if not missing:
                return df_targeted
            log(f"[WARN] Lipsesc in fallback: {', '.join(missing)}. Incerc segmentarea clasica.")

    rows = []

    for i in range(n_rows):
        y0 = int(data_y0 + i * row_h)
        y1 = int(data_y0 + (i + 1) * row_h)

        rec = extract_record_from_window(
            table_bgr=table_bgr,
            cmap=cmap,
            bulletin_date=bulletin_date,
            bulletin_url=bulletin_url,
            image_url=image_url,
            y0=y0,
            y1=y1,
        )

        if targets:
            if rec["raul"] not in targets:
                continue

        # Ignoră rândurile goale / eronate.
        if not rec["raul"] or not rec["statia"]:
            continue

        rows.append(rec)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[OUTPUT_COLUMNS]
    return df


def parse_single_image(
    image_source: str,
    bulletin_date: str = "",
    bulletin_url: str = "",
    image_url: str = "",
    rivers: Optional[Sequence[str]] = None,
    debug_dir: Optional[Path] = None,
    backend: str = DEFAULT_BACKEND,
    deepseek_outputs_dir: Optional[Path] = None,
    deepseek_command_template: Optional[str] = None,
    download_dir: Optional[Path] = None,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    if backend == "deepseek":
        if deepseek_outputs_dir is None:
            raise ValueError("deepseek_outputs_dir este necesar pentru backend=deepseek")
        return parse_single_image_deepseek(
            image_source=image_source,
            bulletin_date=bulletin_date,
            bulletin_url=bulletin_url,
            image_url=image_url,
            rivers=rivers,
            deepseek_outputs_dir=deepseek_outputs_dir,
            deepseek_command_template=deepseek_command_template,
            download_dir=download_dir,
            session=session,
        )

    ensure_tesseract_dependencies()
    img = read_image_any(image_source)
    x, y, w, h = find_table_bbox(img)
    table = img[y:y+h, x:x+w].copy()

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "01_table_bbox.png"), table)

    df = extract_rows_from_table(
        table_bgr=table,
        bulletin_date=bulletin_date,
        bulletin_url=bulletin_url,
        image_url=image_url or image_source,
        target_rivers=rivers,
    )
    return df


def parse_bulletin(
    bulletin_url: str,
    rivers: Optional[Sequence[str]] = None,
    session: Optional[requests.Session] = None,
    debug_dir: Optional[Path] = None,
    backend: str = DEFAULT_BACKEND,
    deepseek_outputs_dir: Optional[Path] = None,
    deepseek_command_template: Optional[str] = None,
    download_dir: Optional[Path] = None,
) -> pd.DataFrame:
    meta = find_main_table_image(bulletin_url, session=session)
    if meta is None:
        raise RuntimeError(f"Nu am găsit imaginea principală pentru buletin: {bulletin_url}")

    log(f"[BULLETIN] {bulletin_url}")
    log(f"[IMAGE] {meta.image_url}")

    df = parse_single_image(
        image_source=meta.image_url,
        bulletin_date=meta.bulletin_date,
        bulletin_url=meta.bulletin_url,
        image_url=meta.image_url,
        rivers=rivers,
        debug_dir=debug_dir,
        backend=backend,
        deepseek_outputs_dir=deepseek_outputs_dir,
        deepseek_command_template=deepseek_command_template,
        download_dir=download_dir,
        session=session,
    )
    return df


def export_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Format neacceptat pentru output: {suffix}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parser OCR pentru buletinele hidrologice hidro.ro")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Imagine locală sau URL direct către imaginea tabelului")
    src.add_argument("--bulletin-url", help="URL către un buletin individual")
    src.add_argument("--root-url", help="URL root cu lista de buletine")
    p.add_argument("-o", "--output", required=True, help="Fișier output: .csv / .xlsx / .parquet")
    p.add_argument("--rivers", nargs="*", default=DEFAULT_RIVERS, help="Râurile de filtrat")
    p.add_argument("--max-pages", type=int, default=1, help="Numărul de pagini listă de parcurs de la root")
    p.add_argument("--sleep", type=float, default=0.4, help="Pauză între requesturi")
    p.add_argument("--tesseract-cmd", default=None, help="Path explicit către binarul tesseract")
    p.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=["tesseract", "deepseek"],
        help="Backend OCR: tesseract (implicit) sau deepseek (parsează result.mmd)",
    )
    p.add_argument(
        "--deepseek-outputs-dir",
        default=".",
        help="Director bază cu output-uri DeepSeek; pentru fiecare imagine se caută <stem>_explained.md/result.mmd",
    )
    p.add_argument(
        "--deepseek-command-template",
        default=None,
        help=(
            "Comandă opțională pentru generarea output-ului DeepSeek când lipsește result.mmd. "
            "Folosește placeholder-ele {image} și {outdir}."
        ),
    )
    p.add_argument(
        "--download-dir",
        default=".hydro_downloads",
        help="Director local pentru imagini descărcate (folosit în backend deepseek pentru URL-uri)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Număr de lucrători pentru procesarea mai multor buletine (util în modul --root-url)",
    )
    p.add_argument(
        "--ocr-text-lang",
        default=DEFAULT_OCR_TEXT_LANG,
        help="Limbă OCR pentru text (ex: ron sau ron+eng)",
    )
    p.add_argument(
        "--ocr-num-lang",
        default=DEFAULT_OCR_NUM_LANG,
        help="Limbă OCR pentru coloane numerice (de regulă eng)",
    )
    p.add_argument("--date", default="", help="Data buletinului pentru modul --image")
    p.add_argument("--debug-dir", default=None, help="Director pentru imagini intermediare")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    global OCR_TEXT_LANG, OCR_NUM_LANG
    OCR_TEXT_LANG = args.ocr_text_lang
    OCR_NUM_LANG = args.ocr_num_lang

    if args.backend == "tesseract":
        ensure_tesseract_dependencies()

    if args.tesseract_cmd:
        if pytesseract is None:
            raise RuntimeError("--tesseract-cmd a fost setat, dar pytesseract nu este instalat.")
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    log(f"[OCR] text={OCR_TEXT_LANG} num={OCR_NUM_LANG}")

    session = requests.Session()
    session.headers.update(HEADERS)

    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    deepseek_outputs_dir = Path(args.deepseek_outputs_dir).expanduser().resolve()
    download_dir = Path(args.download_dir).expanduser().resolve()

    if args.image:
        df = parse_single_image(
            image_source=args.image,
            bulletin_date=args.date,
            bulletin_url="",
            image_url=args.image if re.match(r"^https?://", args.image) else "",
            rivers=args.rivers,
            debug_dir=debug_dir,
            backend=args.backend,
            deepseek_outputs_dir=deepseek_outputs_dir,
            deepseek_command_template=args.deepseek_command_template,
            download_dir=download_dir,
            session=session,
        )
    elif args.bulletin_url:
        df = parse_bulletin(
            bulletin_url=args.bulletin_url,
            rivers=args.rivers,
            session=session,
            debug_dir=debug_dir,
            backend=args.backend,
            deepseek_outputs_dir=deepseek_outputs_dir,
            deepseek_command_template=args.deepseek_command_template,
            download_dir=download_dir,
        )
    else:
        bulletins = list_bulletin_urls(
            root_url=args.root_url or DEFAULT_ROOT_URL,
            max_pages=args.max_pages,
            session=session,
        )
        log(f"[FOUND] {len(bulletins)} buletine")
        parts = []

        def _run_one(idx_url: tuple[int, str]) -> Optional[pd.DataFrame]:
            idx, url = idx_url
            try:
                log(f"[{idx}/{len(bulletins)}] {url}")
                return parse_bulletin(
                    url,
                    rivers=args.rivers,
                    session=session,
                    debug_dir=debug_dir,
                    backend=args.backend,
                    deepseek_outputs_dir=deepseek_outputs_dir,
                    deepseek_command_template=args.deepseek_command_template,
                    download_dir=download_dir,
                )
            except Exception as e:
                log(f"[WARN] Eroare la {url}: {e}")
                return None

        if args.workers <= 1:
            for idx, url in enumerate(bulletins, start=1):
                part = _run_one((idx, url))
                if part is not None:
                    parts.append(part)
                time.sleep(args.sleep)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = [
                    ex.submit(_run_one, (idx, url))
                    for idx, url in enumerate(bulletins, start=1)
                ]
                for fut in concurrent.futures.as_completed(futures):
                    part = fut.result()
                    if part is not None:
                        parts.append(part)

        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=OUTPUT_COLUMNS)

    if df.empty:
        log("Nu s-au extras rânduri.")
    else:
        # ordonare utilă
        sort_cols = [c for c in ["data", "raul", "statia"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

    export_dataframe(df, Path(args.output))
    log(f"[DONE] {args.output} ({len(df)} rânduri)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
