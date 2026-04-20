#!/usr/bin/env python3
"""Client for DeepSeek OCR2 vLLM service with local generation cache."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional, Sequence, TextIO
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from deepseek_ocr2_service_config import (
    DEFAULT_MODE,
    DEFAULT_ROOT_URL,
    FREE_OCR_PROMPT,
    GROUNDING_PROMPT,
    GENERATIONS_SUBDIR,
    HTTP_TIMEOUT_SECONDS,
    IMAGES_SUBDIR,
    SERVER_URL,
    STORE_DIR,
)
from hydro_parser_tesseract import HEADERS, find_main_table_image, list_bulletin_urls


@dataclasses.dataclass
class SourceItem:
    image_source: str
    bulletin_url: str = ""
    bulletin_date: str = ""


FIELDS_BEFORE_FIRST_GP = [
    "nr_crt",
    "raul",
    "statia_hidrometrica",
    "jud",
    "CA_cm",
    "CI_cm",
    "CP_cm",
    "Qmed_apr_m3_s",
    "diag_H_cm",
    "diag_Q_m3_s",
    "diag_dH_cm",
]

FIELDS_FROM_FIRST_GP = [
    "diag_GP",
    "prog_H_cm",
    "prog_Q_m3_s",
    "prog_dH_cm",
    "prog_GP",
]

ALL_TABLE_FIELDS = FIELDS_BEFORE_FIRST_GP + FIELDS_FROM_FIRST_GP
EXPORT_META_FIELDS = ["date", "bulletin_url", "image_source", "generation_key", "image_key"]


def log(msg: str) -> None:
    print(msg)


def is_url(value: str) -> bool:
    return bool(re.match(r"^https?://", value))


def safe_stem(value: str) -> str:
    base = Path(urlparse(value).path).name if is_url(value) else Path(value).name
    stem = Path(base).stem or "image"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "image"


def guess_ext(value: str) -> str:
    ext = Path(urlparse(value).path if is_url(value) else value).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        return ext
    return ".jpg"


def load_bytes(source: str, session: requests.Session, timeout_s: int) -> bytes:
    if is_url(source):
        r = session.get(source, timeout=timeout_s)
        r.raise_for_status()
        return r.content
    return Path(source).expanduser().resolve().read_bytes()


def image_key(source: str, image_bytes: bytes) -> str:
    digest = hashlib.sha256(image_bytes).hexdigest()[:16]
    return f"{safe_stem(source)}__{digest}"


def resolve_prompt(mode: str, prompt_override: Optional[str]) -> str:
    if prompt_override:
        return prompt_override
    if mode == "grounding":
        return GROUNDING_PROMPT
    return FREE_OCR_PROMPT


def generation_key(image_key_value: str, mode: str, prompt: str, ignore_eos: bool) -> str:
    mode_tag = "grounding" if mode == "grounding" else "free"
    eos_tag = "noeos" if ignore_eos else "eos"
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:10]
    return f"{image_key_value}__{mode_tag}__{eos_tag}__{prompt_digest}"


def add_warning(
    warnings: Optional[list[dict[str, Any]]],
    code: str,
    message: str,
    **extra: Any,
) -> None:
    if warnings is None:
        return
    item: dict[str, Any] = {"code": code, "message": message}
    item.update(extra)
    warnings.append(item)


def empty_table_row() -> dict[str, str]:
    return {k: "" for k in ALL_TABLE_FIELDS}


def map_consecutive_best_effort(cells: list[str], fields: list[str]) -> dict[str, str]:
    row = {k: "" for k in fields}
    for i, value in enumerate(cells[: len(fields)]):
        row[fields[i]] = value
    return row


def normalize_river_for_filter(value: str) -> str:
    base = unicodedata.normalize("NFKD", value or "")
    base = "".join(ch for ch in base if not unicodedata.combining(ch))
    base = base.lower()
    base = re.sub(r"[^a-z]", "", base)
    return base


def should_keep_row_for_rivers(row: dict[str, str], river_filters: set[str]) -> bool:
    if not river_filters:
        return True
    river = normalize_river_for_filter(row.get("raul", ""))
    return river in river_filters


def export_row_from_table_row(source: SourceItem, meta: dict[str, Any], table_row: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {
        "date": source.bulletin_date or "",
        "bulletin_url": source.bulletin_url or "",
        "image_source": source.image_source,
        "generation_key": str(meta.get("key", "")),
        "image_key": str(meta.get("image_key", "")),
    }
    for k, v in table_row.items():
        out[k] = "" if v is None else str(v)
    return out


def load_generation_rows(meta: dict[str, Any]) -> list[dict[str, str]]:
    result_path = Path(str(meta.get("result_path", "")))
    generation_dir = result_path.parent if result_path.name else Path()
    rows_path = generation_dir / "table_rows.json"
    if not rows_path.exists():
        return []
    try:
        loaded = json.loads(rows_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(loaded, list):
        return []
    out: list[dict[str, str]] = []
    for row in loaded:
        if isinstance(row, dict):
            out.append({str(k): "" if v is None else str(v) for k, v in row.items()})
    return out


class CsvShardWriter:
    def __init__(self, output_dir: Path, prefix: str, max_rows: int) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_rows = max_rows if max_rows > 0 else 1000000000
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.part_idx = 0
        self.rows_in_part = 0
        self.total_rows = 0
        self.current_fields: list[str] = []
        self.current_file: Optional[TextIO] = None
        self.current_writer: Optional[csv.DictWriter] = None
        self.files_written: list[str] = []

    def _next_path(self) -> Path:
        self.part_idx += 1
        return self.output_dir / f"{self.prefix}_part{self.part_idx:04d}.csv"

    def _close_current(self) -> None:
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None
            self.current_writer = None
            self.rows_in_part = 0

    def _open_new(self, fields: list[str]) -> None:
        self._close_current()
        self.current_fields = fields
        path = self._next_path()
        self.current_file = path.open("w", encoding="utf-8", newline="")
        self.current_writer = csv.DictWriter(self.current_file, fieldnames=self.current_fields, extrasaction="ignore")
        self.current_writer.writeheader()
        self.files_written.append(str(path))

    def write_rows(self, rows: list[dict[str, str]]) -> None:
        for row in rows:
            incoming_fields = [k for k in row.keys()]
            if not self.current_fields:
                self._open_new(incoming_fields)
            elif any(k not in self.current_fields for k in incoming_fields):
                # Keep headers correct when a later row exposes additional columns.
                expanded = self.current_fields + [k for k in incoming_fields if k not in self.current_fields]
                self._open_new(expanded)
            elif self.rows_in_part >= self.max_rows:
                self._open_new(self.current_fields)

            assert self.current_writer is not None
            self.current_writer.writerow(row)
            self.rows_in_part += 1
            self.total_rows += 1

            if self.current_file is not None:
                self.current_file.flush()

    def close(self) -> None:
        self._close_current()


def parse_markdown_table(text: str, warnings: Optional[list[dict[str, Any]]] = None) -> list[dict[str, str]]:
    lines = [line.rstrip() for line in text.splitlines()]

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.count("|") >= 2:
            current.append(stripped)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    if not blocks:
        add_warning(warnings, "markdown_table_missing", "No markdown table block found.")
        return []

    block = max(blocks, key=len)
    if len(block) < 2:
        add_warning(warnings, "markdown_table_too_short", "Markdown table block has fewer than 2 rows.")
        return []

    def row_cells(raw: str) -> list[str]:
        s = raw.strip().strip("|")
        return [c.strip() for c in s.split("|")]

    def is_separator(raw: str) -> bool:
        cells = row_cells(raw)
        if not cells:
            return False
        return all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) is not None for c in cells)

    raw_header = row_cells(block[0])

    def normalize_hydro_header(raw_header: list[str], target_cols: int) -> list[str]:
        if target_cols < 8:
            return raw_header

        probe = " ".join(raw_header[:8]).lower()
        fold = str.maketrans({
            "ă": "a", "â": "a", "î": "i", "ș": "s", "ş": "s", "ț": "t", "ţ": "t",
        })
        probe_fold = probe.translate(fold)
        looks_like_hydro = (
            ("nr" in probe_fold and "raul" in probe_fold and ("stati" in probe_fold or "statie" in probe_fold))
            and ("cote" in probe_fold and "aparare" in probe_fold)
        )
        if not looks_like_hydro:
            return raw_header

        canonical_by_len: dict[int, list[str]] = {
            14: [
                "nr_crt",
                "raul",
                "statia_hidrometrica",
                "jud",
                "CA_cm",
                "CI_cm",
                "CP_cm",
                "Qmed_apr_m3_s",
                "diag_H_cm",
                "diag_Q_m3_s",
                "diag_dH_cm",
                "prog_H_cm",
                "prog_Q_m3_s",
                "prog_dH_cm",
            ],
            15: [
                "nr_crt",
                "raul",
                "statia_hidrometrica",
                "jud",
                "CA_cm",
                "CI_cm",
                "CP_cm",
                "Qmed_apr_m3_s",
                "diag_H_cm",
                "diag_Q_m3_s",
                "diag_dH_cm",
                "diag_G",
                "prog_H_cm",
                "prog_Q_m3_s",
                "prog_dH_cm",
            ],
            16: [
                "nr_crt",
                "raul",
                "statia_hidrometrica",
                "jud",
                "CA_cm",
                "CI_cm",
                "CP_cm",
                "Qmed_apr_m3_s",
                "diag_H_cm",
                "diag_Q_m3_s",
                "diag_dH_cm",
                "diag_G",
                "prog_H_cm",
                "prog_Q_m3_s",
                "prog_dH_cm",
                "prog_G",
            ],
        }

        if target_cols in canonical_by_len:
            return canonical_by_len[target_cols]

        # Fallback for unexpected column counts: preserve first 4 + generic numbered fields.
        out = ["nr_crt", "raul", "statia_hidrometrica", "jud"]
        while len(out) < target_cols:
            out.append(f"col_{len(out)}")
        return out
    data_start = 2 if len(block) > 1 and is_separator(block[1]) else 1

    max_cols = len(raw_header)
    for raw in block[data_start:]:
        max_cols = max(max_cols, len(row_cells(raw)))

    header = normalize_hydro_header(raw_header, max_cols)
    if len(header) < max_cols:
        header = header + [f"col_{i}" for i in range(len(header), max_cols)]

    out: list[dict[str, str]] = []

    for raw in block[data_start:]:
        cells = row_cells(raw)
        if not any(cells):
            continue
        if len(cells) < len(FIELDS_BEFORE_FIRST_GP):
            add_warning(
                warnings,
                "markdown_too_few_columns_pre_gp",
                "Row has fewer columns than required before first GP column; applied best-effort mapping.",
                row=raw,
                columns=len(cells),
                required=len(FIELDS_BEFORE_FIRST_GP),
            )
        if len(cells) > len(header):
            add_warning(
                warnings,
                "markdown_too_many_columns",
                "Row has more columns than header; extra columns were truncated.",
                row=raw,
                columns=len(cells),
                header_columns=len(header),
            )
        if len(cells) < len(header):
            cells = cells + [""] * (len(header) - len(cells))
        elif len(cells) > len(header):
            cells = cells[: len(header)]
        out.append(dict(zip(header, cells)))
    return out


def parse_html_table(text: str, warnings: Optional[list[dict[str, Any]]] = None) -> list[dict[str, str]]:
    soup = BeautifulSoup(text, "html.parser")
    table = soup.find("table")
    if table is None:
        add_warning(warnings, "html_table_missing", "No HTML <table> found in result text.")
        return []

    def is_probably_numeric(value: str) -> bool:
        v = value.strip().replace(" ", "")
        if not v:
            return False
        v = v.replace(",", ".")
        return re.fullmatch(r"[+-]?\d+(?:\.\d+)?", v) is not None

    def is_probably_gp_code(value: str) -> bool:
        v = value.strip().upper()
        if not v:
            return False
        # Typical ice/state codes seen in bulletins.
        known_codes = {"G", "P", "Z", "N", "S", "GP", "PG", "G/P", "P/G"}
        if v in known_codes:
            return True
        # Non-numeric non-empty text is much more likely GP than water level/debit.
        if not is_probably_numeric(v):
            return True
        return False

    data_cells_rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if cells and cells[0].isdigit():
            data_cells_rows.append(cells)

    if not data_cells_rows:
        add_warning(warnings, "html_table_no_data_rows", "HTML table found but no numeric data rows were detected.")
        return []

    len_dist: dict[int, int] = {}
    for cells in data_cells_rows:
        len_dist[len(cells)] = len_dist.get(len(cells), 0) + 1
    if len(len_dist) > 1:
        add_warning(
            warnings,
            "html_mixed_row_column_counts",
            "HTML table rows have mixed column counts.",
            distribution=len_dist,
        )

    # Detect the dominant 15-column layout once per table.
    # Variant A: diag_GP exists at col 12 (often empty), prog starts at col 13.
    # Variant B: diag_GP missing, prog starts at col 12.
    layout_15 = "variant_b"
    rows_15 = [cells for cells in data_cells_rows if len(cells) == 15]
    if rows_15:
        num11 = sum(1 for cells in rows_15 if is_probably_numeric(cells[11]))
        num12 = sum(1 for cells in rows_15 if is_probably_numeric(cells[12]))
        gp11 = sum(1 for cells in rows_15 if is_probably_gp_code(cells[11]))

        if num12 > num11 or gp11 > 0:
            layout_15 = "variant_a"

    rows = []
    for idx, cells in enumerate(data_cells_rows, start=1):
        try:
            row = empty_table_row()

            # Failsafe guarantee: all values before first GP are always mapped consecutively.
            pre_count = min(len(cells), len(FIELDS_BEFORE_FIRST_GP))
            for i in range(pre_count):
                row[FIELDS_BEFORE_FIRST_GP[i]] = cells[i]

            if len(cells) < len(FIELDS_BEFORE_FIRST_GP):
                add_warning(
                    warnings,
                    "html_too_few_columns_pre_gp",
                    "Row has fewer columns than required before first GP; used best-effort consecutive mapping.",
                    row_index=idx,
                    columns=len(cells),
                    required=len(FIELDS_BEFORE_FIRST_GP),
                )

            if len(cells) > 16:
                add_warning(
                    warnings,
                    "html_too_many_columns",
                    "Row has more than expected columns; extras were appended to prog_GP.",
                    row_index=idx,
                    columns=len(cells),
                    expected_max=16,
                )

            # Parse tail columns starting from first GP position (index 11).
            tail = cells[len(FIELDS_BEFORE_FIRST_GP):]
            if len(cells) >= 16:
                # Standard 16-col layout.
                row["diag_GP"] = tail[0] if len(tail) > 0 else ""
                row["prog_H_cm"] = tail[1] if len(tail) > 1 else ""
                row["prog_Q_m3_s"] = tail[2] if len(tail) > 2 else ""
                row["prog_dH_cm"] = tail[3] if len(tail) > 3 else ""
                row["prog_GP"] = tail[4] if len(tail) > 4 else ""
                if len(tail) > 5:
                    extras = " | ".join(x for x in tail[5:] if x)
                    if extras:
                        row["prog_GP"] = (row["prog_GP"] + " | " + extras).strip(" |")
            elif len(cells) == 15:
                # Two 15-col variants. Decide once at table level.
                if layout_15 == "variant_a":
                    # diag_GP present (possibly empty), prog_GP missing.
                    row["diag_GP"] = tail[0] if len(tail) > 0 else ""
                    row["prog_H_cm"] = tail[1] if len(tail) > 1 else ""
                    row["prog_Q_m3_s"] = tail[2] if len(tail) > 2 else ""
                    row["prog_dH_cm"] = tail[3] if len(tail) > 3 else ""
                    row["prog_GP"] = ""
                else:
                    # diag_GP missing, prog columns shifted left.
                    row["diag_GP"] = ""
                    row["prog_H_cm"] = tail[0] if len(tail) > 0 else ""
                    row["prog_Q_m3_s"] = tail[1] if len(tail) > 1 else ""
                    row["prog_dH_cm"] = tail[2] if len(tail) > 2 else ""
                    row["prog_GP"] = tail[3] if len(tail) > 3 else ""
            elif len(cells) == 14:
                # 14-col layout: no G/P columns, tail contains only prog_H/prog_Q/prog_dH.
                row["diag_GP"] = ""
                row["prog_H_cm"] = tail[0] if len(tail) > 0 else ""
                row["prog_Q_m3_s"] = tail[1] if len(tail) > 1 else ""
                row["prog_dH_cm"] = tail[2] if len(tail) > 2 else ""
                row["prog_GP"] = ""
            else:
                # Very short or unusual row: best-effort consecutive assignment after pre-GP.
                for j, value in enumerate(tail[: len(FIELDS_FROM_FIRST_GP)]):
                    row[FIELDS_FROM_FIRST_GP[j]] = value

            rows.append(row)
        except Exception as e:
            add_warning(
                warnings,
                "html_row_parse_error",
                "Exception while parsing HTML table row; applied full consecutive best-effort mapping.",
                row_index=idx,
                error=str(e),
            )
            rows.append(map_consecutive_best_effort(cells, ALL_TABLE_FIELDS))
    return rows


def parse_result_table(text: str, warnings: Optional[list[dict[str, Any]]] = None) -> list[dict[str, str]]:
    try:
        rows = parse_html_table(text, warnings=warnings)
    except Exception as e:
        add_warning(
            warnings,
            "html_parse_error",
            "Exception while parsing HTML table.",
            error=str(e),
        )
        rows = []
    if rows:
        return rows
    try:
        return parse_markdown_table(text, warnings=warnings)
    except Exception as e:
        add_warning(
            warnings,
            "markdown_parse_error",
            "Exception while parsing markdown table.",
            error=str(e),
        )
        return []


def save_table_rows(rows: list[dict[str, str]], generation_dir: Path) -> None:
    json_path = generation_dir / "table_rows.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if not rows:
        return

    csv_path = generation_dir / "table_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def list_bulletin_urls_root_pages(
    root_url: str,
    session: requests.Session,
    timeout_s: int,
    max_pages: int = 0,
) -> list[str]:
    root = root_url.rstrip("/") + "/"
    bulletin_urls: set[str] = set()
    page_num = 1

    while True:
        if max_pages > 0 and page_num > max_pages:
            break

        page_url = root if page_num == 1 else urljoin(root, f"page/{page_num}/")
        log(f"[LIST] {page_url}")

        try:
            resp = session.get(page_url, timeout=timeout_s)
        except requests.RequestException as e:
            log(f"[LIST] stop at page {page_num}: request failed ({e})")
            break

        if resp.status_code == 404:
            break
        if resp.status_code >= 400:
            log(f"[LIST] stop at page {page_num}: HTTP {resp.status_code}")
            break

        soup = BeautifulSoup(resp.content, "html.parser")
        page_bulletins: set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if not isinstance(href, str):
                continue
            if "/bulletin/prognoza-hidrologica-pentru-rauri" not in href:
                continue
            page_bulletins.add(urljoin(page_url, href))

        if not page_bulletins:
            # WordPress paginated endpoints usually stop returning bulletin links at the end.
            break

        bulletin_urls.update(page_bulletins)
        page_num += 1
        time.sleep(0.4)

    return sorted(bulletin_urls)


def generate_if_needed(
    source: SourceItem,
    store_dir: Path,
    server_url: str,
    mode: str,
    prompt_override: Optional[str],
    ignore_eos: bool,
    timeout_s: int,
    force_regenerate: bool,
) -> dict:
    session = requests.Session()
    session.headers.update(HEADERS)

    images_dir = store_dir / IMAGES_SUBDIR
    generations_dir = store_dir / GENERATIONS_SUBDIR
    images_dir.mkdir(parents=True, exist_ok=True)
    generations_dir.mkdir(parents=True, exist_ok=True)

    image_bytes = load_bytes(source.image_source, session=session, timeout_s=timeout_s)
    img_key = image_key(source.image_source, image_bytes)
    prompt = resolve_prompt(mode, prompt_override)
    key = generation_key(img_key, mode=mode, prompt=prompt, ignore_eos=ignore_eos)
    ext = guess_ext(source.image_source)
    local_image_path = images_dir / f"{img_key}{ext}"
    if not local_image_path.exists() or local_image_path.stat().st_size == 0:
        local_image_path.write_bytes(image_bytes)

    generation_dir = generations_dir / key
    generation_dir.mkdir(parents=True, exist_ok=True)
    result_path = generation_dir / "result.mmd"
    response_path = generation_dir / "response.json"
    meta_path = generation_dir / "meta.json"

    cached = result_path.exists() and result_path.stat().st_size > 0 and not force_regenerate
    if not cached:
        payload = {
            "image_path": str(local_image_path),
            "mode": mode,
            "prompt": prompt,
            "ignore_eos": ignore_eos,
        }
        resp = session.post(f"{server_url.rstrip('/')}/generate", json=payload, timeout=timeout_s)
        resp.raise_for_status()
        body = resp.json()
        result_path.write_text(body.get("text", ""), encoding="utf-8")
        response_path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")

    result_text = result_path.read_text(encoding="utf-8", errors="replace")
    parse_warnings: list[dict[str, Any]] = []
    table_rows = parse_result_table(result_text, warnings=parse_warnings)
    save_table_rows(table_rows, generation_dir)

    warning_summary: dict[str, int] = {}
    for w in parse_warnings:
        code = str(w.get("code", "unknown"))
        warning_summary[code] = warning_summary.get(code, 0) + 1

    meta = {
        "key": key,
        "image_key": img_key,
        "mode": mode,
        "ignore_eos": ignore_eos,
        "prompt": prompt,
        "source": source.image_source,
        "bulletin_url": source.bulletin_url,
        "bulletin_date": source.bulletin_date,
        "local_image_path": str(local_image_path),
        "result_path": str(result_path),
        "cached": cached,
        "table_rows": len(table_rows),
        "parse_warning_count": len(parse_warnings),
        "parse_warning_summary": warning_summary,
        "parse_warnings": parse_warnings[:25],
        "parse_warnings_truncated": max(0, len(parse_warnings) - 25),
        "updated_at_unix": int(time.time()),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def source_items_from_args(args: argparse.Namespace, session: requests.Session) -> list[SourceItem]:
    if args.image_manifest:
        manifest_path = Path(args.image_manifest).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Invalid JSON in image manifest: {manifest_path} ({e})") from e

        if not isinstance(raw, list):
            raise RuntimeError(f"Image manifest must be a JSON list: {manifest_path}")

        out: list[SourceItem] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue

            status = str(entry.get("status", ""))
            if status != "ok":
                continue

            local_path_value = str(entry.get("local_path", "")).strip()
            image_url_value = str(entry.get("image_url", "")).strip()

            source = ""
            if local_path_value:
                p = Path(local_path_value).expanduser().resolve()
                if p.exists() and p.is_file() and p.stat().st_size > 0:
                    source = str(p)
            if not source and image_url_value:
                source = image_url_value
            if not source:
                continue

            out.append(
                SourceItem(
                    image_source=source,
                    bulletin_url=str(entry.get("bulletin_url", "")),
                    bulletin_date=str(entry.get("bulletin_date", "")),
                )
            )

            if args.max_images > 0 and len(out) >= args.max_images:
                break

        if not out:
            raise RuntimeError(
                f"No usable entries found in image manifest: {manifest_path}. "
                "Expected entries with status='ok' and local_path or image_url."
            )
        return out

    if args.image:
        return [SourceItem(image_source=args.image)]

    if args.bulletin_url:
        meta = find_main_table_image(args.bulletin_url, session=session)
        if meta is None:
            raise RuntimeError(f"No main image found for bulletin: {args.bulletin_url}")
        return [
            SourceItem(
                image_source=meta.image_url,
                bulletin_url=meta.bulletin_url,
                bulletin_date=meta.bulletin_date,
            )
        ]

    bulletins = list_bulletin_urls_root_pages(
        root_url=args.root_url or DEFAULT_ROOT_URL,
        session=session,
        timeout_s=args.timeout,
        max_pages=args.max_pages,
    )

    # Keep a fallback to the old pagination discovery behavior if sequential page probing finds nothing.
    if not bulletins:
        bulletins = list_bulletin_urls(
            root_url=args.root_url or DEFAULT_ROOT_URL,
            max_pages=args.max_pages if args.max_pages > 0 else 1,
            session=session,
        )
    out: list[SourceItem] = []
    for bulletin_url in bulletins:
        meta = find_main_table_image(bulletin_url, session=session)
        if meta is None:
            continue
        out.append(
            SourceItem(
                image_source=meta.image_url,
                bulletin_url=meta.bulletin_url,
                bulletin_date=meta.bulletin_date,
            )
        )
        if args.max_images > 0 and len(out) >= args.max_images:
            break
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Client for DeepSeek OCR2 cached generations")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Local image path or URL")
    src.add_argument("--bulletin-url", help="Single bulletin URL")
    src.add_argument("--root-url", help="Hydro root URL")
    src.add_argument(
        "--image-manifest",
        help=(
            "Path to image manifest generated by hydro_image_finder_downloader.py "
            "(uses local_path when available, otherwise image_url)"
        ),
    )

    p.add_argument("--store-dir", default=STORE_DIR)
    p.add_argument("--server-url", default=SERVER_URL)
    p.add_argument("--mode", choices=["free_ocr", "grounding"], default=DEFAULT_MODE)
    p.add_argument("--prompt", default=None)
    p.add_argument("--ignore-eos", action="store_true", default=False)
    p.add_argument("--timeout", type=int, default=HTTP_TIMEOUT_SECONDS)
    p.add_argument("--max-pages", type=int, default=0, help="0 means crawl /page/N until no bulletin links are found")
    p.add_argument("--max-images", type=int, default=0, help="0 means no limit")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--force-regenerate", action="store_true")
    p.add_argument("--rivers", nargs="*", default=[], help="Optional river names to keep in CSV export")
    p.add_argument("--csv-dir", default="", help="Optional output directory for chunked CSV export")
    p.add_argument("--csv-prefix", default="rivers", help="Filename prefix for CSV shards")
    p.add_argument("--max-csv-rows", type=int, default=5000, help="Maximum rows per CSV shard before rolling over")
    p.add_argument("--manifest", default="", help="Optional output manifest path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    store_dir = Path(args.store_dir).expanduser().resolve()

    session = requests.Session()
    session.headers.update(HEADERS)

    items = source_items_from_args(args, session=session)
    if not items:
        log("No images to process.")
        return 0

    log(f"[FOUND] {len(items)} image(s)")
    results: list[dict] = []

    river_filters = {normalize_river_for_filter(x) for x in args.rivers if normalize_river_for_filter(x)}
    csv_dir = Path(args.csv_dir).expanduser().resolve() if args.csv_dir else (store_dir / "csv_exports")
    csv_writer = CsvShardWriter(output_dir=csv_dir, prefix=args.csv_prefix, max_rows=args.max_csv_rows)

    def export_rows_for_item(source: SourceItem, meta: dict[str, Any]) -> int:
        table_rows = load_generation_rows(meta)
        export_rows = [
            export_row_from_table_row(source, meta, row)
            for row in table_rows
            if should_keep_row_for_rivers(row, river_filters)
        ]
        if export_rows:
            csv_writer.write_rows(export_rows)
        return len(export_rows)

    exported_rows = 0

    if args.workers <= 1:
        for idx, item in enumerate(items, start=1):
            log(f"[{idx}/{len(items)}] {item.image_source}")
            res = generate_if_needed(
                source=item,
                store_dir=store_dir,
                server_url=args.server_url,
                mode=args.mode,
                prompt_override=args.prompt,
                ignore_eos=args.ignore_eos,
                timeout_s=args.timeout,
                force_regenerate=args.force_regenerate,
            )
            results.append(res)
            exported_rows += export_rows_for_item(item, res)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_to_item = {
                ex.submit(
                    generate_if_needed,
                    item,
                    store_dir,
                    args.server_url,
                    args.mode,
                    args.prompt,
                    args.ignore_eos,
                    args.timeout,
                    args.force_regenerate,
                ): item
                for item in items
            }
            for fut in concurrent.futures.as_completed(future_to_item):
                res = fut.result()
                item = future_to_item[fut]
                results.append(res)
                exported_rows += export_rows_for_item(item, res)

    csv_writer.close()

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else (store_dir / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    cached_count = sum(1 for x in results if x.get("cached"))
    log(
        f"[DONE] processed={len(results)} cached={cached_count} "
        f"manifest={manifest_path} csv_rows={exported_rows} csv_parts={len(csv_writer.files_written)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
