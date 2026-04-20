#!/usr/bin/env python3
"""Lightweight hydro.ro bulletin image finder + downloader with resume support.

What this script does:
- discovers bulletin pages from a root listing (or processes one bulletin URL)
- extracts the most likely main table image URL per bulletin
- downloads images with retries, jittered backoff, and adaptive pacing
- stores persistent state so reruns can skip already discovered/downloaded items

Designed to be polite with WordPress infrastructure by default:
- single-threaded
- configurable base delay between requests
- longer delays after transient failures/timeouts
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

DEFAULT_ROOT_URL = "https://www.hidro.ro/bulletin_type/prognoza-hidrologica-pentru-rauri/"
DEFAULT_TIMEOUT_CONNECT = 20
DEFAULT_TIMEOUT_READ = 120
DEFAULT_BASE_DELAY = 0.45
DEFAULT_MAX_DELAY = 6.0
DEFAULT_MAX_RETRIES = 6

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    )
}


@dataclasses.dataclass
class BulletinImage:
    bulletin_url: str
    bulletin_date: str
    image_url: str


@dataclasses.dataclass
class RequestPacer:
    base_delay: float = DEFAULT_BASE_DELAY
    max_delay: float = DEFAULT_MAX_DELAY
    current_delay: float = DEFAULT_BASE_DELAY
    last_request_ts: float = 0.0

    def before_request(self) -> None:
        now = time.time()
        wait_for = (self.last_request_ts + self.current_delay) - now
        if wait_for > 0:
            # A bit of jitter prevents periodic thundering patterns.
            jitter = random.uniform(0.0, min(0.2, self.current_delay * 0.25))
            time.sleep(wait_for + jitter)

    def mark_success(self) -> None:
        self.last_request_ts = time.time()
        self.current_delay = max(self.base_delay, self.current_delay * 0.92)

    def mark_failure(self) -> None:
        self.last_request_ts = time.time()
        self.current_delay = min(self.max_delay, self.current_delay * 1.5)


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    pacer: RequestPacer,
    timeout: tuple[int, int],
    max_retries: int,
    stream: bool = False,
) -> requests.Response:
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        pacer.before_request()
        try:
            resp = session.request(method, url, timeout=timeout, stream=stream)

            # Retry common transient errors and rate-limits.
            if resp.status_code in {408, 425, 429, 500, 502, 503, 504}:
                pacer.mark_failure()
                backoff = min(20.0, (2 ** (attempt - 1)) * 0.6 + random.uniform(0.0, 0.8))
                log(f"[RETRY {attempt}/{max_retries}] HTTP {resp.status_code} for {url} (sleep {backoff:.2f}s)")
                time.sleep(backoff)
                continue

            resp.raise_for_status()
            pacer.mark_success()
            return resp
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_err = e
            pacer.mark_failure()
            if attempt >= max_retries:
                break
            backoff = min(20.0, (2 ** (attempt - 1)) * 0.8 + random.uniform(0.0, 1.0))
            log(f"[RETRY {attempt}/{max_retries}] {type(e).__name__} for {url} (sleep {backoff:.2f}s)")
            time.sleep(backoff)

    assert last_err is not None
    raise last_err


def parse_date_from_bulletin_url(url: str) -> str:
    m = re.search(r"intervalul-(\d{2})-(\d{2})-(\d{4})-ora", url)
    if not m:
        return ""
    dd, mm, yyyy = m.groups()
    return f"{yyyy}-{mm}-{dd}"


def has_resized_suffix(url: str) -> bool:
    path = urlparse(url).path.lower()
    return re.search(r"-\d{2,5}x\d{2,5}\.[a-z0-9]+$", path) is not None


def original_variant_url(url: str) -> str:
    parsed = urlparse(url)
    new_path = re.sub(r"-\d{2,5}x\d{2,5}(\.[a-z0-9]+)$", r"\1", parsed.path, flags=re.IGNORECASE)
    if new_path == parsed.path:
        return url
    return urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, parsed.query, parsed.fragment))


def absolute_url(base: str, maybe_relative: str) -> str:
    return urljoin(base, maybe_relative)


def discover_bulletins(
    root_url: str,
    session: requests.Session,
    pacer: RequestPacer,
    timeout: tuple[int, int],
    max_retries: int,
    max_pages: int,
) -> list[str]:
    root = root_url.rstrip("/") + "/"
    page_num = 1
    out: set[str] = set()

    while True:
        if max_pages > 0 and page_num > max_pages:
            break

        page_url = root if page_num == 1 else urljoin(root, f"page/{page_num}/")
        log(f"[LIST] {page_url}")

        try:
            resp = request_with_retry(
                session=session,
                method="GET",
                url=page_url,
                pacer=pacer,
                timeout=timeout,
                max_retries=max_retries,
            )
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 404:
                break
            raise

        soup = BeautifulSoup(resp.content, "html.parser")
        page_bulletins: set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if not isinstance(href, str):
                continue
            if "/bulletin/prognoza-hidrologica-pentru-rauri" in href:
                page_bulletins.add(urljoin(page_url, href))

        if not page_bulletins:
            break

        out.update(page_bulletins)
        page_num += 1

    return sorted(out)


def find_main_table_image(
    bulletin_url: str,
    session: requests.Session,
    pacer: RequestPacer,
    timeout: tuple[int, int],
    max_retries: int,
) -> Optional[BulletinImage]:
    resp = request_with_retry(
        session=session,
        method="GET",
        url=bulletin_url,
        pacer=pacer,
        timeout=timeout,
        max_retries=max_retries,
    )
    soup = BeautifulSoup(resp.content, "html.parser")

    image_candidates: list[tuple[float, str]] = []

    def add_candidate(src_url: str, base_score: float) -> None:
        score = base_score
        if has_resized_suffix(src_url):
            score -= 8
            original = original_variant_url(src_url)
            if original != src_url:
                image_candidates.append((score + 10, original))
        else:
            score += 4
        image_candidates.append((score, src_url))

    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue

        src = absolute_url(bulletin_url, src)
        width = int(img.get("width") or 0)
        height = int(img.get("height") or 0)

        score = 0.0
        if "wp-content/uploads" in src:
            score += 10
        if width >= 600 or height >= 800:
            score += 10
        if any(token in src.lower() for token in ["pg2", "640x890", "640x904", "3-10", "4-"]):
            score += 5
        if height > width:
            score += 5

        add_candidate(src, score)

        srcset = img.get("srcset")
        if srcset:
            for part in srcset.split(","):
                p = part.strip().split(" ")[0]
                if not p:
                    continue
                add_candidate(absolute_url(bulletin_url, p), score)

    if not image_candidates:
        return None

    # Deduplicate keeping highest score; shorter URL wins tie.
    best: dict[str, float] = {}
    for score, src in image_candidates:
        best[src] = max(score, best.get(src, -1e9))

    ranked = sorted(best.items(), key=lambda x: (-x[1], len(x[0])))
    image_url = ranked[0][0]
    return BulletinImage(
        bulletin_url=bulletin_url,
        bulletin_date=parse_date_from_bulletin_url(bulletin_url),
        image_url=image_url,
    )


def safe_stem_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name or "image"
    stem = Path(name).stem or "image"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "image"


def guess_ext(url: str, content_type: str) -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        return ext

    ctype = (content_type or "").lower()
    if "png" in ctype:
        return ".png"
    if "webp" in ctype:
        return ".webp"
    if "bmp" in ctype:
        return ".bmp"
    if "tiff" in ctype:
        return ".tiff"
    return ".jpg"


def download_image(
    image_url: str,
    out_dir: Path,
    session: requests.Session,
    pacer: RequestPacer,
    timeout: tuple[int, int],
    max_retries: int,
) -> dict[str, Any]:
    resp = request_with_retry(
        session=session,
        method="GET",
        url=image_url,
        pacer=pacer,
        timeout=timeout,
        max_retries=max_retries,
        stream=True,
    )

    ctype = resp.headers.get("Content-Type", "")
    ext = guess_ext(image_url, ctype)
    stem = safe_stem_from_url(image_url)
    digest = hashlib.md5(image_url.encode("utf-8")).hexdigest()[:10]

    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = out_dir / f"{stem}-{digest}{ext}"

    h = hashlib.sha256()
    total = 0
    with local_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 64):
            if not chunk:
                continue
            f.write(chunk)
            h.update(chunk)
            total += len(chunk)

    return {
        "local_path": str(local_path),
        "bytes": total,
        "sha256": h.hexdigest(),
        "content_type": ctype,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hydro image finder/downloader with robust resume")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--root-url", default=DEFAULT_ROOT_URL)
    src.add_argument("--bulletin-url", help="Process only this bulletin URL")

    p.add_argument("--state-file", default="deepseek_ocr_store/image_fetch_state.json")
    p.add_argument("--images-dir", default="deepseek_ocr_store/images")
    p.add_argument("--manifest-json", default="deepseek_ocr_store/image_manifest.json")

    p.add_argument("--max-pages", type=int, default=0, help="0 means crawl until pages stop")
    p.add_argument("--max-bulletins", type=int, default=0, help="0 means no limit")
    p.add_argument("--connect-timeout", type=int, default=DEFAULT_TIMEOUT_CONNECT)
    p.add_argument("--read-timeout", type=int, default=DEFAULT_TIMEOUT_READ)
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)

    p.add_argument("--base-delay", type=float, default=DEFAULT_BASE_DELAY)
    p.add_argument("--max-delay", type=float, default=DEFAULT_MAX_DELAY)

    p.add_argument("--force-rediscover", action="store_true", help="Ignore cached bulletin->image mapping")
    p.add_argument("--force-redownload", action="store_true", help="Download even if previous success exists")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    state_file = Path(args.state_file).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    manifest_json = Path(args.manifest_json).expanduser().resolve()

    state = load_json(state_file, default={"bulletins": {}, "downloads": {}})
    if "bulletins" not in state:
        state["bulletins"] = {}
    if "downloads" not in state:
        state["downloads"] = {}

    session = requests.Session()
    session.headers.update(HEADERS)

    pacer = RequestPacer(base_delay=args.base_delay, max_delay=args.max_delay, current_delay=args.base_delay)
    timeout = (args.connect_timeout, args.read_timeout)

    if args.bulletin_url:
        bulletin_urls = [args.bulletin_url]
    else:
        bulletin_urls = discover_bulletins(
            root_url=args.root_url,
            session=session,
            pacer=pacer,
            timeout=timeout,
            max_retries=args.max_retries,
            max_pages=args.max_pages,
        )

    if args.max_bulletins > 0:
        bulletin_urls = bulletin_urls[: args.max_bulletins]

    log(f"[FOUND BULLETINS] {len(bulletin_urls)}")

    manifest: list[dict[str, Any]] = []
    discovered_ok = 0
    downloaded_ok = 0

    for i, bulletin_url in enumerate(bulletin_urls, start=1):
        log(f"[DISCOVER {i}/{len(bulletin_urls)}] {bulletin_url}")

        cached = state["bulletins"].get(bulletin_url)
        rec: dict[str, Any]

        if cached and cached.get("status") == "ok" and not args.force_rediscover:
            rec = dict(cached)
        else:
            try:
                meta = find_main_table_image(
                    bulletin_url=bulletin_url,
                    session=session,
                    pacer=pacer,
                    timeout=timeout,
                    max_retries=args.max_retries,
                )
                if meta is None:
                    rec = {
                        "status": "no_image",
                        "bulletin_url": bulletin_url,
                        "bulletin_date": parse_date_from_bulletin_url(bulletin_url),
                        "image_url": "",
                        "updated_at": int(time.time()),
                    }
                else:
                    discovered_ok += 1
                    rec = {
                        "status": "ok",
                        "bulletin_url": meta.bulletin_url,
                        "bulletin_date": meta.bulletin_date,
                        "image_url": meta.image_url,
                        "updated_at": int(time.time()),
                    }
            except Exception as e:
                rec = {
                    "status": "discover_error",
                    "bulletin_url": bulletin_url,
                    "bulletin_date": parse_date_from_bulletin_url(bulletin_url),
                    "image_url": "",
                    "error": f"{type(e).__name__}: {e}",
                    "updated_at": int(time.time()),
                }

            state["bulletins"][bulletin_url] = rec
            save_json_atomic(state_file, state)

        image_url = rec.get("image_url", "")
        if rec.get("status") != "ok" or not image_url:
            manifest.append(rec)
            continue

        d_cached = state["downloads"].get(image_url)
        if (
            d_cached
            and d_cached.get("status") == "ok"
            and not args.force_redownload
            and Path(str(d_cached.get("local_path", ""))).exists()
        ):
            manifest.append({**rec, **d_cached})
            continue

        log(f"[DOWNLOAD] {image_url}")
        try:
            d = download_image(
                image_url=image_url,
                out_dir=images_dir,
                session=session,
                pacer=pacer,
                timeout=timeout,
                max_retries=args.max_retries,
            )
            downloaded_ok += 1
            drec = {
                "status": "ok",
                "image_url": image_url,
                "local_path": d["local_path"],
                "bytes": d["bytes"],
                "sha256": d["sha256"],
                "content_type": d["content_type"],
                "updated_at": int(time.time()),
            }
        except Exception as e:
            drec = {
                "status": "download_error",
                "image_url": image_url,
                "error": f"{type(e).__name__}: {e}",
                "updated_at": int(time.time()),
            }

        state["downloads"][image_url] = drec
        save_json_atomic(state_file, state)
        manifest.append({**rec, **drec})

    save_json_atomic(manifest_json, manifest)

    ok_discovered_total = sum(1 for x in state["bulletins"].values() if x.get("status") == "ok")
    ok_downloaded_total = sum(1 for x in state["downloads"].values() if x.get("status") == "ok")

    log(f"[RUN] discovered_ok={discovered_ok} downloaded_ok={downloaded_ok}")
    log(f"[TOTAL] discovered_ok={ok_discovered_total} downloaded_ok={ok_downloaded_total}")
    log(f"[STATE] {state_file}")
    log(f"[MANIFEST] {manifest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
