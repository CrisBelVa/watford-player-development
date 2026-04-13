from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _slugify(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "player"


def _player_cover_dir(base_dir: str, player_key: str) -> Path:
    root = Path(base_dir) / "data" / "pdf_cover_photos"
    player_dir = root / _slugify(player_key)
    player_dir.mkdir(parents=True, exist_ok=True)
    return player_dir


def list_cover_photos(base_dir: str, player_key: str) -> list[dict[str, Any]]:
    player_dir = _player_cover_dir(base_dir, player_key)
    files = [
        p
        for p in player_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    history: list[dict[str, Any]] = []
    for p in files:
        ts = datetime.fromtimestamp(p.stat().st_mtime)
        history.append(
            {
                "path": str(p.resolve()),
                "filename": p.name,
                "label": ts.strftime("%Y-%m-%d %H:%M") + f" - {p.name}",
                "timestamp": ts,
            }
        )
    return history


def save_cover_photo(
    uploaded_file: Any,
    base_dir: str,
    player_key: str,
    max_side_px: int = 900,
) -> str:
    player_dir = _player_cover_dir(base_dir, player_key)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = player_dir / f"{timestamp}.jpg"

    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    # Keep a square crop to render consistently on all PDF covers.
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))

    if image.width > max_side_px:
        image = image.resize((max_side_px, max_side_px), Image.Resampling.LANCZOS)

    image.save(output_path, format="JPEG", quality=88, optimize=True, progressive=True)
    return str(output_path.resolve())
