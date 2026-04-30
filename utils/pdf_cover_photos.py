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


def build_cover_player_key(player_name: str | None = None, player_id: Any | None = None) -> str:
    """
    Shared cover key used across different PDF modules/pages.
    Prefer name so Player Dashboard and Individual Development resolve to the same folder.
    """
    if isinstance(player_name, str) and player_name.strip():
        return player_name.strip()
    if player_id is not None:
        return f"player-{player_id}"
    return "player"


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
    image = _normalize_cover_image(image=image, max_side_px=max_side_px)
    image.save(output_path, format="JPEG", quality=88, optimize=True, progressive=True)
    return str(output_path.resolve())


def _normalize_cover_image(image: Image.Image, max_side_px: int = 900) -> Image.Image:
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

    return image


def migrate_cover_photo_from_path(
    image_path: str,
    base_dir: str,
    player_key: str,
    max_side_px: int = 900,
) -> str:
    """
    Import an existing image file into the shared cover-photo history for player_key.
    """
    player_dir = _player_cover_dir(base_dir, player_key)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = player_dir / f"{timestamp}.jpg"

    image = Image.open(image_path)
    image = _normalize_cover_image(image=image, max_side_px=max_side_px)
    image.save(output_path, format="JPEG", quality=88, optimize=True, progressive=True)
    return str(output_path.resolve())
