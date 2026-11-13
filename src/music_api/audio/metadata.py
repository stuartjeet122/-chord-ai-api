from __future__ import annotations

import base64
import hashlib
import logging
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from mutagen import File as MutagenFile
from mutagen import version_string as mutagen_version
from mutagen.flac import Picture
from PIL import Image, UnidentifiedImageError

from music_api.audio.toolchain import ProbeResult

logger = logging.getLogger(__name__)
MAX_ARTWORK_BYTES = 10 * 1024 * 1024
MAX_ARTWORK_DIMENSION = 6000
MAX_ARTWORK_PIXELS = 24_000_000
MAX_ARTWORKS = 4
MAX_RAW_TAGS = 128
MAX_TAG_VALUE_LENGTH = 4096

_TAG_NAMES: dict[str, tuple[str, ...]] = {
    "title": ("title", "tit2", "©nam"),
    "artist": ("artist", "tpe1", "©art"),
    "album": ("album", "talb", "©alb"),
    "album_artist": ("albumartist", "album artist", "tpe2", "aart"),
    "genre": ("genre", "tcon", "©gen"),
    "release_date": ("date", "year", "tyer", "tdrc", "©day"),
    "track_number": ("tracknumber", "track", "trck", "trkn"),
    "disc_number": ("discnumber", "disc", "tpos", "disk"),
    "composer": ("composer", "tcom", "©wrt"),
    "publisher": ("publisher", "tpub", "label"),
    "copyright": ("copyright", "tcop", "cprt"),
    "comment": ("comment", "comm", "©cmt", "description"),
    "encoder": ("encoder", "encodedby", "tenc", "©too"),
}
_TAG_LOOKUP = {
    alias.casefold(): normalized for normalized, aliases in _TAG_NAMES.items() for alias in aliases
}


@dataclass(frozen=True)
class Artwork:
    data: bytes
    mime_type: str
    picture_type: str | None
    description: str | None


@dataclass(frozen=True)
class ExtractedMetadata:
    tags: dict[str, Any]
    raw_tags: dict[str, Any]
    technical: dict[str, Any]
    artworks: list[Artwork]
    mutagen_version: str


def extract_metadata(path: Path, probe: ProbeResult) -> ExtractedMetadata:
    tags: dict[str, Any] = {}
    raw_tags: dict[str, Any] = {}
    artworks: list[Artwork] = []
    seen_art: set[str] = set()
    try:
        audio = MutagenFile(path, easy=False)
    except Exception:
        logger.exception("Mutagen could not read audio tags")
        audio = None

    if audio is not None:
        try:
            source_tags = audio.tags
            if source_tags is not None:
                for index, (raw_key, raw_value) in enumerate(source_tags.items()):
                    if index >= MAX_RAW_TAGS:
                        break
                    key = str(raw_key)[:128]
                    if key.upper().startswith("APIC") or key.casefold() in {
                        "covr",
                        "metadata_block_picture",
                    }:
                        raw_tags[key] = "[embedded artwork]"
                        for picture in _pictures_for_tag(key, raw_value):
                            _append_valid_artwork(picture, artworks, seen_art)
                        continue
                    rendered = _render_tag(raw_value)
                    raw_tags[key] = rendered
                    normalized = _TAG_LOOKUP.get(key.casefold())
                    if normalized is not None and normalized not in tags and rendered:
                        tags[normalized] = (
                            rendered[0]
                            if isinstance(rendered, list) and len(rendered) == 1
                            else rendered
                        )
                    for picture in _pictures_for_tag(key, raw_value):
                        _append_valid_artwork(picture, artworks, seen_art)
            for picture in getattr(audio, "pictures", ()):
                _append_valid_artwork(_artwork_from_object(picture), artworks, seen_art)
        except Exception:
            logger.exception("Could not extract all embedded metadata or artwork")

    technical = probe.as_dict()
    technical["container_tags"] = probe.raw_format_tags
    if audio is not None:
        info = getattr(audio, "info", None)
        bitrate_mode = str(getattr(info, "bitrate_mode", "")).casefold()
        if "vbr" in bitrate_mode:
            technical["variable_bitrate"] = True
        elif "cbr" in bitrate_mode:
            technical["variable_bitrate"] = False
    return ExtractedMetadata(
        tags=tags,
        raw_tags=raw_tags,
        technical=technical,
        artworks=artworks[:MAX_ARTWORKS],
        mutagen_version=mutagen_version,
    )


def _render_tag(value: Any) -> Any:
    if hasattr(value, "text"):
        value = value.text
    if isinstance(value, bytes):
        return value[:MAX_TAG_VALUE_LENGTH].decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_render_tag(item) for item in value[:16] if item is not None]
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)[:MAX_TAG_VALUE_LENGTH]
    return str(value)[:MAX_TAG_VALUE_LENGTH]


def _pictures_for_tag(key: str, value: Any) -> list[Artwork]:
    if key.upper().startswith("APIC"):
        try:
            return [_artwork_from_object(value)]
        except (
            OSError,
            ValueError,
            TypeError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ):
            logger.warning("Ignoring invalid APIC artwork frame")
            return []
    if key.casefold() == "covr":
        values = value if isinstance(value, (list, tuple)) else [value]
        pictures: list[Artwork] = []
        for item in values[:MAX_ARTWORKS]:
            try:
                pictures.append(_artwork_from_bytes(bytes(item), None, None))
            except (
                OSError,
                ValueError,
                TypeError,
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
            ):
                logger.warning("Ignoring invalid MP4 artwork frame")
        return pictures
    if key.casefold() == "metadata_block_picture":
        values = value if isinstance(value, (list, tuple)) else [value]
        block_pictures: list[Artwork] = []
        for item in values:
            try:
                encoded = item if isinstance(item, bytes) else str(item).encode("ascii")
                decoded = base64.b64decode(encoded, validate=True)
                block_pictures.append(
                    _artwork_from_object(Picture(decoded))  # type: ignore[no-untyped-call]
                )
            except Exception:
                logger.warning("Ignoring malformed embedded FLAC artwork block")
        return block_pictures
    if hasattr(value, "data") and hasattr(value, "mime"):
        try:
            return [_artwork_from_object(value)]
        except (
            OSError,
            ValueError,
            TypeError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ):
            logger.warning("Ignoring invalid embedded artwork frame")
            return []
    return []


def _artwork_from_object(value: Any) -> Artwork:
    data = bytes(value.data)
    mime = str(getattr(value, "mime", "") or "")
    picture_type = getattr(value, "type", None)
    description = getattr(value, "desc", None)
    return _artwork_from_bytes(
        data,
        mime or None,
        str(picture_type)[:32] if picture_type is not None else None,
        str(description)[:255] if description else None,
    )


def _artwork_from_bytes(
    data: bytes,
    claimed_mime: str | None,
    picture_type: str | None,
    description: str | None = None,
) -> Artwork:
    if len(data) > MAX_ARTWORK_BYTES:
        raise ValueError("Embedded artwork exceeds the configured extraction limit.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as image:
            detected_format = (image.format or "").upper()
            width, height = image.size
            if (
                width <= 0
                or height <= 0
                or width > MAX_ARTWORK_DIMENSION
                or height > MAX_ARTWORK_DIMENSION
                or width * height > MAX_ARTWORK_PIXELS
            ):
                raise ValueError("Embedded artwork dimensions exceed safe limits.")
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image.load()
    mime_by_format = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "GIF": "image/gif",
        "WEBP": "image/webp",
    }
    detected_mime = mime_by_format.get(detected_format)
    if detected_mime is None:
        raise ValueError("Embedded artwork uses an unsupported image format.")
    if claimed_mime and claimed_mime.casefold() not in {detected_mime, "image/jpg"}:
        logger.info("Embedded artwork MIME claim does not match detected image signature")
    return Artwork(data, detected_mime, picture_type, description)


def _append_valid_artwork(
    picture: Artwork,
    artworks: list[Artwork],
    seen: set[str],
) -> None:
    if len(artworks) >= MAX_ARTWORKS:
        return
    try:
        digest = hashlib.sha256(picture.data).hexdigest()
        if digest not in seen:
            seen.add(digest)
            artworks.append(picture)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        logger.warning("Ignoring invalid or unsafe embedded artwork")
