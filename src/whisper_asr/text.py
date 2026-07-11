"""Persian text normalization."""

from __future__ import annotations

try:
    from hazm import Normalizer as HazmNormalizer
except ImportError:
    HazmNormalizer = None  # type: ignore[misc]

_CHAR_FIXES = str.maketrans({
    "ي": "ی",
    "ك": "ک",
    "ة": "ه",
    "ؤ": "و",
    "أ": "ا",
    "إ": "ا",
    "ۀ": "ه",
    "ۀ": "ه",
})


def normalize(text: str | None, use_hazm: bool = False) -> str:
    if not text:
        return ""
    text = str(text)
    text = text.translate(_CHAR_FIXES)
    text = text.replace("‌", " ")  # half-space → space
    text = " ".join(text.split())
    result = text.strip()
    if use_hazm and HazmNormalizer is not None:
        result = HazmNormalizer().normalize(result)
    return result
