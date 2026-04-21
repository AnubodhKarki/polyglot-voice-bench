import re
import unicodedata


def normalize_english(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    A lightweight stand-in for whisper.normalizers.EnglishTextNormalizer.
    Sufficient for v0 WER comparisons. Will be replaced with a proper
    normalizer (number words, contractions) in v4.
    """
    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_devanagari(text: str) -> str:
    # TODO v4: strip punctuation, normalise Unicode composed/decomposed forms,
    # handle zero-width joiners, map visually-identical characters.
    return text
