"""Conservative category parsing; this is not a natural-language reasoning model.

Unknown noun phrases remain open vocabulary. Only explicitly listed aliases and
plural nouns are rewritten, never an arbitrary trailing ``s``.
"""

import json
from pathlib import Path
import re
import unicodedata

PARSER_VERSION = "1.1.0"
MAX_PROMPT_LENGTH = 256
MAX_CLASSES = 8
_ALIASES = json.loads(
    (Path(__file__).resolve().parents[1] / "config/language_aliases.json").read_text(encoding="utf-8")
)
_PLURALS = {
    "people": "person", "persons": "person", "men": "man", "women": "woman",
    "children": "child", "cars": "car", "motorcycles": "motorcycle",
    "motorbikes": "motorcycle", "bicycles": "bicycle", "bikes": "bicycle",
    "helmets": "helmet", "bottles": "bottle", "forklifts": "forklift",
    "pallets": "pallet", "boxes": "box", "trucks": "truck", "buses": "bus",
    "chairs": "chair", "tables": "table", "cats": "cat", "dogs": "dog",
    "bags": "bag", "shoes": "shoe", "vests": "vest", "cones": "cone",
}
_WRAPPER = re.compile(r"^(?:count|find|detect|hitung|cari|deteksi)\s+", re.I)
_ARTICLE = re.compile(r"^(?:(?:all|the|a|an|semua|seluruh)\s+|yang ada\s+)+", re.I)
_UNSUPPORTED = re.compile(
    r"\b(?:except|excluding|exclude|without|not|only|or|either|both|"
    r"holding|carrying|wearing|standing|sitting|walking|running|riding|touching|pushing|pulling|"
    r"next to|left of|right of|in front of|at least|at most|more than|less than|"
    r"how many|are there|kecuali|tanpa|atau|bukan|hanya|memegang|membawa|"
    r"memakai|mengenakan|berdiri|duduk|berjalan|sebelah|di kiri|di kanan|"
    r"di depan|di belakang|berapa|yang sedang)\b"
    r"|\s(?:on|in|under|over|above|below|behind|between|beside|near|inside|outside|with|di|dengan)\s",
    re.I,
)
_VAGUE = {"all", "everything", "anything", "something", "objects", "object", "things", "thing",
          "semua", "semuanya", "apapun", "benda", "objek"}
# Established category compounds are nouns, despite containing action words.
# This is a deliberately small grammar exception list, not a category whitelist.
_NOUN_COMPOUNDS = re.compile(r"\b(?:standing desks?|carrying cases?|walking sticks?|running shoes?|riding helmets?|sitting cushions?)\b")


def resolve_target_phrases(prompt: str) -> list[str]:
    """Return ordered English/common-alias class phrases or raise ``ValueError``.

    Canonical syntax is comma-separated visual categories. Simple command
    wrappers and ``and``/``dan`` are supported; relationships, exclusions,
    questions, and quantitative instructions are deliberately rejected.
    """
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be text containing explicit object categories.")
    if any(unicodedata.category(char).startswith("C") for char in prompt):
        raise ValueError("Prompt must not contain control characters.")
    prompt = unicodedata.normalize("NFC", prompt.strip())
    if not 1 <= len(prompt) <= MAX_PROMPT_LENGTH:
        raise ValueError(f"Prompt must contain 1 to {MAX_PROMPT_LENGTH} characters after trimming.")
    normalized = " ".join(prompt.casefold().split())
    if any(not (char.isalnum() or char.isspace() or char in ",-'’") for char in normalized):
        raise ValueError("Use comma-separated object categories without question marks or instruction punctuation.")
    normalized = _WRAPPER.sub("", normalized, count=1)
    pieces = re.split(r"\s+(?:and|dan)\s+|,", normalized)
    classes: list[str] = []
    for piece in pieces:
        phrase = _ARTICLE.sub("", piece.strip()).strip()
        phrase = re.sub(r"\s+yang ada$", "", phrase).strip()
        if _UNSUPPORTED.search(phrase) and not _NOUN_COMPOUNDS.fullmatch(phrase):
            raise ValueError("Relationships, exclusions, logic, and questions are unsupported. Use categories such as 'person, red helmet'.")
        if re.search(r"\b(?:and|dan|yang)\b", phrase):
            raise ValueError("Use complete category phrases separated by commas; incomplete conjunctions or relative clauses are unsupported.")
        if not any(char.isalpha() for char in phrase) or phrase in _VAGUE or phrase in {"count", "find", "detect", "hitung", "cari", "deteksi"}:
            raise ValueError("Specify explicit object categories, for example 'person, forklift'; empty or vague categories are unsupported.")
        words = phrase.split()
        if any(word.isdecimal() for word in words):
            raise ValueError("Quantitative instructions are unsupported; specify category names only.")
        phrase = _ALIASES.get(phrase, phrase)
        words = phrase.split()
        words[-1] = _PLURALS.get(words[-1], words[-1])
        phrase = " ".join(words)
        if phrase not in classes:
            classes.append(phrase)
    if len(classes) > MAX_CLASSES:
        raise ValueError(f"Request at most {MAX_CLASSES} different categories at a time.")
    return classes
