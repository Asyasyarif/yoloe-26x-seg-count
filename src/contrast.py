"""Prompt-agnostic contrast classes chosen from a generic vocabulary.

YOLOE scores each class with an independent sigmoid, so a lone requested
category has no competitor and can label any look-alike region with that
category. Instead of a hand-curated map, the detector embeds a large generic
vocabulary once with the model's own text encoder and, per request, adds the
nearest vocabulary entries as hidden competitors.

Only the comparison lives here; the caller embeds with the model and keeps the
per-request cost at one small text-encoder pass.
"""

from __future__ import annotations

import json
import math
import operator
from pathlib import Path

VOCAB_PATH = Path(__file__).resolve().parents[1] / "config/contrast_vocab.json"

# Competitors per request. Prompt classes plus this many stay inside the model's
# 16-class budget; measured per-request cost is one short text-encoder pass.
CONTRAST_PER_REQUEST = 8
# Vocabulary entries at or above this cosine similarity are synonyms of the
# request and never used as competitors. Prompt aliases share MobileCLIP2's text
# space, so "women"/"woman" and "aeroplane"/"airplane" land above it.
SYNONYM_SIMILARITY = 0.90
# Near-duplicates of an already chosen competitor are skipped for the same reason.
CONTRAST_DUPLICATE_SIMILARITY = 0.95


def normalize_label(name: str) -> str:
    """Lowercase, singular, alphanumeric words: "Red Helmet" == "red helmets"."""
    lowered = "".join(char if char.isalnum() else " " for char in str(name).casefold())
    return " ".join(word[:-1] if len(word) > 3 and word.endswith("s") else word
                    for word in lowered.split())


def load_vocab(path: str | Path = VOCAB_PATH) -> list[str]:
    """Read the generic vocabulary, keeping order so selection stays deterministic."""
    try:
        vocab = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Invalid contrast vocabulary: {path}") from exc
    if (not isinstance(vocab, list) or not vocab
            or any(not isinstance(name, str) or not name.strip() for name in vocab)
            or len({normalize_label(name) for name in vocab}) != len(vocab)):
        raise RuntimeError(f"Invalid contrast vocabulary: {path}")
    return vocab


def cosine(first: list[float], second: list[float]) -> float:
    """Cosine similarity, matching the text encoder's unit-length features.

    shortcut: pure Python, ~40ms per requested class over the 1198-entry
    vocabulary (measured: 36ms for 1 class, 265ms for the 8-class maximum), well
    under one GPU inference. Switch to a numpy matrix product if the vocabulary
    or the per-request class count grows.
    """
    dot = sum(map(operator.mul, first, second))
    norm = math.sqrt(sum(map(operator.mul, first, first)) * sum(map(operator.mul, second, second)))
    return dot / norm if norm else 0.0


def select_contrast(
    entities: list[str],
    entity_embeddings: list[list[float]],
    vocab: list[str],
    vocab_embeddings: list[list[float]],
    limit: int = CONTRAST_PER_REQUEST,
    blocked: list[str] | None = None,
) -> list[str]:
    """Return up to ``limit`` nearest vocabulary entries, most similar first.

    ``entity_embeddings`` matches ``entities``. ``blocked`` are additional labels
    to keep out (things already requested or already chosen for other entities).
    Synonyms — same normalized label, or similarity at/above
    ``SYNONYM_SIMILARITY`` — are skipped. Ranking is by best similarity, then
    request order, then vocabulary index, so equal scores never reorder.
    """
    if limit <= 0 or not entities:
        return []
    if len(entities) != len(entity_embeddings):
        raise ValueError("Each entity needs one embedding.")
    excluded = {normalize_label(name) for name in [*entities, *(blocked or [])]}
    scored = []
    for index, name in enumerate(vocab):
        if normalize_label(name) in excluded:
            continue
        similarities = [cosine(embedding, vocab_embeddings[index]) for embedding in entity_embeddings]
        if max(similarities) >= SYNONYM_SIMILARITY:
            continue
        best = max(similarities)
        scored.append((-best, similarities.index(best), index, name))
    scored.sort()

    chosen: list[str] = []
    chosen_indices: list[int] = []
    for _, _, index, name in scored:
        if len(chosen) >= limit:
            break
        if any(cosine(vocab_embeddings[index], vocab_embeddings[other]) >= CONTRAST_DUPLICATE_SIMILARITY
               for other in chosen_indices):
            continue
        chosen.append(name)
        chosen_indices.append(index)
    return chosen
