"""Canonical detection records shared by annotation and JSON serialization."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectedObject:
    id: str
    label: str
    prompt_class_index: int
    confidence: float
    bbox: dict[str, int]
