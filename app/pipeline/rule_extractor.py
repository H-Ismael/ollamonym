"""Deterministic regex-based extraction for patterned entities."""

import re
from typing import List, Set

from app.schemas.models import ExtractedEntity


class RuleExtractor:
    """Regex extraction for high-confidence patterned PII."""

    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )
    PHONE_PATTERN = re.compile(
        r"(?:(?:\+?\d{1,3}[\s().-]*)?(?:\(?\d{3}\)?[\s().-]*)?\d{3}[\s.-]?\d{4})"
    )

    @classmethod
    def extract(cls, text: str, enabled_entity_ids: Set[str]) -> List[ExtractedEntity]:
        """Extract rule-based entities for enabled IDs."""
        entities: List[ExtractedEntity] = []

        if "EMAIL" in enabled_entity_ids:
            for match in cls.EMAIL_PATTERN.finditer(text):
                entities.append(
                    ExtractedEntity(entity_id="EMAIL", text=match.group(0))
                )

        if "PHONE" in enabled_entity_ids:
            for match in cls.PHONE_PATTERN.finditer(text):
                raw = match.group(0).strip()
                digit_count = sum(1 for ch in raw if ch.isdigit())
                if digit_count >= 7:
                    entities.append(
                        ExtractedEntity(entity_id="PHONE", text=raw)
                    )

        return entities
