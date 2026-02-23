"""Deterministic span resolution using multi-pattern matching."""

import logging
from typing import List, Dict, Set, Tuple

from app.schemas.models import EntityOccurrence, Canonicalizer, ExtractedEntity
from app.utils.text_norm import SearchView

logger = logging.getLogger(__name__)


class SpanResolver:
    """Resolves entity spans deterministically, handling overlaps and repeats."""

    @staticmethod
    def resolve_spans(
        text: str,
        extracted_entities: List[ExtractedEntity],
        canon_config: Canonicalizer,
        entity_id_priority: Dict[str, int] = None,
    ) -> List[EntityOccurrence]:
        """
        Find all spans in text for extracted entities.
        Handles overlaps and repeats deterministically.

        Args:
            text: Original text
            extracted_entities: List of (entity_id, text) from LLM
            canon_config: Canonicalization config
            entity_id_priority: Optional priority dict for tie-breaking (higher = higher priority)

        Returns:
            List of EntityOccurrence with assigned tokens, sorted by position.
        """
        if not extracted_entities:
            return []

        # Build search view for text
        search_view = SearchView(text, canon_config)

        # Find all candidate spans
        candidates = []  # List of (start, end, entity_id, entity_text, priority)
        for entity in extracted_entities:
            spans = search_view.find_span(entity.text)
            priority = entity_id_priority.get(entity.entity_id, 0) if entity_id_priority else 0

            for start, end in spans:
                exact_surface = text[start:end]
                candidates.append((start, end, entity.entity_id, exact_surface, priority))

        # Sort candidates for deterministic overlap resolution
        # Longer spans first (end - start descending), then earlier start, then higher priority
        candidates.sort(
            key=lambda x: (-(x[1] - x[0]), x[0], -x[4]),
            reverse=False
        )

        # Greedy overlap resolution
        resolved = []
        used_spans: Set[Tuple[int, int]] = set()

        for start, end, entity_id, entity_text, priority in candidates:
            # Check for overlap with already selected spans
            overlaps = False
            for used_start, used_end in used_spans:
                if not (end <= used_start or start >= used_end):
                    overlaps = True
                    break

            if not overlaps:
                resolved.append(EntityOccurrence(
                    entity_id=entity_id,
                    text=entity_text,
                    start=start,
                    end=end,
                    token="",  # Placeholder; will be filled in
                ))
                used_spans.add((start, end))

        # Sort by position in text (for consistent replacement order)
        resolved.sort(key=lambda x: x.start)

        return resolved
