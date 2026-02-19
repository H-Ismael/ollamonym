"""Entity normalization and deduplication."""

import logging
from typing import List, Tuple

from app.schemas.models import ExtractedEntity
from app.utils.text_norm import TextNormalizer
from app.schemas.models import Canonicalizer

logger = logging.getLogger(__name__)


class Normalizer:
    """Normalizes and deduplicates extracted entities."""

    @staticmethod
    def normalize_entities(
        entities: List[ExtractedEntity],
        canon_config: Canonicalizer,
    ) -> List[ExtractedEntity]:
        """
        Deduplicate entities based on canonical form.

        Args:
            entities: List of extracted entities (may contain duplicates)
            canon_config: Canonicalization config

        Returns:
            Deduplicated list (keeping first occurrence of each canonical form)
        """
        seen: set[Tuple[str, str]] = set()  # (entity_id, canonical_form)

        result = []
        for entity in entities:
            canon_text = TextNormalizer.canonicalize(entity.text, canon_config)
            key = (entity.entity_id, canon_text)

            if key not in seen:
                seen.add(key)
                result.append(entity)

        return result

    @staticmethod
    def merge_chunk_results(
        chunk_results: List[List[ExtractedEntity]],
        canon_config: Canonicalizer,
    ) -> List[ExtractedEntity]:
        """
        Merge entities from multiple chunks, deduplicating.

        Args:
            chunk_results: List of entity lists (one per chunk)
            canon_config: Canonicalization config

        Returns:
            Deduplicated merged entity list
        """
        all_entities = []
        for chunk_entities in chunk_results:
            all_entities.extend(chunk_entities)

        return Normalizer.normalize_entities(all_entities, canon_config)
