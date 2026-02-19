"""Text normalization and canonicalization utilities."""

import unicodedata
from typing import List, Tuple

from app.schemas.models import Canonicalizer


class TextNormalizer:
    """Handles text normalization and canonicalization."""

    @staticmethod
    def canonicalize(text: str, canon_config: Canonicalizer) -> str:
        """Apply canonicalization rules to text."""
        # Unicode normalization
        text = unicodedata.normalize(canon_config.unicode_normalize, text)

        # Collapse whitespace
        if canon_config.collapse_whitespace:
            text = " ".join(text.split())

        # Casefold
        if canon_config.casefold:
            text = text.casefold()

        # Strip outer punctuation
        if canon_config.strip_outer_punct:
            text = text.strip('.,;:!?\'"')

        return text

    @staticmethod
    def build_normalized_view(
        text: str, canon_config: Canonicalizer
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Build a normalized view of text for matching.
        Returns (normalized_text, mapping) where mapping[i] = (original_start, original_end)
        for each character position in normalized_text.
        """
        normalized = TextNormalizer.canonicalize(text, canon_config)
        
        # Simple mapping: for now, just track positions
        # This is a simplified version; in production, maintain character-level mapping
        mapping = []
        orig_idx = 0
        norm_idx = 0

        # Build character-by-character mapping
        char_map = []
        for orig_pos, char in enumerate(text):
            # Normalize this character
            norm_char = unicodedata.normalize(canon_config.unicode_normalize, char)
            if canon_config.casefold:
                norm_char = norm_char.casefold()

            char_map.append(orig_pos)

        return normalized, char_map


class SearchView:
    """Maintains a normalized search view with back-mapping to original."""

    def __init__(self, text: str, canon_config: Canonicalizer):
        self.original_text = text
        self.canon_config = canon_config
        self.normalized_text, self.norm_to_orig = self._normalize_with_mapping(text)

    def _normalize_with_mapping(self, text: str) -> Tuple[str, List[int]]:
        """
        Normalize text while preserving a map from each normalized char index
        back to its source index in the original text.
        """
        normalized_chars: List[str] = []
        mapping: List[int] = []
        last_was_space = False

        for orig_idx, char in enumerate(text):
            piece = unicodedata.normalize(self.canon_config.unicode_normalize, char)
            if self.canon_config.casefold:
                piece = piece.casefold()

            for out_char in piece:
                if self.canon_config.collapse_whitespace and out_char.isspace():
                    if normalized_chars and not last_was_space:
                        normalized_chars.append(" ")
                        mapping.append(orig_idx)
                        last_was_space = True
                    continue

                normalized_chars.append(out_char)
                mapping.append(orig_idx)
                last_was_space = False

        if self.canon_config.collapse_whitespace and normalized_chars and normalized_chars[-1] == " ":
            normalized_chars.pop()
            mapping.pop()

        return "".join(normalized_chars), mapping

    @staticmethod
    def _strip_outer_punct(text: str) -> str:
        """Apply the same outer punctuation trimming used by canonicalization."""
        return text.strip('.,;:!?\'"')

    def find_span(self, entity_text: str) -> List[Tuple[int, int]]:
        """
        Find all occurrences of entity_text in original text.
        Returns list of (start, end) tuples in original text coordinates.
        """
        normalized_entity, _ = self._normalize_with_mapping(entity_text)
        if self.canon_config.strip_outer_punct:
            normalized_entity = self._strip_outer_punct(normalized_entity)
        normalized_entity = normalized_entity.strip()

        if not normalized_entity:
            return []

        spans = []
        require_word_boundaries = self._requires_word_boundaries(normalized_entity)

        # Find all occurrences in normalized text
        start = 0
        while True:
            pos = self.normalized_text.find(normalized_entity, start)
            if pos == -1:
                break

            # Map back to original coordinates.
            end_pos = pos + len(normalized_entity) - 1
            if end_pos < len(self.norm_to_orig):
                orig_start = self.norm_to_orig[pos]
                orig_end = self.norm_to_orig[end_pos] + 1
                if require_word_boundaries and not self._has_word_boundaries(orig_start, orig_end):
                    start = pos + 1
                    continue
                spans.append((orig_start, orig_end))

            start = pos + 1

        return spans

    @staticmethod
    def _requires_word_boundaries(normalized_entity: str) -> bool:
        """Enable boundary checks for word-like entities to avoid substring hits."""
        return bool(normalized_entity) and all(
            ch.isalnum() or ch in {" ", "-", "'"} for ch in normalized_entity
        )

    def _has_word_boundaries(self, start: int, end: int) -> bool:
        """Validate that the span is not inside a larger alphanumeric token."""
        if start > 0 and self.original_text[start - 1].isalnum():
            return False
        if end < len(self.original_text) and self.original_text[end].isalnum():
            return False
        return True
