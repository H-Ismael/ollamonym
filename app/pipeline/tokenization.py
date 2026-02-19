"""Tokenization: placeholder insertion and token-to-original mapping."""

import logging
from typing import Dict, List

from app.schemas.models import EntityOccurrence

logger = logging.getLogger(__name__)


class Tokenizer:
    """Manages placeholder token insertion and mapping."""

    @staticmethod
    def insert_placeholders(
        text: str,
        spans: List[EntityOccurrence],
        placeholder_format: str,
    ) -> tuple[str, Dict[str, str]]:
        """
        Replace spans with placeholder tokens.

        Args:
            text: Original text
            spans: List of EntityOccurrence (sorted by position)
            placeholder_format: Format string e.g., "<<{ENTITY}:{ID}>>"

        Returns:
            (anonymized_text, token_to_original)
        """
        if not spans:
            return text, {}

        token_to_original = {}
        result_parts = []
        last_end = 0

        # Process spans right-to-left to maintain indices
        for span in reversed(spans):
            # Extract the token from the token field
            token = span.token
            token_to_original[token] = span.text

            # Build replacement
            before = text[last_end:span.start]
            result_parts.append(before)
            result_parts.append(token)
            last_end = span.end

        # Add remaining text
        result_parts.append(text[last_end:])

        # Reverse to get correct order (since we built right-to-left)
        result_parts.reverse()
        anonymized_text = "".join(result_parts)

        # Wait, reversing doesn't work that way. Let me fix this.
        # Actually, process left-to-right with offset adjustment

        token_to_original = {}
        result_parts = []
        last_end = 0

        for span in spans:
            token = span.token
            token_to_original[token] = span.text

            # Add text before span
            result_parts.append(text[last_end:span.start])
            # Add token
            result_parts.append(token)
            last_end = span.end

        # Add remaining text
        result_parts.append(text[last_end:])

        anonymized_text = "".join(result_parts)

        return anonymized_text, token_to_original
