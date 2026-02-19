"""Deanonymization: reverse mapping."""

import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class Deanonymizer:
    """Reverses anonymized text back to original."""

    @staticmethod
    def deanonymize(
        text: str,
        token_to_original: Optional[Dict[str, str]] = None,
        fake_to_token: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Reverse anonymized text back to original.

        Args:
            text: Anonymized text (may contain tokens or fakes)
            token_to_original: Mapping from tokens to originals
            fake_to_token: Mapping from fakes to tokens (optional)

        Returns:
            Original text
        """
        if not token_to_original:
            token_to_original = {}
        if not fake_to_token:
            fake_to_token = {}

        result = text

        # First, try to replace tokens directly
        if token_to_original:
            for token, original in token_to_original.items():
                result = result.replace(token, original)

        # If there are fake-to-token mappings, try those (for realistic mode input)
        if fake_to_token:
            for fake, token in fake_to_token.items():
                if token in token_to_original:
                    # Replace fake with original (via token)
                    result = result.replace(fake, token_to_original[token])

        return result
