"""Hashing and token ID generation utilities."""

import hmac
import hashlib
import base64
import logging

logger = logging.getLogger(__name__)


class TokenIDGenerator:
    """Generates deterministic, collision-resistant token IDs."""

    @staticmethod
    def generate_token_id(
        secret: str,
        session_id: str,
        entity_id: str,
        canon_text: str,
        token_id_len: int = 6,
        suffix: int = 0,
    ) -> str:
        """
        Generate a deterministic token ID using HMAC.

        Args:
            secret: Secret key for HMAC
            session_id: Session identifier
            entity_id: Entity type (e.g., PERSON)
            canon_text: Canonicalized entity text
            token_id_len: Length of base32 token ID
            suffix: Retry suffix (0, 1, 2, ...)

        Returns:
            Token ID string (e.g., "K7D2QH")
        """
        # Build deterministic input
        suffix_str = f"|#{suffix}" if suffix > 0 else ""
        input_str = f"{session_id}|{entity_id}|{canon_text}{suffix_str}"

        # HMAC-SHA256
        digest = hmac.new(
            secret.encode("utf-8"),
            input_str.encode("utf-8"),
            hashlib.sha256
        ).digest()

        # Base32 encode and take first token_id_len chars
        b32_str = base64.b32encode(digest).decode("utf-8").rstrip("=")
        token_id = b32_str[:token_id_len]

        return token_id

    @staticmethod
    def format_token(entity_id: str, token_id: str, format_str: str = "<<{ENTITY}:{ID}>>") -> str:
        """
        Format a token using the placeholder format string.

        Args:
            entity_id: Entity type
            token_id: Token ID
            format_str: Format string (e.g., "<<{ENTITY}:{ID}>>")

        Returns:
            Formatted token (e.g., "<<PERSON:K7D2QH>>")
        """
        return format_str.format(ENTITY=entity_id, ID=token_id)
