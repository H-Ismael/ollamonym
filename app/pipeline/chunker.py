"""Text chunking strategy for large texts."""

import logging
from typing import List

logger = logging.getLogger(__name__)


class TextChunker:
    """Chunks large texts for LLM processing."""

    @staticmethod
    def chunk_by_paragraphs(
        text: str,
        target_char_size: int,
        overlap: int = 0,
    ) -> List[str]:
        """
        Chunk text by paragraph boundaries.

        Args:
            text: Text to chunk
            target_char_size: Target size per chunk
            overlap: Character overlap between chunks (default 0)

        Returns:
            List of chunks
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para) + 2  # +2 for newlines

            if current_size + para_size > target_char_size and current_chunk:
                # Start new chunk
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        if not chunks:
            chunks = [text]

        if overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped_chunks = [chunks[0]]
        for idx in range(1, len(chunks)):
            prefix = chunks[idx - 1][-overlap:]
            overlapped_chunks.append(f"{prefix}{chunks[idx]}")

        return overlapped_chunks

    @staticmethod
    def chunk_by_sentences(
        text: str,
        target_char_size: int,
    ) -> List[str]:
        """
        Chunk text by sentence boundaries (simple heuristic).

        Args:
            text: Text to chunk
            target_char_size: Target size per chunk

        Returns:
            List of chunks
        """
        # Simple sentence splitting on .[space], ![space], ?[space]
        sentences = []
        current = []
        for char in text:
            current.append(char)
            if char in ".!?" and len(current) > 1:
                sentences.append("".join(current).strip())
                current = []
        if current:
            sentences.append("".join(current).strip())

        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_size = 0

        for sent in sentences:
            sent_size = len(sent) + 1
            if current_size + sent_size > target_char_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_size = sent_size
            else:
                current_chunk.append(sent)
                current_size += sent_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [text]

    @staticmethod
    def should_chunk(text: str, target_char_size: int) -> bool:
        """Determine if text should be chunked."""
        return len(text) > target_char_size
