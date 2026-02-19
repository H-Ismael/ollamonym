"""Main detection and anonymization pipeline."""

import logging
import re
from typing import Dict, List, Optional

from app.schemas.models import (
    AnonymizeRequest,
    ExtractedEntity,
)
from app.llm.ollama_client import OllamaClient
from app.llm.inference_queue import InferenceQueue
from app.pipeline.alias_postpass import AliasPostPass
from app.pipeline.normalizer import Normalizer
from app.pipeline.rule_extractor import RuleExtractor
from app.pipeline.span_resolver import SpanResolver
from app.pipeline.tokenization import Tokenizer
from app.pipeline.rendering import RealisticRenderer
from app.pipeline.chunker import TextChunker
from app.templates.registry import TemplateRegistry
from app.templates.compiler import PromptCompiler
from app.utils.hashing import TokenIDGenerator
from app.utils.text_norm import TextNormalizer

logger = logging.getLogger(__name__)
TLD_FRAGMENT_RE = re.compile(r"^[a-z]{2,6}$", flags=re.IGNORECASE)


class Detector:
    """Main anonymization pipeline orchestrator."""

    def __init__(
        self,
        template_registry: TemplateRegistry,
        ollama_client: OllamaClient,
        pseudonym_secret: str,
        token_id_len: int = 6,
        chunking_enabled: bool = True,
        chunk_char_target: int = 8000,
        chunk_overlap_chars: int = 200,
        chunk_max_parallel: int = 2,
        rule_preextract_enabled: bool = True,
    ):
        """
        Initialize detector.

        Args:
            template_registry: Registry of templates
            ollama_client: Ollama client
            pseudonym_secret: Secret for token generation
            token_id_len: Length of token IDs
            chunking_enabled: Whether to enable chunking
            chunk_char_target: Target chunk size in characters
            chunk_overlap_chars: Overlap characters between neighboring chunks
            chunk_max_parallel: Max in-flight chunk extraction calls
            rule_preextract_enabled: Enable deterministic pre-extraction for patterned entities
        """
        self.registry = template_registry
        self.ollama = ollama_client
        self.pseudonym_secret = pseudonym_secret
        self.token_id_len = token_id_len
        self.chunking_enabled = chunking_enabled
        self.chunk_char_target = chunk_char_target
        self.chunk_overlap_chars = max(0, chunk_overlap_chars)
        self.chunk_max_parallel = max(1, chunk_max_parallel)
        self.rule_preextract_enabled = rule_preextract_enabled
        self.compiler = PromptCompiler()
        self.inference_queue = InferenceQueue(max_concurrency=self.chunk_max_parallel)
        self.alias_postpass = AliasPostPass()

    def detect_and_anonymize(
        self,
        request: AnonymizeRequest,
    ) -> tuple[str, Dict[str, str], Optional[Dict[str, str]], Optional[Dict[str, str]]]:
        """
        Full anonymization pipeline.

        Args:
            request: Anonymization request

        Returns:
            (anonymized_text, token_to_original, token_to_fake, fake_to_token)
        """
        # Load template
        template = self.registry.get_template(request.template_id)
        if not template:
            raise ValueError(f"Template not found: {request.template_id}")

        # Step 1: Extract entities (hybrid rules + LLM)
        entities: List[ExtractedEntity] = []
        enabled_entity_ids = {
            entity.id for entity in template.entities if entity.enabled
        }

        if self.rule_preextract_enabled:
            entities.extend(RuleExtractor.extract(request.text, enabled_entity_ids))

        entities.extend(self._extract_entities(request.text, template))

        # Step 2: Normalize (deduplicate)
        entities = Normalizer.normalize_entities(entities, template.canon)
        entities = self._filter_spurious_structured_entities(entities)
        if template.postpass_alias and template.postpass_alias.enabled:
            # Post-LLM pass only: after all chunk/model extraction is complete.
            entities = self.alias_postpass.augment(
                request.session_id,
                request.text,
                entities,
                template.canon,
                template.postpass_alias,
            )
            entities = Normalizer.normalize_entities(entities, template.canon)

        # Step 3: Resolve spans
        spans = SpanResolver.resolve_spans(
            request.text,
            entities,
            template.canon,
        )

        # Step 4: Assign tokens and build mappings
        token_to_original = {}
        for span in spans:
            canon_text = TextNormalizer.canonicalize(span.text, template.canon)
            token_id = TokenIDGenerator.generate_token_id(
                self.pseudonym_secret,
                request.session_id,
                span.entity_id,
                canon_text,
                self.token_id_len,
            )
            token = TokenIDGenerator.format_token(
                span.entity_id,
                token_id,
                template.replacement.placeholder_format,
            )
            span.token = token
            token_to_original[token] = span.text

        # Step 5: Insert placeholders
        anonymized_text, _ = Tokenizer.insert_placeholders(
            request.text,
            spans,
            template.replacement.placeholder_format,
        )

        # Step 6: Realistic rendering (optional)
        token_to_fake = None
        fake_to_token = None
        if request.render_mode == "realistic":
            provider_map = None
            if template.replacement.pseudonym and template.replacement.pseudonym.providers:
                provider_map = template.replacement.pseudonym.providers
            renderer = RealisticRenderer(
                self.pseudonym_secret,
                request.session_id,
                provider_map=provider_map,
            )
            token_to_fake = {}
            fake_to_token = {}

            for token, original in token_to_original.items():
                # Extract entity_id and token_id from token
                # Format: <<{ENTITY}:{ID}>>
                # Extract entity_id from token (hacky but works)
                if token.startswith("<<") and token.endswith(">>"):
                    inner = token[2:-2]
                    entity_id, token_id = inner.split(":")
                    group_key = self.alias_postpass.resolve_group_key(
                        request.session_id,
                        entity_id,
                        original,
                        template.canon,
                    )
                    fake = renderer.generate_fake(
                        entity_id,
                        token_id,
                        original=original,
                        group_key=group_key or token_id,
                    )
                    token_to_fake[token] = fake
                    fake_to_token[fake] = token
                    anonymized_text = anonymized_text.replace(token, fake)

        return anonymized_text, token_to_original, token_to_fake, fake_to_token

    def _extract_entities(
        self,
        text: str,
        template,
    ) -> List[ExtractedEntity]:
        """Extract entities using LLM."""
        if self.chunking_enabled and TextChunker.should_chunk(text, self.chunk_char_target):
            return self._extract_entities_chunked(text, template)
        else:
            return self._extract_entities_single(text, template)

    def _extract_entities_single(
        self,
        text: str,
        template,
    ) -> List[ExtractedEntity]:
        """Extract from a single text (no chunking)."""
        system_msg = self.compiler.get_cached_system_message(template)
        user_msg = self.compiler.compile_user_message(template, text)

        try:
            entities = self.ollama.extract_entities(system_msg, user_msg)
            return entities
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            raise

    def _extract_entities_chunked(
        self,
        text: str,
        template,
    ) -> List[ExtractedEntity]:
        """Extract from chunked text and merge."""
        chunks = TextChunker.chunk_by_paragraphs(
            text,
            self.chunk_char_target,
            overlap=self.chunk_overlap_chars,
        )
        logger.info(f"Chunked text into {len(chunks)} chunks")

        system_msg = self.compiler.get_cached_system_message(template)
        tasks = []

        for idx, chunk in enumerate(chunks):
            user_msg = self.compiler.compile_user_message(
                template, chunk, idx, len(chunks)
            )

            def task_fn(system_msg=system_msg, user_msg=user_msg, idx=idx):
                try:
                    return self.ollama.extract_entities(system_msg, user_msg)
                except Exception as e:
                    logger.error(f"LLM extraction failed for chunk {idx}: {e}")
                    raise

            tasks.append(task_fn)

        chunk_results = self.inference_queue.run_batch_sync(tasks)
        return Normalizer.merge_chunk_results(chunk_results, template.canon)

    def shutdown(self):
        """Release detector resources."""
        self.inference_queue.shutdown()

    @staticmethod
    def _filter_spurious_structured_entities(
        entities: List[ExtractedEntity],
    ) -> List[ExtractedEntity]:
        """
        Drop low-signal fragments for URL/LINK-like entity classes.
        Example: bare 'com' emitted as LINKS should be ignored.
        """
        filtered: List[ExtractedEntity] = []
        for entity in entities:
            entity_id = entity.entity_id.upper()
            text = entity.text.strip()

            is_link_like = any(tag in entity_id for tag in ("LINK", "URL"))
            if is_link_like:
                if TLD_FRAGMENT_RE.fullmatch(text) and "." not in text and "/" not in text:
                    continue
            filtered.append(entity)
        return filtered
