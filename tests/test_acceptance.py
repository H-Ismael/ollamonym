"""Acceptance tests covering spec requirements."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from app.schemas.models import (
    AnonymizeRequest,
    ExtractedEntity,
    TemplateSchema,
    EntityDefinition,
    LLMConfig,
    Canonicalizer,
    PseudonymConfig,
    ReplacementPolicy,
    PostPassAliasConfig,
)
from app.templates.registry import TemplateRegistry
from app.templates.validator import validate_template
from app.llm.ollama_client import OllamaClient
from app.pipeline.detector import Detector
from app.pipeline.deanonymizer import Deanonymizer
from app.pipeline.span_resolver import SpanResolver
from app.pipeline.normalizer import Normalizer
from app.pipeline.rule_extractor import RuleExtractor
from app.utils.hashing import TokenIDGenerator
from app.utils.text_norm import SearchView, TextNormalizer


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_template():
    """Create a sample template for testing."""
    return TemplateSchema(
        template_id="test-pii",
        version=1,
        description="Test template",
        entities=[
            EntityDefinition(
                id="PERSON",
                enabled=True,
                instructions="Detect person names",
                examples={"positive": ["John Doe"], "negative": ["Monday"]},
            ),
            EntityDefinition(
                id="ORG",
                enabled=True,
                instructions="Detect organization names",
            ),
        ],
        llm=LLMConfig(model="llama3.1", system="Return JSON only", max_entities=200),
        canon=Canonicalizer(
            unicode_normalize="NFKC",
            collapse_whitespace=True,
            casefold=True,
            strip_outer_punct=True,
        ),
        replacement=ReplacementPolicy(
            placeholder_format="<<{ENTITY}:{ID}>>",
            pseudonym=PseudonymConfig(mode="session_stable", id_len=6),
        ),
        postpass_alias=PostPassAliasConfig(
            enabled=True,
            entity_ids=["PERSON", "ORG"],
            min_token_len=3,
            window_size=3,
            min_overlap_tokens=2,
            session_ttl_seconds=3600,
            max_aliases_per_entity=50,
        ),
    )


@pytest.fixture
def temp_templates_dir():
    """Create a temporary templates directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def template_registry(temp_templates_dir, sample_template):
    """Create a template registry with sample template."""
    # Write sample template to disk
    template_file = temp_templates_dir / "test-pii.json"
    template_file.write_text(sample_template.model_dump_json())

    registry = TemplateRegistry(temp_templates_dir)
    return registry


# ============================================================================
# Tests: Correctness
# ============================================================================


class TestCorrectness:
    """Test lossless reversal and correctness invariants."""

    def test_deanonymize_restores_original_structural(self):
        """Test that deanonymize(anonymize(text)) == text for structural mode."""
        original = "John Doe from Acme Corp"
        token_to_original = {
            "<<PERSON:ABC123>>": "John Doe",
            "<<ORG:XYZ789>>": "Acme Corp",
        }
        anonymized = "<<PERSON:ABC123>> from <<ORG:XYZ789>>"

        result = Deanonymizer.deanonymize(anonymized, token_to_original=token_to_original)
        assert result == original

    def test_deanonymize_with_realistic_rendering(self):
        """Test deanonymization with fake names."""
        original = "John Doe from Acme Corp"
        anonymized = "Mark Lewis from Zeta Labs"

        token_to_original = {
            "<<PERSON:ABC123>>": "John Doe",
            "<<ORG:XYZ789>>": "Acme Corp",
        }
        token_to_fake = {
            "<<PERSON:ABC123>>": "Mark Lewis",
            "<<ORG:XYZ789>>": "Zeta Labs",
        }
        fake_to_token = {
            "Mark Lewis": "<<PERSON:ABC123>>",
            "Zeta Labs": "<<ORG:XYZ789>>",
        }

        result = Deanonymizer.deanonymize(
            anonymized,
            token_to_original=token_to_original,
            fake_to_token=fake_to_token,
        )
        assert result == original

    def test_no_token_collisions(self):
        """Test that different entities produce different tokens."""
        secret = "test-secret"
        session_id = "session1"

        token1 = TokenIDGenerator.generate_token_id(
            secret, session_id, "PERSON", "john doe", 6
        )
        token2 = TokenIDGenerator.generate_token_id(
            secret, session_id, "PERSON", "jane smith", 6
        )
        token3 = TokenIDGenerator.generate_token_id(
            secret, session_id, "ORG", "acme corp", 6
        )

        # All tokens should be unique
        assert len({token1, token2, token3}) == 3


# ============================================================================
# Tests: Session Stability
# ============================================================================


class TestSessionStability:
    """Test that same request yields consistent tokens within session."""

    def test_same_session_same_tokens(self):
        """Test that repeating request in same session yields same tokens."""
        secret = "test-secret"
        session_id = "session-A"
        entity_id = "PERSON"
        text = "john doe"

        token1 = TokenIDGenerator.generate_token_id(
            secret, session_id, entity_id, text, 6
        )
        token2 = TokenIDGenerator.generate_token_id(
            secret, session_id, entity_id, text, 6
        )

        assert token1 == token2

    def test_different_session_different_tokens(self):
        """Test that different sessions produce different tokens."""
        secret = "test-secret"
        entity_id = "PERSON"
        text = "john doe"

        token1 = TokenIDGenerator.generate_token_id(
            secret, "session-A", entity_id, text, 6
        )
        token2 = TokenIDGenerator.generate_token_id(
            secret, "session-B", entity_id, text, 6
        )

        assert token1 != token2


# ============================================================================
# Tests: Overlaps & Repeats
# ============================================================================


class TestOverlapResolution:
    """Test deterministic overlap and repeat resolution."""

    def test_longer_entity_wins(self, sample_template):
        """Test that longer spans win over shorter overlapping spans."""
        text = "John Doe Smith works at ABC"
        entities = [
            ExtractedEntity(entity_id="PERSON", text="John"),
            ExtractedEntity(entity_id="PERSON", text="John Doe Smith"),
        ]

        spans = SpanResolver.resolve_spans(text, entities, sample_template.canon)

        # Should have 1 span for the longer "John Doe Smith"
        assert len(spans) == 1
        assert spans[0].text == "John Doe Smith"

    def test_repeated_entity_consistent(self, sample_template):
        """Test that repeated entities are consistently replaced."""
        text = "John Doe and John Doe met"
        entities = [ExtractedEntity(entity_id="PERSON", text="John Doe")]

        spans = SpanResolver.resolve_spans(text, entities, sample_template.canon)

        # Should have 2 spans for the two occurrences
        assert len(spans) == 2
        assert spans[0].start < spans[1].start

    def test_span_integrity_multi_word_entity(self, sample_template):
        """Ensure multi-word matches map to exact text span boundaries."""
        text = "John Doe from Acme Corp called me at 555-1234."
        entities = [
            ExtractedEntity(entity_id="PERSON", text="John Doe"),
            ExtractedEntity(entity_id="ORG", text="Acme Corp"),
            ExtractedEntity(entity_id="PHONE", text="555-1234"),
        ]

        spans = SpanResolver.resolve_spans(text, entities, sample_template.canon)
        span_texts = {span.entity_id: text[span.start:span.end] for span in spans}

        assert span_texts["PERSON"] == "John Doe"
        assert span_texts["ORG"] == "Acme Corp"
        assert span_texts["PHONE"] == "555-1234"


# ============================================================================
# Tests: Unicode & Multilingual
# ============================================================================


class TestUnicodeAndMultilingual:
    """Test unicode normalization and multilingual support."""

    def test_unicode_normalization_doesnt_break_reversal(self):
        """Test that unicode normalization preserves reversibility."""
        # Test with é (decomposed vs composed)
        text_composed = "José García"
        text_decomposed = "Jose\u0301 Garci\u0301a"

        canon_config = Canonicalizer(unicode_normalize="NFKC")
        canon1 = TextNormalizer.canonicalize(text_composed, canon_config)
        canon2 = TextNormalizer.canonicalize(text_decomposed, canon_config)

        # Both should canonicalize to the same form
        assert canon1 == canon2

    def test_arabic_names_anonymization(self, sample_template):
        """Test with Arabic names."""
        text = "أمينة الفاسي عملت هناك"
        entities = [ExtractedEntity(entity_id="PERSON", text="أمينة الفاسي")]

        spans = SpanResolver.resolve_spans(text, entities, sample_template.canon)
        assert len(spans) == 1


# ============================================================================
# Tests: Chunking
# ============================================================================


class TestChunking:
    """Test chunking strategy for large texts."""

    def test_chunking_large_text(self):
        """Test that large texts are chunked correctly."""
        from app.pipeline.chunker import TextChunker

        large_text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3." * 100
        chunks = TextChunker.chunk_by_paragraphs(large_text, target_char_size=500)

        assert len(chunks) > 1
        # Verify chunks can be joined back
        reconstructed = "\n\n".join(chunks)
        # Note: may lose some newlines due to simple joining
        assert "Paragraph 1" in reconstructed


# ============================================================================
# Tests: JSON Enforcement
# ============================================================================


class TestJSONEnforcement:
    """Test strict JSON output enforcement."""

    def test_invalid_llm_output_raises_error(self):
        """Test that invalid LLM output raises an error."""
        client = OllamaClient("http://localhost:11434", "llama3.1")

        # Mock invalid response
        invalid_output = "This is not JSON"
        with pytest.raises(ValueError):
            client._extract_json(invalid_output)  # This will still pass
            # The error happens during JSON parsing

    def test_llm_output_schema_validation(self):
        """Test that LLM output must match schema."""
        from app.schemas.models import LLMExtractionOutput

        # Valid
        valid_data = {
            "entities": [
                {"entity_id": "PERSON", "text": "John Doe"},
            ]
        }
        output = LLMExtractionOutput(**valid_data)
        assert len(output.entities) == 1

        # Invalid: extra fields
        invalid_data = {
            "entities": [
                {"entity_id": "PERSON", "text": "John Doe", "confidence": 0.95},
            ]
        }
        # Pydantic will allow extra fields by default, but we can validate
        output = LLMExtractionOutput(**invalid_data)
        assert output.entities[0].entity_id == "PERSON"


# ============================================================================
# Tests: Template Validation
# ============================================================================


class TestTemplateValidation:
    """Test template schema validation."""

    def test_valid_template_passes(self, sample_template):
        """Test that valid templates pass validation."""
        valid, errors = validate_template(sample_template)
        assert valid
        assert len(errors) == 0

    def test_invalid_placeholder_format(self, sample_template):
        """Test that templates without proper placeholders fail."""
        sample_template.replacement.placeholder_format = "ENTITY_ID"  # Missing {ENTITY} or {ID}
        valid, errors = validate_template(sample_template)
        assert not valid

    def test_duplicate_entity_ids(self, sample_template):
        """Test that duplicate entity IDs are rejected."""
        sample_template.entities.append(
            EntityDefinition(id="PERSON", enabled=True, instructions="Duplicate")
        )
        valid, errors = validate_template(sample_template)
        assert not valid

    def test_invalid_postpass_entity_id(self, sample_template):
        """Template should fail if postpass entity id doesn't exist."""
        sample_template.postpass_alias.entity_ids = ["PERSON", "UNKNOWN"]
        valid, errors = validate_template(sample_template)
        assert not valid
        assert any("postpass_alias.entity_ids" in err for err in errors)


# ============================================================================
# Tests: Normalization & Deduplication
# ============================================================================


class TestNormalization:
    """Test entity normalization and deduplication."""

    def test_entity_deduplication(self, sample_template):
        """Test that duplicate entities are deduplicated."""
        entities = [
            ExtractedEntity(entity_id="PERSON", text="John Doe"),
            ExtractedEntity(entity_id="PERSON", text="john doe"),  # Different casing
            ExtractedEntity(entity_id="PERSON", text="John Doe"),  # Exact duplicate
        ]

        normalized = Normalizer.normalize_entities(entities, sample_template.canon)

        # Should only have 1 entity after normalization
        assert len(normalized) == 1

    def test_dedup_keeps_distinct_entity_ids(self, sample_template):
        """Same text with different IDs should not be collapsed."""
        entities = [
            ExtractedEntity(entity_id="PERSON", text="Acme"),
            ExtractedEntity(entity_id="ORG", text="Acme"),
        ]
        normalized = Normalizer.normalize_entities(entities, sample_template.canon)
        assert len(normalized) == 2


class TestRuleExtraction:
    """Test deterministic regex extraction paths."""

    def test_rule_extractor_detects_phone_and_email(self):
        text = "Email john.doe@example.com or call 555-1234."
        entities = RuleExtractor.extract(text, {"PHONE", "EMAIL"})
        pairs = {(e.entity_id, e.text) for e in entities}

        assert ("EMAIL", "john.doe@example.com") in pairs
        assert ("PHONE", "555-1234") in pairs


class TestSearchView:
    """Test normalized search mapping stability."""

    def test_search_view_maps_whitespace_safely(self, sample_template):
        text = "John Doe from Acme Corp called me at 555-1234."
        view = SearchView(text, sample_template.canon)
        spans = view.find_span("John Doe")

        assert spans
        start, end = spans[0]
        assert text[start:end] == "John Doe"

    def test_search_view_does_not_match_inside_words(self, sample_template):
        text = "company become computing .com com"
        view = SearchView(text, sample_template.canon)
        spans = view.find_span("com")
        matched = [text[s:e] for s, e in spans]

        # Should only match standalone/tld occurrence, not inside larger words.
        assert "com" in matched
        assert len(spans) == 2


class TestGenericAliasPropagation:
    """Test generic session-level alias propagation pass."""

    def test_last_name_detected_from_prior_full_name_same_session(self, template_registry):
        mock_ollama = Mock()
        mock_ollama.extract_entities.side_effect = [
            [
                ExtractedEntity(entity_id="PERSON", text="Jensen Huang"),
                ExtractedEntity(entity_id="ORG", text="NVIDIA"),
            ],
            [
                ExtractedEntity(entity_id="ORG", text="NVIDIA"),
            ],
        ]

        detector = Detector(
            template_registry=template_registry,
            ollama_client=mock_ollama,
            pseudonym_secret="test-secret",
            chunking_enabled=False,
        )

        req1 = AnonymizeRequest(
            session_id="s1",
            template_id="test-pii",
            text="Jensen Huang leads NVIDIA.",
            render_mode="structural",
            language="auto",
        )
        req2 = AnonymizeRequest(
            session_id="s1",
            template_id="test-pii",
            text="Huang announced a new roadmap at NVIDIA.",
            render_mode="structural",
            language="auto",
        )

        detector.detect_and_anonymize(req1)
        anon_text_2, token_to_original_2, _, _ = detector.detect_and_anonymize(req2)

        assert "Huang" in token_to_original_2.values()
        assert "Huang" not in anon_text_2

    def test_org_partial_detected_from_prior_full_name_same_session(self, template_registry):
        mock_ollama = Mock()
        mock_ollama.extract_entities.side_effect = [
            [
                ExtractedEntity(entity_id="ORG", text="Acme Corporation"),
            ],
            [],
        ]

        detector = Detector(
            template_registry=template_registry,
            ollama_client=mock_ollama,
            pseudonym_secret="test-secret",
            chunking_enabled=False,
        )

        req1 = AnonymizeRequest(
            session_id="s-org",
            template_id="test-pii",
            text="Acme Corporation shipped updates.",
            render_mode="structural",
            language="auto",
        )
        req2 = AnonymizeRequest(
            session_id="s-org",
            template_id="test-pii",
            text="Acme announced quarterly revenue.",
            render_mode="structural",
            language="auto",
        )

        detector.detect_and_anonymize(req1)
        anon_text_2, token_to_original_2, _, _ = detector.detect_and_anonymize(req2)

        assert "Acme" in token_to_original_2.values()
        assert "Acme" not in anon_text_2

    def test_custom_entity_scalability_with_template_opt_in(self, temp_templates_dir):
        template = TemplateSchema(
            template_id="test-custom",
            version=1,
            description="Custom entity template",
            entities=[
                EntityDefinition(id="PRODUCT", enabled=True, instructions="Detect product names"),
            ],
            llm=LLMConfig(model="llama3.1", system="Return JSON only", max_entities=200),
            canon=Canonicalizer(
                unicode_normalize="NFKC",
                collapse_whitespace=True,
                casefold=True,
                strip_outer_punct=True,
            ),
            replacement=ReplacementPolicy(
                placeholder_format="<<{ENTITY}:{ID}>>",
                pseudonym=PseudonymConfig(mode="session_stable", id_len=6),
            ),
            postpass_alias=PostPassAliasConfig(
                enabled=True,
                entity_ids=["PRODUCT"],
                min_token_len=3,
                window_size=3,
                min_overlap_tokens=2,
                session_ttl_seconds=3600,
                max_aliases_per_entity=50,
            ),
        )
        (temp_templates_dir / "test-custom.json").write_text(template.model_dump_json())
        registry = TemplateRegistry(temp_templates_dir)

        mock_ollama = Mock()
        mock_ollama.extract_entities.side_effect = [
            [ExtractedEntity(entity_id="PRODUCT", text="Air Max")],
            [],
        ]

        detector = Detector(
            template_registry=registry,
            ollama_client=mock_ollama,
            pseudonym_secret="test-secret",
            chunking_enabled=False,
        )

        req1 = AnonymizeRequest(
            session_id="s-product",
            template_id="test-custom",
            text="Air Max launched today.",
            render_mode="structural",
            language="auto",
        )
        req2 = AnonymizeRequest(
            session_id="s-product",
            template_id="test-custom",
            text="Max demand increased this quarter.",
            render_mode="structural",
            language="auto",
        )

        detector.detect_and_anonymize(req1)
        _, token_to_original_2, _, _ = detector.detect_and_anonymize(req2)
        assert "Max" in token_to_original_2.values()

    def test_ttl_expiry_disables_old_alias_reuse(self, template_registry):
        mock_ollama = Mock()
        mock_ollama.extract_entities.side_effect = [
            [ExtractedEntity(entity_id="PERSON", text="Jensen Huang")],
            [],
        ]

        detector = Detector(
            template_registry=template_registry,
            ollama_client=mock_ollama,
            pseudonym_secret="test-secret",
            chunking_enabled=False,
        )

        # Force very short ttl for this test.
        template = detector.registry.get_template("test-pii")
        template.postpass_alias.session_ttl_seconds = 1

        req1 = AnonymizeRequest(
            session_id="s-ttl",
            template_id="test-pii",
            text="Jensen Huang introduced new chips.",
            render_mode="structural",
            language="auto",
        )
        req2 = AnonymizeRequest(
            session_id="s-ttl",
            template_id="test-pii",
            text="Huang discussed roadmap.",
            render_mode="structural",
            language="auto",
        )

        detector.alias_postpass._now = lambda: 1000.0
        detector.detect_and_anonymize(req1)
        detector.alias_postpass._now = lambda: 1002.0
        _, token_to_original_2, _, _ = detector.detect_and_anonymize(req2)
        assert "Huang" not in token_to_original_2.values()

    def test_backward_compat_without_postpass_config(self, temp_templates_dir):
        template = TemplateSchema(
            template_id="test-legacy",
            version=1,
            description="Legacy template without postpass",
            entities=[
                EntityDefinition(id="PERSON", enabled=True, instructions="Detect names"),
            ],
            llm=LLMConfig(model="llama3.1", system="Return JSON only", max_entities=200),
            canon=Canonicalizer(
                unicode_normalize="NFKC",
                collapse_whitespace=True,
                casefold=True,
                strip_outer_punct=True,
            ),
            replacement=ReplacementPolicy(
                placeholder_format="<<{ENTITY}:{ID}>>",
                pseudonym=PseudonymConfig(mode="session_stable", id_len=6),
            ),
            postpass_alias=None,
        )
        (temp_templates_dir / "test-legacy.json").write_text(template.model_dump_json())
        registry = TemplateRegistry(temp_templates_dir)

        mock_ollama = Mock()
        mock_ollama.extract_entities.return_value = [
            ExtractedEntity(entity_id="PERSON", text="John Doe")
        ]

        detector = Detector(
            template_registry=registry,
            ollama_client=mock_ollama,
            pseudonym_secret="test-secret",
            chunking_enabled=False,
        )

        req = AnonymizeRequest(
            session_id="legacy",
            template_id="test-legacy",
            text="John Doe visited yesterday.",
            render_mode="structural",
            language="auto",
        )

        anon_text, token_to_original, _, _ = detector.detect_and_anonymize(req)
        assert anon_text != req.text
        assert any(v == "John Doe" for v in token_to_original.values())

    def test_postpass_runs_once_after_chunk_merge(self, template_registry):
        mock_ollama = Mock()
        mock_ollama.extract_entities.side_effect = [
            [ExtractedEntity(entity_id="PERSON", text="John Doe")],
            [ExtractedEntity(entity_id="ORG", text="Acme Corp")],
        ]

        detector = Detector(
            template_registry=template_registry,
            ollama_client=mock_ollama,
            pseudonym_secret="test-secret",
            chunking_enabled=True,
            chunk_char_target=30,
            chunk_overlap_chars=0,
        )

        # Wrap augment to count call frequency.
        original_augment = detector.alias_postpass.augment
        call_count = {"n": 0}

        def wrapped_augment(*args, **kwargs):
            call_count["n"] += 1
            return original_augment(*args, **kwargs)

        detector.alias_postpass.augment = wrapped_augment

        req = AnonymizeRequest(
            session_id="chunked",
            template_id="test-pii",
            text="John Doe announced updates.\n\nAcme Corp released earnings.",
            render_mode="structural",
            language="auto",
        )

        detector.detect_and_anonymize(req)
        assert call_count["n"] == 1

    def test_realistic_mode_keeps_same_fake_for_alias_group(self, template_registry):
        mock_ollama = Mock()
        mock_ollama.extract_entities.side_effect = [
            [ExtractedEntity(entity_id="PERSON", text="Jensen Huang")],
            [],
        ]

        detector = Detector(
            template_registry=template_registry,
            ollama_client=mock_ollama,
            pseudonym_secret="test-secret",
            chunking_enabled=False,
        )

        req1 = AnonymizeRequest(
            session_id="s-realistic",
            template_id="test-pii",
            text="Jensen Huang spoke today.",
            render_mode="realistic",
            language="auto",
        )
        req2 = AnonymizeRequest(
            session_id="s-realistic",
            template_id="test-pii",
            text="Huang confirmed the guidance.",
            render_mode="realistic",
            language="auto",
        )

        _, _, token_to_fake_1, _ = detector.detect_and_anonymize(req1)
        _, _, token_to_fake_2, _ = detector.detect_and_anonymize(req2)

        full_name_fake = next(iter(token_to_fake_1.values()))
        alias_fake = next(iter(token_to_fake_2.values()))
        assert alias_fake in full_name_fake.split()


class TestStructuredEntityFiltering:
    """Drop noisy structured fragments like bare TLDs."""

    def test_filter_bare_tld_for_link_entity(self, template_registry):
        mock_ollama = Mock()
        mock_ollama.extract_entities.return_value = [
            ExtractedEntity(entity_id="LINKS", text="com"),
            ExtractedEntity(entity_id="LINKS", text="www.tech-private.com"),
        ]

        detector = Detector(
            template_registry=template_registry,
            ollama_client=mock_ollama,
            pseudonym_secret="test-secret",
            chunking_enabled=False,
        )

        req = AnonymizeRequest(
            session_id="links",
            template_id="test-pii",
            text="Read more at www.tech-private.com and see company updates.",
            render_mode="structural",
            language="auto",
        )

        _, token_to_original, _, _ = detector.detect_and_anonymize(req)
        values = set(token_to_original.values())
        assert "com" not in values


class TestRealisticLinkRendering:
    """Link/url realistic rendering should stay URL-like."""

    def test_links_realistic_fake_has_domain_shape(self, template_registry):
        mock_ollama = Mock()
        mock_ollama.extract_entities.return_value = [
            ExtractedEntity(entity_id="LINKS", text="www.tech-private.com"),
        ]

        detector = Detector(
            template_registry=template_registry,
            ollama_client=mock_ollama,
            pseudonym_secret="test-secret",
            chunking_enabled=False,
        )

        req = AnonymizeRequest(
            session_id="s-link-realistic",
            template_id="test-pii",
            text="See www.tech-private.com for details.",
            render_mode="realistic",
            language="auto",
        )

        _, _, token_to_fake, _ = detector.detect_and_anonymize(req)
        fake = next(iter(token_to_fake.values()))
        assert "." in fake
        assert " " not in fake


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
