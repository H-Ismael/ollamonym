"""Pydantic models for API contracts."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Request/Response Models (API)
# ============================================================================


class AnonymizeRequest(BaseModel):
    """Request for anonymization endpoint."""

    session_id: str = Field(..., description="Session identifier for stability")
    template_id: str = Field(..., description="Template ID to use")
    text: str = Field(..., description="Text to anonymize")
    render_mode: str = Field(
        "structural",
        description="'structural' (placeholders only) or 'realistic' (with fake names)"
    )
    language: str = Field("auto", description="Language hint: 'auto', 'en', 'fr', 'ar'")


class DeanonymizeRequest(BaseModel):
    """Request for deanonymization endpoint."""

    text: str = Field(..., description="Text to deanonymize (may contain tokens or fakes)")
    mapping: Dict[str, Any] = Field(
        ...,
        description="Mapping dict with token_to_original, token_to_fake (optional), fake_to_token (optional)"
    )


class ModelRuntimeInfo(BaseModel):
    """Runtime model details resolved from Ollama when available."""

    requested_model: str
    resolved_model: Optional[str] = None
    quantization_level: Optional[str] = None


class MappingMetadata(BaseModel):
    """Metadata about the mapping."""

    session_id: str
    template_id: str
    template_version: int
    render_mode: str
    model_runtime: Optional[ModelRuntimeInfo] = None


class AnonymizationMapping(BaseModel):
    """Complete mapping returned by anonymize endpoint."""

    token_to_original: Dict[str, str] = Field(
        ..., description="Canonical reverse mapping: token -> original entity text"
    )
    token_to_fake: Optional[Dict[str, str]] = Field(
        None, description="Optional realistic rendering: token -> fake name/org"
    )
    fake_to_token: Optional[Dict[str, str]] = Field(
        None, description="Optional reverse of token_to_fake for deanonymize with realistic input"
    )
    meta: MappingMetadata


class AnonymizeResponse(BaseModel):
    """Response from anonymization endpoint."""

    anonymized_text: str
    mapping: AnonymizationMapping


class DeanonymizeResponse(BaseModel):
    """Response from deanonymization endpoint."""

    text: str


# ============================================================================
# Template Schema Models
# ============================================================================


class EntityDefinition(BaseModel):
    """Definition of a single entity type in a template."""

    id: str = Field(..., description="Entity ID (e.g., PERSON, ORG)")
    enabled: bool = Field(True, description="Whether this entity is active")
    instructions: str = Field(..., description="Instructions for LLM to detect this entity")
    examples: Optional[Dict[str, List[str]]] = Field(
        None, description="Optional positive/negative examples"
    )


class Canonicalizer(BaseModel):
    """Canonicalization rules."""

    unicode_normalize: str = Field("NFKC", description="Unicode normalization form")
    collapse_whitespace: bool = Field(True, description="Collapse multiple spaces")
    casefold: bool = Field(True, description="Convert to lowercase")
    strip_outer_punct: bool = Field(True, description="Strip outer punctuation")


class PseudonymConfig(BaseModel):
    """Configuration for realistic pseudonym generation."""

    mode: str = Field("session_stable", description="Pseudonym generation mode")
    id_len: int = Field(6, description="Length of token ID")
    secret_env: str = Field("PSEUDONYM_SECRET", description="Env var name for secret")
    providers: Optional[Dict[str, str]] = Field(
        None, description="e.g., {'PERSON': 'faker.person', 'ORG': 'faker.company'}"
    )


class ReplacementPolicy(BaseModel):
    """Policy for replacing entities with placeholders and fakes."""

    placeholder_format: str = Field(
        "<<{ENTITY}:{ID}>>", description="Format string for placeholders"
    )
    pseudonym: Optional[PseudonymConfig] = Field(None, description="Pseudonym generation config")


class LLMConfig(BaseModel):
    """LLM configuration in template."""

    model: str = Field("llama3.1:8b-instruct-q4_K_M", description="Ollama model name")
    system: str = Field("Return JSON only. No extra keys.", description="System message")
    max_entities: int = Field(200, description="Max entities per chunk")


class PostPassAliasConfig(BaseModel):
    """Template policy for session alias propagation post-pass."""

    enabled: bool = Field(False, description="Enable alias post-pass")
    entity_ids: List[str] = Field(default_factory=list, description="Entity IDs to include")
    min_token_len: int = Field(3, description="Minimum token length to keep")
    window_size: int = Field(3, description="Max moving window size")
    min_overlap_tokens: int = Field(2, description="Minimum overlap with known tokens")
    session_ttl_seconds: int = Field(3600, description="In-memory session TTL in seconds")
    max_aliases_per_entity: int = Field(50, description="Max aliases/signatures retained per entity")


class TemplateSchema(BaseModel):
    """Complete template specification."""

    template_id: str
    version: int
    description: str
    entities: List[EntityDefinition]
    llm: LLMConfig
    canon: Canonicalizer
    replacement: ReplacementPolicy
    postpass_alias: Optional[PostPassAliasConfig] = None


# ============================================================================
# LLM Output Schema
# ============================================================================


class ExtractedEntity(BaseModel):
    """Single extracted entity from LLM."""

    entity_id: str = Field(..., description="Must match an enabled template entity ID")
    text: str = Field(..., description="Exact surface form from original text")


class LLMExtractionOutput(BaseModel):
    """Strict LLM output schema."""

    entities: List[ExtractedEntity] = Field(
        ..., description="List of extracted entities. No offsets, confidence, or extra fields."
    )


# ============================================================================
# Template Management Responses
# ============================================================================


class TemplateInfo(BaseModel):
    """Brief info about a template."""

    template_id: str
    version: int
    description: str


class TemplatesListResponse(BaseModel):
    """Response from GET /v2/templates."""

    templates: List[TemplateInfo]


class TemplateValidationResult(BaseModel):
    """Result of template validation."""

    valid: bool
    errors: List[str] = Field(default_factory=list, description="Validation errors if any")


# ============================================================================
# Internal helper models
# ============================================================================


class EntityOccurrence(BaseModel):
    """A span in the text where an entity occurs."""

    entity_id: str
    text: str
    start: int
    end: int
    token: str = Field(..., description="Placeholder token assigned")


class ChunkResult(BaseModel):
    """Result from processing a single chunk through LLM."""

    chunk_index: int
    chunk_text: str
    entities: List[ExtractedEntity]
    error: Optional[str] = None
