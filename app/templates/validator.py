"""Template validation and sanity checks."""

import logging
from typing import List

from app.schemas.models import TemplateSchema

logger = logging.getLogger(__name__)


def validate_template(template: TemplateSchema) -> tuple[bool, List[str]]:
    """
    Validate a template against schema + sanity rules.
    Returns (is_valid, error_list).
    """
    errors = []

    # Check template_id and version
    if not template.template_id or not isinstance(template.template_id, str):
        errors.append("template_id must be a non-empty string")

    if template.version < 1:
        errors.append("version must be >= 1")

    # Check entities
    if not template.entities:
        errors.append("entities list must not be empty")

    entity_ids = set()
    for entity in template.entities:
        if not entity.id or not isinstance(entity.id, str):
            errors.append(f"Entity ID must be non-empty string, got: {entity.id}")
        if entity.id in entity_ids:
            errors.append(f"Duplicate entity ID: {entity.id}")
        entity_ids.add(entity.id)

    # Check LLM config
    if not template.llm.model:
        errors.append("llm.model must be non-empty")
    if template.llm.max_entities < 1:
        errors.append("llm.max_entities must be >= 1")

    # Check canonicalization
    valid_norms = {"NFC", "NFD", "NFKC", "NFKD"}
    if template.canon.unicode_normalize not in valid_norms:
        errors.append(
            f"canon.unicode_normalize must be one of {valid_norms}, got: {template.canon.unicode_normalize}"
        )

    # Check replacement policy
    if "{ENTITY}" not in template.replacement.placeholder_format:
        errors.append("placeholder_format must contain {ENTITY}")
    if "{ID}" not in template.replacement.placeholder_format:
        errors.append("placeholder_format must contain {ID}")

    # Check postpass alias policy (optional)
    if template.postpass_alias:
        cfg = template.postpass_alias
        unknown_ids = [eid for eid in cfg.entity_ids if eid not in entity_ids]
        if unknown_ids:
            errors.append(
                f"postpass_alias.entity_ids contains unknown IDs: {', '.join(sorted(unknown_ids))}"
            )
        if cfg.min_token_len < 1:
            errors.append("postpass_alias.min_token_len must be >= 1")
        if cfg.window_size < 1:
            errors.append("postpass_alias.window_size must be >= 1")
        if cfg.min_overlap_tokens < 1:
            errors.append("postpass_alias.min_overlap_tokens must be >= 1")
        if cfg.session_ttl_seconds < 1:
            errors.append("postpass_alias.session_ttl_seconds must be >= 1")
        if cfg.max_aliases_per_entity < 1:
            errors.append("postpass_alias.max_aliases_per_entity must be >= 1")
        if cfg.min_overlap_tokens > cfg.window_size:
            errors.append("postpass_alias.min_overlap_tokens must be <= postpass_alias.window_size")

    return len(errors) == 0, errors
