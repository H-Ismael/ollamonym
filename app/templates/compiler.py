"""Prompt compilation from template."""

import logging
from app.schemas.models import TemplateSchema

logger = logging.getLogger(__name__)


class PromptCompiler:
    """Compiles prompts from templates for LLM extraction."""

    def __init__(self):
        # Cache of compiled prompts: key = (template_id, version)
        self.prompt_cache = {}

    def compile_system_message(self, template: TemplateSchema) -> str:
        """Compile system message from template."""
        return template.llm.system

    def compile_entity_definitions(self, template: TemplateSchema) -> str:
        """Compile entity definitions block from template."""
        lines = ["Entity Definitions:"]

        for entity in template.entities:
            if not entity.enabled:
                continue

            lines.append(f"\n{entity.id}: {entity.instructions}")

            # Add examples if present
            if entity.examples:
                if entity.examples.get("positive"):
                    examples_str = ", ".join(entity.examples["positive"][:3])
                    lines.append(f"  Positive examples: {examples_str}")
                if entity.examples.get("negative"):
                    examples_str = ", ".join(entity.examples["negative"][:3])
                    lines.append(f"  Negative examples: {examples_str}")

        return "\n".join(lines)

    def compile_output_constraints(self, template: TemplateSchema) -> str:
        """Compile output schema constraints."""
        enabled_ids = [e.id for e in template.entities if e.enabled]
        ids_str = ", ".join(enabled_ids)

        return (
            f"""Output Format (JSON only):
{{
  "entities": [
    {{"entity_id": "<ID>", "text": "<EXACT_SURFACE_FORM>"}},
    ...
  ]
}}

Constraints:
- Only include entity IDs from this list: {ids_str}
- No offsets, confidence scores, rationale, or other fields.
- Do not invent entities not present in the text.
- Ensure "text" field is an exact surface form from the input."""
        )

    def compile_user_message(self, template: TemplateSchema, text: str, chunk_index: int = 0, total_chunks: int = 1) -> str:
        """Compile user message for extraction."""
        lines = []

        # Task header
        lines.append("Extract sensitive entities from TEXT using ENTITY DEFINITIONS:")
        lines.append("")

        # Entity definitions
        lines.append(self.compile_entity_definitions(template))
        lines.append("")

        # Output constraints
        lines.append(self.compile_output_constraints(template))
        lines.append("")

        # Text block
        if total_chunks > 1:
            lines.append(f"TEXT (Chunk {chunk_index + 1}/{total_chunks}):")
        else:
            lines.append("TEXT:")
        lines.append(text)

        return "\n".join(lines)

    def get_cached_system_message(self, template: TemplateSchema) -> str:
        """Get or compile system message (cached by template_id@version)."""
        key = (template.template_id, template.version, "system")
        if key not in self.prompt_cache:
            self.prompt_cache[key] = self.compile_system_message(template)
        return self.prompt_cache[key]

    def get_cached_entity_definitions(self, template: TemplateSchema) -> str:
        """Get or compile entity definitions block (cached by template_id@version)."""
        key = (template.template_id, template.version, "entities")
        if key not in self.prompt_cache:
            self.prompt_cache[key] = self.compile_entity_definitions(template)
        return self.prompt_cache[key]
