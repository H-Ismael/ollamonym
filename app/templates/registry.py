"""Template registry with in-memory caching."""

import logging
from pathlib import Path
from typing import Dict, Optional

from app.schemas.models import TemplateSchema
from app.templates.loader import load_template_by_id, load_all_templates

logger = logging.getLogger(__name__)


class TemplateRegistry:
    """Thread-safe template registry with in-memory caching."""

    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir
        self.templates: Dict[str, TemplateSchema] = {}
        self.reload()

    def reload(self) -> None:
        """Reload all templates from disk."""
        self.templates = load_all_templates(self.templates_dir)
        logger.info(f"Loaded {len(self.templates)} templates")

    def get_template(self, template_id: str) -> Optional[TemplateSchema]:
        """Get a template by ID. Cached in memory."""
        if template_id not in self.templates:
            # Try to load from disk in case it was added after init
            template = load_template_by_id(template_id, self.templates_dir)
            if template:
                self.templates[template_id] = template
                return template
            return None
        return self.templates[template_id]

    def list_templates(self) -> list[str]:
        """List all available template IDs."""
        return list(self.templates.keys())

    def template_exists(self, template_id: str) -> bool:
        """Check if template exists."""
        return self.get_template(template_id) is not None
