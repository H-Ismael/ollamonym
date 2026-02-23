"""Template loading from disk."""

import json
from pathlib import Path
from typing import Dict, Optional

from app.schemas.models import TemplateSchema


def load_template_by_id(template_id: str, templates_dir: Path) -> Optional[TemplateSchema]:
    """Load a template JSON file by ID and parse into TemplateSchema."""
    template_file = templates_dir / f"{template_id}.json"

    if not template_file.exists():
        return None

    try:
        with open(template_file, "r") as f:
            data = json.load(f)
        return TemplateSchema(**data)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to load template {template_id}: {e}")


def list_template_ids(templates_dir: Path) -> list[str]:
    """List all available template IDs (from .json files in templates_dir)."""
    if not templates_dir.exists():
        return []
    return [f.stem for f in templates_dir.glob("*.json")]


def load_all_templates(templates_dir: Path) -> Dict[str, TemplateSchema]:
    """Load all templates from templates_dir into memory."""
    templates = {}
    for template_id in list_template_ids(templates_dir):
        try:
            template = load_template_by_id(template_id, templates_dir)
            if template:
                templates[template_id] = template
        except Exception:
            # Log but skip invalid templates
            pass
    return templates
