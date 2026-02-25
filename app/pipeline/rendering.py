"""Realistic rendering with deterministic fake generation and uniqueness."""

import hashlib
import logging
import random
import re
from urllib.parse import urlsplit, urlunsplit
from typing import Dict, Optional, Set
from faker import Faker

logger = logging.getLogger(__name__)


class RealisticRenderer:
    """Generates deterministic realistic fakes while ensuring uniqueness."""

    # Map entity IDs to Faker providers
    PROVIDER_MAP = {
        "PERSON": "name",
        "ORG": "company",
        "EMAIL": "email",
        "PHONE": "phone_number",
    }

    def __init__(
        self,
        secret: str,
        session_id: str,
        seed_base: int = 0,
        provider_map: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize renderer with deterministic seed.

        Args:
            secret: Secret for deterministic seeding
            session_id: Session ID for stability
            seed_base: Base seed value
        """
        self.secret = secret
        self.session_id = session_id
        self.seed_base = seed_base
        self.used_fakes: Dict[str, Set[str]] = {}  # entity_id -> set of used fakes
        self.group_fake_cache: Dict[tuple[str, str], str] = {}
        self.view_fake_cache: Dict[tuple[str, str, str], str] = {}
        self.provider_map = dict(self.PROVIDER_MAP)
        if provider_map:
            self.provider_map.update(provider_map)

    def _compute_seed(self, entity_id: str, token_id: str, attempt: int = 0) -> int:
        """Compute deterministic seed from inputs."""
        input_str = f"{self.session_id}|{entity_id}|{token_id}|{attempt}"
        digest = hashlib.sha256(input_str.encode("utf-8")).digest()
        hash_val = int.from_bytes(digest[:8], "big")
        return (self.seed_base + hash_val) % (2**31)

    def generate_fake(
        self,
        entity_id: str,
        token_id: str,
        original: str | None = None,
        group_key: str | None = None,
    ) -> str:
        """
        Generate a deterministic fake value for the entity.

        Args:
            entity_id: Entity type (e.g., PERSON, ORG)
            token_id: Token ID for uniqueness

        Returns:
            Fake value (e.g., "Mark Lewis")
        """
        if entity_id not in self.used_fakes:
            self.used_fakes[entity_id] = set()

        effective_group_key = group_key or token_id
        mention_key = (original or token_id).strip().casefold()
        view_cache_key = (entity_id, effective_group_key, mention_key)
        if view_cache_key in self.view_fake_cache:
            return self.view_fake_cache[view_cache_key]

        provider = self.provider_map.get(entity_id, "word")
        faker = Faker()
        group_cache_key = (entity_id, effective_group_key)
        base_fake = self.group_fake_cache.get(group_cache_key)

        attempt = 0
        while attempt < 100:  # Max retries for uniqueness
            if base_fake is None:
                seed = self._compute_seed(entity_id, effective_group_key, attempt)
                faker.seed_instance(seed)
                if self._is_link_like_entity(entity_id):
                    base_fake = self._generate_link_like_fake(original or "", seed)
                elif hasattr(faker, provider):
                    base_fake = getattr(faker, provider)()
                else:
                    base_fake = faker.word()

            fake = self._project_group_fake(entity_id, original or "", effective_group_key, base_fake)

            # Check uniqueness
            if fake not in self.used_fakes[entity_id]:
                self.used_fakes[entity_id].add(fake)
                self.group_fake_cache[group_cache_key] = base_fake
                self.view_fake_cache[view_cache_key] = fake
                return fake

            if fake == base_fake:
                attempt += 1
                base_fake = None
                continue
            attempt += 1

        # Fallback: use token_id as fake
        fallback = f"{entity_id}_{token_id}"
        logger.warning(f"Failed to generate unique fake after retries; using {fallback}")
        self.used_fakes[entity_id].add(fallback)
        self.group_fake_cache[group_cache_key] = fallback
        self.view_fake_cache[view_cache_key] = fallback
        return fallback

    def _project_group_fake(
        self,
        entity_id: str,
        original: str,
        group_key: str,
        base_fake: str,
    ) -> str:
        """
        Project base fake to alias-length where possible so related mentions stay
        consistent while remaining deanonymizable (distinct fake strings).
        """
        if self._is_link_like_entity(entity_id):
            return base_fake

        orig_tokens = [t for t in re.split(r"\s+", original.strip()) if t]
        group_tokens = [t for t in re.split(r"\s+", group_key.strip()) if t]
        fake_tokens = [t for t in re.split(r"\s+", base_fake.strip()) if t]
        if not orig_tokens or not group_tokens or not fake_tokens:
            return base_fake

        # Preserve single-token surface shape even for standalone mentions
        # (e.g., "Madonna" should not become a two-token fake name).
        if len(orig_tokens) == 1 and len(group_tokens) == 1:
            return fake_tokens[0]

        if len(orig_tokens) >= len(group_tokens):
            return base_fake

        # Single-token alias maps to same relative token when unambiguous.
        if len(orig_tokens) == 1:
            target = orig_tokens[0].casefold()
            indexes = [idx for idx, tok in enumerate(group_tokens) if tok.casefold() == target]
            if len(indexes) == 1:
                idx = min(indexes[0], len(fake_tokens) - 1)
                return fake_tokens[idx]
            return fake_tokens[-1]

        # Multi-token alias: project contiguous tail segment.
        width = min(len(orig_tokens), len(fake_tokens))
        return " ".join(fake_tokens[-width:])

    @staticmethod
    def _is_link_like_entity(entity_id: str) -> bool:
        eid = entity_id.upper()
        return "LINK" in eid or "URL" in eid

    @staticmethod
    def _looks_like_domain_fragment(text: str) -> bool:
        return bool(re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", text))

    def _generate_link_like_fake(self, original: str, seed: int) -> str:
        """Generate deterministic realistic URL/domain-like fakes."""
        rng = random.Random(seed)
        prefixes = ["portal", "secure", "global", "north", "prime", "atlas", "blue", "rapid"]
        nouns = ["insight", "network", "systems", "labs", "group", "connect", "hub", "data"]

        raw = (original or "").strip()
        has_scheme = "://" in raw
        domain_only = (not has_scheme) and ("/" not in raw) and self._looks_like_domain_fragment(raw)
        parsed = urlsplit(raw if has_scheme else f"https://{raw}")

        host = parsed.netloc or ""
        if not host:
            return f"www.{rng.choice(prefixes)}-{rng.choice(nouns)}.com"

        host_no_port = host.split(":", 1)[0]
        parts = [p for p in host_no_port.split(".") if p]
        tld = parts[-1] if parts and parts[-1].isalpha() and 2 <= len(parts[-1]) <= 10 else "com"
        use_www = host_no_port.startswith("www.")
        new_label = f"{rng.choice(prefixes)}-{rng.choice(nouns)}"
        new_host = f"{'www.' if use_www else ''}{new_label}.{tld.lower()}"

        rebuilt = urlunsplit(
            (
                parsed.scheme if has_scheme else "https",
                new_host,
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )
        if domain_only:
            return new_host
        if not has_scheme:
            return rebuilt.replace("https://", "", 1)
        return rebuilt

    def reset_session(self) -> None:
        """Reset session state for a new session."""
        self.used_fakes.clear()
        self.group_fake_cache.clear()
        self.view_fake_cache.clear()
