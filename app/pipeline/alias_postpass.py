"""Template-driven session alias post-pass for generic entity propagation."""

import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from app.schemas.models import Canonicalizer, ExtractedEntity, PostPassAliasConfig
from app.utils.text_norm import TextNormalizer


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


@dataclass
class EntityAliasState:
    """In-memory alias memory per entity class."""

    single_tokens: Set[str] = field(default_factory=set)
    ngram_signatures: Set[Tuple[str, ...]] = field(default_factory=set)
    token_vocab: Set[str] = field(default_factory=set)
    anchor_phrases: Set[str] = field(default_factory=set)
    token_to_anchors: Dict[str, Set[str]] = field(default_factory=dict)


@dataclass
class SessionAliasState:
    """In-memory alias memory per session."""

    last_seen: float
    entities: Dict[str, EntityAliasState] = field(default_factory=dict)


class AliasPostPass:
    """Applies template-driven alias propagation after extraction completes."""

    def __init__(self, now_fn: Callable[[], float] | None = None):
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionAliasState] = {}
        self._now = now_fn or time.time

    def augment(
        self,
        session_id: str,
        text: str,
        entities: List[ExtractedEntity],
        canon_config: Canonicalizer,
        policy: PostPassAliasConfig,
    ) -> List[ExtractedEntity]:
        """Return entities augmented with session-stable alias-derived candidates."""
        if not policy.enabled or not policy.entity_ids:
            return entities

        now = self._now()
        self._evict_expired(now, policy.session_ttl_seconds)

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionAliasState(last_seen=now)
                self._sessions[session_id] = session
            session.last_seen = now

            self._learn_from_entities(session, entities, canon_config, policy)
            generated = self._generate_candidates(session, text, canon_config, policy)

        if not generated:
            return entities
        return [*entities, *generated]

    def resolve_group_key(
        self,
        session_id: str,
        entity_id: str,
        text: str,
        canon_config: Canonicalizer,
    ) -> str:
        """
        Resolve a stable identity key for fake generation.
        Prefers a unique multi-token anchor phrase when available.
        """
        canonical_phrase = TextNormalizer.canonicalize(text, canon_config).strip()
        tokens = self._tokenize(text, canon_config, min_token_len=1)
        if not canonical_phrase or not tokens:
            return canonical_phrase

        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return canonical_phrase
            state = session.entities.get(entity_id)
            if not state:
                return canonical_phrase

            if canonical_phrase in state.anchor_phrases:
                return canonical_phrase

            if len(tokens) == 1:
                anchors = state.token_to_anchors.get(tokens[0], set())
                if len(anchors) == 1:
                    return next(iter(anchors))
                return canonical_phrase

            candidate_anchors: Optional[Set[str]] = None
            for token in set(tokens):
                anchors = state.token_to_anchors.get(token, set())
                if not anchors:
                    continue
                if candidate_anchors is None:
                    candidate_anchors = set(anchors)
                else:
                    candidate_anchors &= anchors

            if candidate_anchors and len(candidate_anchors) == 1:
                return next(iter(candidate_anchors))

        return canonical_phrase

    def _evict_expired(self, now: float, ttl_seconds: int) -> None:
        with self._lock:
            stale_ids = [
                sid
                for sid, state in self._sessions.items()
                if now - state.last_seen > ttl_seconds
            ]
            for sid in stale_ids:
                self._sessions.pop(sid, None)

    def _learn_from_entities(
        self,
        session: SessionAliasState,
        entities: List[ExtractedEntity],
        canon_config: Canonicalizer,
        policy: PostPassAliasConfig,
    ) -> None:
        for entity in entities:
            if entity.entity_id not in policy.entity_ids:
                continue

            tokens = self._tokenize(entity.text, canon_config, policy.min_token_len)
            if not tokens:
                continue

            entity_state = session.entities.setdefault(entity.entity_id, EntityAliasState())
            entity_state.token_vocab.update(tokens)

            for token in tokens:
                entity_state.single_tokens.add(token)

            max_n = min(policy.window_size, len(tokens))
            for n in range(2, max_n + 1):
                for idx in range(0, len(tokens) - n + 1):
                    ngram = tuple(tokens[idx:idx + n])
                    entity_state.ngram_signatures.add(ngram)

            if len(tokens) >= 2:
                anchor = TextNormalizer.canonicalize(entity.text, canon_config).strip()
                if anchor:
                    entity_state.anchor_phrases.add(anchor)
                    for token in set(tokens):
                        if token not in entity_state.token_to_anchors:
                            entity_state.token_to_anchors[token] = set()
                        entity_state.token_to_anchors[token].add(anchor)

            self._trim_entity_state(entity_state, policy.max_aliases_per_entity)

    def _trim_entity_state(self, state: EntityAliasState, cap: int) -> None:
        if len(state.single_tokens) > cap:
            state.single_tokens = set(sorted(state.single_tokens)[:cap])
        if len(state.ngram_signatures) > cap:
            state.ngram_signatures = set(sorted(state.ngram_signatures)[:cap])
        if len(state.token_vocab) > cap:
            state.token_vocab = set(sorted(state.token_vocab)[:cap])
        if len(state.anchor_phrases) > cap:
            state.anchor_phrases = set(sorted(state.anchor_phrases)[:cap])
        if len(state.token_to_anchors) > cap:
            kept_tokens = sorted(state.token_to_anchors.keys())[:cap]
            state.token_to_anchors = {
                tok: state.token_to_anchors[tok] for tok in kept_tokens
            }

    def _generate_candidates(
        self,
        session: SessionAliasState,
        text: str,
        canon_config: Canonicalizer,
        policy: PostPassAliasConfig,
    ) -> List[ExtractedEntity]:
        text_tokens = self._tokenize_with_originals(text, canon_config, policy.min_token_len)
        if not text_tokens:
            return []

        by_entity: Dict[str, Set[str]] = defaultdict(set)
        max_n = min(policy.window_size, len(text_tokens))

        for entity_id in sorted(policy.entity_ids):
            state = session.entities.get(entity_id)
            if not state:
                continue

            for canon_tok, orig_tok in text_tokens:
                if canon_tok in state.single_tokens:
                    by_entity[entity_id].add(orig_tok)

            for n in range(2, max_n + 1):
                for idx in range(0, len(text_tokens) - n + 1):
                    window = text_tokens[idx:idx + n]
                    canon_window = [tok[0] for tok in window]
                    window_set = set(canon_window)

                    overlap = len(window_set & state.token_vocab)
                    if overlap < policy.min_overlap_tokens:
                        continue

                    if not self._matches_known_ngrams(
                        canon_window,
                        state.ngram_signatures,
                        policy.min_overlap_tokens,
                    ):
                        continue

                    phrase = " ".join(tok[1] for tok in window)
                    by_entity[entity_id].add(phrase)

        generated: List[ExtractedEntity] = []
        for entity_id in sorted(by_entity.keys()):
            for text_value in sorted(by_entity[entity_id]):
                generated.append(ExtractedEntity(entity_id=entity_id, text=text_value))
        return generated

    @staticmethod
    def _matches_known_ngrams(
        window_tokens: List[str],
        known_ngrams: Set[Tuple[str, ...]],
        min_overlap_tokens: int,
    ) -> bool:
        window_set = set(window_tokens)
        for signature in known_ngrams:
            if len(window_set & set(signature)) >= min_overlap_tokens:
                return True
        return False

    @staticmethod
    def _tokenize(text: str, canon_config: Canonicalizer, min_token_len: int) -> List[str]:
        tokens: List[str] = []
        for match in TOKEN_RE.finditer(text):
            raw = match.group(0)
            canon = TextNormalizer.canonicalize(raw, canon_config).strip()
            if len(canon) < min_token_len:
                continue
            tokens.append(canon)
        return tokens

    @staticmethod
    def _tokenize_with_originals(
        text: str,
        canon_config: Canonicalizer,
        min_token_len: int,
    ) -> List[Tuple[str, str]]:
        tokens: List[Tuple[str, str]] = []
        for match in TOKEN_RE.finditer(text):
            raw = match.group(0)
            canon = TextNormalizer.canonicalize(raw, canon_config).strip()
            if len(canon) < min_token_len:
                continue
            tokens.append((canon, raw))
        return tokens
