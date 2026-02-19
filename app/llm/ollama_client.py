"""Ollama client with schema-enforced JSON output."""

import json
import logging
from typing import List, Optional

import httpx

from app.schemas.models import ExtractedEntity, LLMExtractionOutput, ModelRuntimeInfo

logger = logging.getLogger(__name__)


class OllamaClient:
    """Wrapper around Ollama API with schema enforcement."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        fallback_urls: Optional[List[str]] = None,
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama base URL (e.g., "http://localhost:11434")
            model: Model name (e.g., "llama3.1")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.fallback_urls = [url.rstrip("/") for url in (fallback_urls or [])]
        self.model = model
        self._effective_model: Optional[str] = None
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def _candidate_base_urls(self) -> List[str]:
        """Return base URLs in priority order, de-duplicated."""
        seen = set()
        candidates: List[str] = []
        for url in [self.base_url, *self.fallback_urls]:
            if url and url not in seen:
                candidates.append(url)
                seen.add(url)
        return candidates

    def _request_json(
        self,
        method: str,
        path: str,
        json_payload: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """Request JSON with base URL fallback on network errors."""
        last_error: Optional[Exception] = None
        for base_url in self._candidate_base_urls():
            url = f"{base_url}{path}"
            try:
                response = self.client.request(
                    method,
                    url,
                    json=json_payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                if base_url != self.base_url:
                    logger.warning(
                        "Ollama reachable via fallback URL %s (primary was %s)",
                        base_url,
                        self.base_url,
                    )
                    self.base_url = base_url
                return response.json()
            except httpx.RequestError as e:
                last_error = e
                logger.warning("Ollama request failed at %s: %s", base_url, e)
                continue

        if last_error:
            logger.error("Ollama request failed on all candidate URLs: %s", last_error)
            raise last_error
        raise RuntimeError("No Ollama URLs configured")

    def _model_names(self) -> List[str]:
        """Return available model names from Ollama tags."""
        payload = self._request_json("GET", "/api/tags", timeout=5.0)
        models = payload.get("models", [])
        if not isinstance(models, list):
            return []
        names = []
        for model in models:
            name = model.get("name")
            if isinstance(name, str):
                names.append(name)
        return names

    def _resolve_model_name(self) -> str:
        """Resolve requested model to an installed model name when possible."""
        if self._effective_model:
            return self._effective_model

        names = self._model_names()
        if self.model in names:
            self._effective_model = self.model
            return self._effective_model

        requested_base = self.model.split(":", 1)[0]
        latest_candidate = f"{requested_base}:latest"
        if latest_candidate in names:
            self._effective_model = latest_candidate
            logger.warning(
                "Requested model %s is unavailable; using %s",
                self.model,
                self._effective_model,
            )
            return self._effective_model

        for name in names:
            if name.split(":", 1)[0] == requested_base:
                self._effective_model = name
                logger.warning(
                    "Requested model %s is unavailable; using %s",
                    self.model,
                    self._effective_model,
                )
                return self._effective_model

        self._effective_model = self.model
        return self._effective_model

    def extract_entities(
        self,
        system_message: str,
        user_message: str,
    ) -> List[ExtractedEntity]:
        """
        Call Ollama to extract entities, with schema enforcement.

        Args:
            system_message: System message
            user_message: User message with text and instructions

        Returns:
            List of ExtractedEntity

        Raises:
            ValueError: If LLM output doesn't match schema
        """
        strict_suffix = (
            "\nReturn ONLY a raw JSON object matching the schema. "
            "No markdown, no backticks, no extra keys, no prose."
        )
        last_error: Optional[Exception] = None

        for attempt in range(2):
            effective_user_message = user_message
            if attempt == 1:
                effective_user_message = f"{user_message}{strict_suffix}"

            raw_output = self._chat(system_message, effective_user_message)

            try:
                json_str = self._extract_json(raw_output)
                llm_data = json.loads(json_str)
                output = LLMExtractionOutput(**llm_data)
                return output.entities
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                logger.warning(
                    "Failed to parse/validate LLM output on attempt %d: %s",
                    attempt + 1,
                    e,
                )

        logger.error("Invalid LLM output after retry: %s", last_error)
        raise ValueError(f"Invalid LLM output: {last_error}")

    def _chat(self, system_message: str, user_message: str) -> str:
        """Submit a non-streaming Ollama chat request and return text content."""
        model_name = self._resolve_model_name()
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
        }

        data = self._request_json("POST", "/api/chat", json_payload=payload)
        return data.get("message", {}).get("content", "")

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON object from text (handles markdown code blocks)."""
        text = text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]  # Remove ```json
        if text.startswith("```"):
            text = text[3:]  # Remove ```
        if text.endswith("```"):
            text = text[:-3]  # Remove trailing ```

        return text.strip()

    def health_check(self) -> bool:
        """Check if Ollama is reachable and model is available."""
        try:
            self._request_json("GET", "/api/tags", timeout=5.0)
            return True
        except Exception:
            return False

    def get_model_runtime_info(self) -> Optional[ModelRuntimeInfo]:
        """Get runtime metadata for the configured model, if available."""
        try:
            payload = self._request_json("GET", "/api/tags", timeout=5.0)
        except Exception as e:
            logger.warning("Failed to read Ollama model tags: %s", e)
            return None

        models = payload.get("models", [])
        if not isinstance(models, list):
            return None

        requested = self.model
        requested_base = requested.split(":", 1)[0]

        selected = None
        for model in models:
            if model.get("name") == requested:
                selected = model
                break
        if not selected:
            for model in models:
                name = model.get("name", "")
                if name.split(":", 1)[0] == requested_base:
                    selected = model
                    break
        if not selected:
            return ModelRuntimeInfo(
                requested_model=requested,
                resolved_model=None,
                quantization_level=None,
            )

        details = selected.get("details", {}) or {}
        quantization = details.get("quantization_level") or selected.get("quantization_level")

        return ModelRuntimeInfo(
            requested_model=requested,
            resolved_model=selected.get("name"),
            quantization_level=quantization,
        )

    def close(self):
        """Close the HTTP client."""
        self.client.close()
