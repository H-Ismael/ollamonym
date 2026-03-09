"""FastAPI application and endpoints."""

import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import Body, FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.schemas.models import (
    AnonymizeStreamRequest,
    AnonymizeRequest,
    AnonymizeResponse,
    AnonymizationMapping,
    MappingMetadata,
    DeanonymizeRequest,
    DeanonymizeResponse,
    TemplatesListResponse,
    TemplateInfo,
    TemplateSchema,
    TemplateSaveResponse,
    TemplateDeleteResponse,
    TemplateValidationResult,
)
from app.templates.registry import TemplateRegistry
from app.templates.validator import validate_template
from app.llm.ollama_client import OllamaClient
from app.pipeline.detector import Detector
from app.pipeline.deanonymizer import Deanonymizer

logger = logging.getLogger(__name__)

# Global instances
template_registry: TemplateRegistry = None
ollama_client: OllamaClient = None
detector: Detector = None
TEMPLATE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
PROTECTED_TEMPLATE_PREFIX = "default-"
WEB_ROOT = Path(__file__).parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle: startup and shutdown."""
    global template_registry, ollama_client, detector

    # Startup
    logger.info("Starting up PII Anonymizer Service v2")
    template_registry = TemplateRegistry(config.TEMPLATES_DIR)
    ollama_client = OllamaClient(
        config.OLLAMA_BASE_URL,
        config.LLM_MODEL,
        fallback_urls=config.OLLAMA_FALLBACK_URLS,
        keep_alive=config.OLLAMA_KEEP_ALIVE,
        num_predict=config.LLM_NUM_PREDICT,
        temperature=config.LLM_TEMPERATURE,
    )

    # Check Ollama health
    if not ollama_client.health_check():
        logger.warning(f"Ollama health check failed at {config.OLLAMA_BASE_URL}")
    else:
        logger.info("Ollama health check passed")

    model_runtime_info = ollama_client.get_model_runtime_info()
    if model_runtime_info:
        logger.info(
            "Model runtime requested=%s resolved=%s quantization=%s",
            model_runtime_info.requested_model,
            model_runtime_info.resolved_model,
            model_runtime_info.quantization_level,
        )

    detector = Detector(
        template_registry=template_registry,
        ollama_client=ollama_client,
        pseudonym_secret=config.PSEUDONYM_SECRET,
        token_id_len=config.TOKEN_ID_LEN,
        chunking_enabled=config.CHUNKING_ENABLED,
        chunk_char_target=config.CHUNK_CHAR_TARGET,
        chunk_overlap_chars=config.CHUNK_OVERLAP_CHARS,
        chunk_max_parallel=min(config.CHUNK_MAX_PARALLEL, max(1, config.LLM_CONCURRENCY)),
        rule_preextract_enabled=config.RULE_PREEXTRACT_ENABLED,
    )

    logger.info(f"Loaded {len(template_registry.list_templates())} templates")

    yield

    # Shutdown
    logger.info("Shutting down PII Anonymizer Service")
    if detector:
        detector.shutdown()
    if ollama_client:
        ollama_client.close()


app = FastAPI(
    title="PII Anonymizer v2",
    description="Template-driven, session-stable, lossless PII anonymization",
    version="2.0.0",
    lifespan=lifespan,
)

if WEB_ROOT.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_ROOT)), name="web")


def _is_protected_template(template_id: str) -> bool:
    return template_id.startswith(PROTECTED_TEMPLATE_PREFIX)


def _assert_editable_template_id(template_id: str) -> None:
    if _is_protected_template(template_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Template '{template_id}' is protected and cannot be modified in-place.",
        )
    if not TEMPLATE_ID_PATTERN.match(template_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="template_id must match ^[a-z0-9][a-z0-9._-]*$",
        )


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/", tags=["UI"])
async def root_ui():
    """Serve the UX demo."""
    if not WEB_ROOT.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Web UI assets are missing.",
        )
    return RedirectResponse(url="/web/index.html")


# ============================================================================
# Anonymization Endpoints
# ============================================================================


@app.post("/v2/anonymize", response_model=AnonymizeResponse, tags=["Anonymization"])
async def anonymize(request: AnonymizeRequest):
    """
    Anonymize text using a template.

    Returns:
        - anonymized_text: Text with entities replaced by placeholders (or fakes)
        - mapping: Complete mapping including token_to_original, token_to_fake, etc.
    """
    try:
        # Validate template exists
        if not template_registry.template_exists(request.template_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template not found: {request.template_id}",
            )

        # Run anonymization pipeline
        anon_text, token_to_orig, token_to_fake, fake_to_token = (
            detector.detect_and_anonymize(request)
        )

        # Build response
        template = template_registry.get_template(request.template_id)
        model_runtime_info = ollama_client.get_model_runtime_info(template.llm.model)
        mapping = AnonymizationMapping(
            token_to_original=token_to_orig,
            token_to_fake=token_to_fake,
            fake_to_token=fake_to_token,
            meta=MappingMetadata(
                session_id=request.session_id,
                template_id=request.template_id,
                template_version=template.version,
                render_mode=request.render_mode,
                fake_provider=request.fake_provider,
                model_runtime=model_runtime_info,
            ),
        )

        return AnonymizeResponse(
            anonymized_text=anon_text,
            mapping=mapping,
        )

    except ValueError as e:
        logger.error(f"Anonymization error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except httpx.RequestError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM backend unavailable. Check OLLAMA_BASE_URL / network connectivity.",
        )
    except httpx.HTTPStatusError as e:
        detail = "LLM backend request failed."
        try:
            err_payload = e.response.json()
            err_text = err_payload.get("error")
            if err_text:
                detail = f"LLM backend error: {err_text}"
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
        )
    except Exception as e:
        logger.error(f"Unexpected error during anonymization: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@app.post("/v2/anonymize/stream", tags=["Anonymization"])
async def anonymize_stream(request: AnonymizeStreamRequest):
    """Stream showcase events: structured first, then realistic output."""
    if not template_registry.template_exists(request.template_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template not found: {request.template_id}",
        )

    async def _sse() -> AsyncGenerator[str, None]:
        def _evt(event: str, data: dict) -> str:
            payload = json.dumps(data, ensure_ascii=False)
            return f"event: {event}\ndata: {payload}\n\n"

        try:
            yield _evt(
                "phase",
                {"phase": "start", "template_id": request.template_id, "session_id": request.session_id},
            )

            structural_req = AnonymizeRequest(
                session_id=request.session_id,
                template_id=request.template_id,
                text=request.text,
                render_mode="structural",
                language=request.language,
            )
            structured_text, token_to_orig, _, _ = detector.detect_and_anonymize(structural_req)
            tokens = re.findall(r"\S+\s*", structured_text)
            current = ""
            for token in tokens:
                current += token
                yield _evt(
                    "structured_token",
                    {"token": token, "text_so_far": current, "length": len(current)},
                )
                if request.token_delay_ms:
                    await asyncio.sleep(request.token_delay_ms / 1000.0)
            yield _evt(
                "structured_done",
                {
                    "anonymized_text": structured_text,
                    "mapping": token_to_orig,
                },
            )

            realistic_req = AnonymizeRequest(
                session_id=request.session_id,
                template_id=request.template_id,
                text=request.text,
                render_mode="realistic",
                fake_provider=request.fake_provider,
                language=request.language,
            )
            realistic_text, _, token_to_fake, fake_to_token = detector.detect_and_anonymize(realistic_req)
            realistic_tokens = re.findall(r"\S+\s*", realistic_text)
            realistic_current = ""
            for token in realistic_tokens:
                realistic_current += token
                yield _evt(
                    "realistic_token",
                    {"token": token, "text_so_far": realistic_current, "length": len(realistic_current)},
                )
                if request.token_delay_ms:
                    await asyncio.sleep(request.token_delay_ms / 1000.0)
            yield _evt(
                "realistic_done",
                {
                    "anonymized_text": realistic_text,
                    "token_to_fake": token_to_fake or {},
                    "fake_to_token": fake_to_token or {},
                },
            )

            yield _evt("done", {"ok": True})
        except Exception as e:
            logger.error("Streaming error: %s", e, exc_info=True)
            yield _evt("error", {"detail": str(e)})

    return StreamingResponse(
        _sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v2/deanonymize", response_model=DeanonymizeResponse, tags=["Anonymization"])
async def deanonymize(request: DeanonymizeRequest):
    """
    Reverse anonymization.

    Accepts:
        - text: Anonymized text (with tokens or realistic fakes)
        - mapping: Mapping from anonymize endpoint

    Returns:
        - text: Original text
    """
    try:
        mapping = request.mapping
        token_to_original = mapping.get("token_to_original")
        fake_to_token = mapping.get("fake_to_token")

        result_text = Deanonymizer.deanonymize(
            request.text,
            token_to_original=token_to_original,
            fake_to_token=fake_to_token,
        )

        return DeanonymizeResponse(text=result_text)

    except Exception as e:
        logger.error(f"Deanonymization error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


# ============================================================================
# Template Management Endpoints
# ============================================================================


@app.get("/v2/templates", response_model=TemplatesListResponse, tags=["Templates"])
async def list_templates():
    """List all available templates."""
    template_ids = template_registry.list_templates()
    templates = []
    for tid in template_ids:
        t = template_registry.get_template(tid)
        if t:
            templates.append(
                TemplateInfo(
                    template_id=t.template_id,
                    version=t.version,
                    description=t.description,
                )
            )
    return TemplatesListResponse(templates=templates)


@app.get("/v2/templates/{template_id}", tags=["Templates"])
async def get_template(template_id: str):
    """Get a specific template."""
    template = template_registry.get_template(template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template not found: {template_id}",
        )
    return template.model_dump()


@app.post("/v2/templates/validate", response_model=TemplateValidationResult, tags=["Templates"])
async def validate(template_data: dict):
    """Validate a template JSON."""
    try:
        from app.schemas.models import TemplateSchema
        template = TemplateSchema(**template_data)
        valid, errors = validate_template(template)
        return TemplateValidationResult(valid=valid, errors=errors)
    except Exception as e:
        return TemplateValidationResult(
            valid=False,
            errors=[str(e)],
        )


@app.post("/v2/templates/save", response_model=TemplateSaveResponse, tags=["Templates"])
async def save_template(template_data: dict = Body(...)):
    """Create or overwrite a custom template on disk."""
    try:
        template = TemplateSchema(**template_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid template payload: {e}",
        )

    _assert_editable_template_id(template.template_id)
    valid, errors = validate_template(template)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"errors": errors},
        )

    template_path = config.TEMPLATES_DIR / f"{template.template_id}.json"
    template_path.write_text(template.model_dump_json(indent=2), encoding="utf-8")
    template_registry.reload()

    return TemplateSaveResponse(
        template_id=template.template_id,
        version=template.version,
        description=template.description,
        protected=False,
    )


@app.delete("/v2/templates/{template_id}", response_model=TemplateDeleteResponse, tags=["Templates"])
async def delete_template(template_id: str):
    """Delete a custom template from disk."""
    _assert_editable_template_id(template_id)
    template_path = config.TEMPLATES_DIR / f"{template_id}.json"
    if not template_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template not found: {template_id}",
        )

    template_path.unlink()
    template_registry.reload()
    return TemplateDeleteResponse(detail=f"Deleted template: {template_id}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
