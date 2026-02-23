"""FastAPI application and endpoints."""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, status

from app import config
from app.schemas.models import (
    AnonymizeRequest,
    AnonymizeResponse,
    AnonymizationMapping,
    MappingMetadata,
    ModelRuntimeInfo,
    DeanonymizeRequest,
    DeanonymizeResponse,
    TemplatesListResponse,
    TemplateInfo,
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
model_runtime_info: ModelRuntimeInfo | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle: startup and shutdown."""
    global template_registry, ollama_client, detector, model_runtime_info

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


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


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
    global model_runtime_info
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
        mapping = AnonymizationMapping(
            token_to_original=token_to_orig,
            token_to_fake=token_to_fake,
            fake_to_token=fake_to_token,
            meta=MappingMetadata(
                session_id=request.session_id,
                template_id=request.template_id,
                template_version=template.version,
                render_mode=request.render_mode,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
