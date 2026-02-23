# ollamonym

Template-driven, session-stable anonymization for teams that want to use LLM workflows while reducing data leakage risk.

## Why This Project Exists

Most teams want LLM-powered automation, but they cannot expose raw sensitive text to third-party providers without creating legal, security, and reputational risk.

ollamonym gives you a practical middle path:

- Detect sensitive entities locally
- Replace them deterministically with reversible tokens
- Keep text structurally usable for downstream LLM tasks
- Restore originals only when authorized

In short: **LLM utility without shipping raw private identifiers by default**.

## Core Value Proposition

- **Leakage-risk reduction**: sensitive fields are transformed before downstream processing.
- **Operational realism**: anonymized text remains readable and coherent.
- **Reversible by design**: exact restoration via mapping when needed.
- **Template extensibility**: add domain entities without changing core pipeline.
- **Session consistency**: repeated mentions stay stable within a session.
- **Provider flexibility**: run with local quantized models today, swap model/provider config as needs evolve.

## Key Features

- **Hybrid Detection Engine (Deterministic + Local Quantized LLM)**
  - Deterministic rule extraction for patterned data (`EMAIL`, `PHONE`, links, etc.).
  - Local Ollama-hosted quantized LLM extraction for contextual entities (`PERSON`, `ORG`, domain entities).
  - Configurable model/runtime via template and environment.
- **Template-Driven Entity Taxonomy**
  - Define `PERSON`, `ORG`, `LINKS`, `PRODUCT`, or any custom class in JSON.
- **Deterministic Placeholder Mode**
  - Example token: `<<PERSON:K7D2QH>>`
- **Realistic Rendering Mode**
  - Optional fake values for human-readable anonymized output.
- **Generic Post-Pass Alias Propagation**
  - Moving-window + token-overlap propagation can link full and partial mentions in-session (e.g., `Jensen Huang` and `Huang`).
- **No LLM Offsets Required**
  - LLM returns only `(entity_id, text)`; span resolution is deterministic in code.
- **Robust Span Resolution**
  - Boundary-safe matching prevents substring corruption (for example, avoids replacing `com` inside `company`).
- **Chunked + Bounded Parallel Inference**
  - Handles long documents with predictable concurrency.
- **Model Runtime Observability**
  - Response metadata includes requested/resolved model and quantization info.
- **Dockerized Deployment**
  - FastAPI + Ollama stack with persistent model volume.

## High-Level Architecture

1. **Detection Plane**
   - Template compilation
   - Hybrid extraction: deterministic rules + local quantized LLM
   - Normalization and deduplication
2. **Transformation Plane**
   - Deterministic span resolution
   - Placeholder insertion
   - Session-aware alias propagation (moving window + overlap policy)
   - Optional realistic rendering
3. **Reversal Plane**
   - Token/fake back-mapping to exact original text

## Input / Output Example

### Input (`POST /v2/anonymize`)

```json
{
  "session_id": "case-42",
  "template_id": "default-pii-v1",
  "text": "Jensen Huang leads NVIDIA. Visit www.tech-private.com",
  "render_mode": "structural",
  "language": "auto"
}
```

### Output (structural)

```json
{
  "anonymized_text": "<<PERSON:XXXXXX>> leads <<ORG:XXXXXX>>. Visit <<LINKS:XXXXXX>>",
  "mapping": {
    "token_to_original": {
      "<<PERSON:XXXXXX>>": "Jensen Huang",
      "<<ORG:XXXXXX>>": "NVIDIA",
      "<<LINKS:XXXXXX>>": "www.tech-private.com"
    },
    "meta": {
      "session_id": "case-42",
      "template_id": "default-pii-v1",
      "template_version": 3,
      "render_mode": "structural"
    }
  }
}
```

### Input (`POST /v2/anonymize`, realistic)

```json
{
  "session_id": "string_id_test_23",
  "template_id": "default-pii-v1",
  "text": "AI Overview Jensen Huang is the co-founder, President, and CEO of NVIDIA ... you can find more at www.tech-private.com ... for now Jensen Huang is doing great",
  "render_mode": "realistic",
  "language": "auto"
}
```

### Output (realistic)

```json
{
  "anonymized_text": "AI Overview William Adams is the co-founder, President, and CEO of Elliott, Wilson and Terry and father of Shannon Gomez MD and Maria Thompson ... A pivotal figure in the AI revolution, Adams has guided Elliott, Wilson and Terry ... you can find more at www.blue-connect.com ... for now William Adams is doing great",
  "mapping": {
    "token_to_original": {
      "<<PERSON:A2N72P>>": "Jensen Huang",
      "<<ORG:T5YKPW>>": "NVIDIA",
      "<<PERSON:XG6QHD>>": "Huang",
      "<<LINKS:XA4JRC>>": "www.tech-private.com"
    },
    "token_to_fake": {
      "<<PERSON:A2N72P>>": "William Adams",
      "<<ORG:T5YKPW>>": "Elliott, Wilson and Terry",
      "<<PERSON:XG6QHD>>": "Adams",
      "<<LINKS:XA4JRC>>": "www.blue-connect.com"
    },
    "fake_to_token": {
      "William Adams": "<<PERSON:A2N72P>>",
      "Elliott, Wilson and Terry": "<<ORG:T5YKPW>>",
      "Adams": "<<PERSON:XG6QHD>>",
      "www.blue-connect.com": "<<LINKS:XA4JRC>>"
    },
    "meta": {
      "session_id": "string_id_test_23",
      "template_id": "default-pii-v1",
      "template_version": 3,
      "render_mode": "realistic",
      "model_runtime": {
        "requested_model": "llama3.1:8b-instruct-q4_K_M",
        "resolved_model": "llama3.1:latest",
        "quantization_level": "Q4_K_M"
      }
    }
  }
}
```

## Where This Is Most Useful

- **LLM preprocessing gateway** for enterprise copilots
- **Support and CRM text handling** before summarization/classification
- **Medical/legal document workflows** requiring controlled exposure
- **Model evaluation datasets** needing repeatable anonymization
- **Cross-team AI enablement** where security/compliance gate AI usage

## Quick Start

### Prerequisites

- Docker + Docker Compose
- `PSEUDONYM_SECRET` environment variable

### Run

```bash
export PSEUDONYM_SECRET="replace-with-a-strong-secret"
docker compose up --build
```

### Health

```bash
curl http://localhost:8000/health
```

### API Endpoints

- `POST /v2/anonymize`
- `POST /v2/deanonymize`
- `GET /v2/templates`
- `GET /v2/templates/{template_id}`
- `POST /v2/templates/validate`

## Configuration Highlights

Important env vars:

- `OLLAMA_BASE_URL`
- `OLLAMA_FALLBACK_URLS`
- `LLM_MODEL`
- `OLLAMA_KEEP_ALIVE`
- `LLM_NUM_PREDICT`
- `LLM_TEMPERATURE`
- `LLM_CONCURRENCY`
- `CHUNK_CHAR_TARGET`
- `CHUNK_MAX_PARALLEL`
- `RULE_PREEXTRACT_ENABLED`
- `TOKEN_ID_LEN`

Template controls:

- entity definitions and examples
- placeholder format and pseudonym providers
- post-pass alias policy (`postpass_alias`)
- per-template model selection (`template.llm.model`)

## Security and Compliance Positioning

- Keeps sensitive originals out of downstream prompts by default.
- Supports self-hosted/local inference stacks.
- Allows strict control of what is reversible and by whom (through mapping handling policy).

Note: this reduces leakage risk materially, but final security posture still depends on infrastructure, access control, logging policy, and secret management.

## Adaptability by Design

- Add new entity classes in template JSON without pipeline rewrites.
- Tune matching behavior through policy knobs:
  - `postpass_alias.window_size`
  - `postpass_alias.min_overlap_tokens`
  - `postpass_alias.min_token_len`
  - `postpass_alias.entity_ids`
- Mix strict deterministic extraction with contextual LLM extraction per use case.
- Keep consistent anonymization across non-exact mentions in the same session.

## What’s Next

### Product / UX

- Web UI for:
  - live text testing
  - template editing
  - rule tuning and validation
  - side-by-side structural vs realistic preview

### Metrics and Governance

- Detection quality dashboard (precision/recall by entity class)
- Latency and throughput dashboards (p50/p95/p99)
- Drift alerts for template/model changes
- Session-memory effectiveness metrics (alias recovery success)
- Audit-friendly anonymization/deanonymization event logs

### Platform Evolution

- Pluggable provider strategies for advanced fake generation by entity family
- Multi-tenant policy isolation
- Optional external session memory (e.g., Redis) for horizontal scale

## Repository Notes

- Detailed implementation notes: `README_IMPLEMENTATION.md`
- Build summary: `IMPLEMENTATION.md`
- Full technical spec: `spec_readme.md`
