# PII Anonymizer v2 — Template-Driven, Session-Stable, Lossless

**FastAPI + Llama via Ollama + Placeholder Tokenization (no offsets)**
**Dockerized + runnable with Docker Compose (persistent templates + persistent Ollama models)**

This document is a **complete implementation spec/README** intended to be consumed by a code-generation agent (Codex/Cursor).
It defines **requirements, API contracts, template format, prompting policy, algorithms, module boundaries, docker/compose layout, persistence, and acceptance tests**.
**No code appears here by design.**

---

## 1) Purpose

Build a modular service that:

* Detects **PII and template-defined sensitive entities** using a local Llama model via **Ollama**.
* Anonymizes text **losslessly** by replacing detected spans with deterministic placeholder tokens:

  * Example: `<<PERSON:K7D2QH>>`
* Optionally produces **realistic** anonymized output (human-looking names/orgs) while keeping placeholder tokens as the ground truth.
* Ensures **session-stable** replacements:

  * Within the same `session_id`, the same entity yields the same token and same fake.

The system is **latency-first**, avoids LLM offsets, and keeps prompts **atomic** and traceable.

---

## 2) Core invariants (must-haves)

1. **Lossless reversal**

* The anonymized output must be reversible to the exact original.
* Users are expected to keep the mapping returned by the anonymize endpoint.

2. **1–1 mapping**

* Each placeholder token corresponds to exactly one original string.
* No collisions allowed in `token_to_original`.

3. **Placeholder-first**

* Placeholders (`<<ENTITY:ID>>`) are the canonical anonymized form.
* Realistic rendering is a second layer applied after placeholderization.

4. **No offsets from LLM**

* LLM output contains only `{ entity_id, text }`.
* Span finding is deterministic in code.

5. **Session-stable**

* Same `(session_id, entity_id, canon(entity_text))` produces the same token id and (if realistic mode) the same fake.

6. **No confidence / no rationale**

* LLM must not output confidence scores or explanations.

---

## 3) High-level pipeline

### Plane A — Detection (LLM)

1. Load template by `template_id`
2. Compile prompt from template
3. (Optional) chunk large text
4. LLM extracts entities as `(entity_id, text)` JSON
5. Code deduplicates + normalizes entity strings

### Plane B — Transformation (Deterministic)

6. Deterministic span resolver finds occurrences in original text (no LLM offsets)
7. Replace spans with **placeholder tokens**
8. Build `token_to_original`
9. If `render_mode="realistic"`:

   * deterministically generate per-session pseudonyms
   * ensure uniqueness
   * replace tokens with fakes
   * return `token_to_fake` and `fake_to_token`

### Plane C — Deanonymize (Deterministic)

* Replace tokens back to original using provided mapping.
* Optionally reverse from realistic output via `fake_to_token`.

---

## 4) API contracts

### 4.1 `POST /v2/anonymize`

**Request**

```json
{
  "session_id": "custA-xyz",
  "template_id": "default-pii-v1",
  "text": "John Doe from Acme Corp called me.",
  "render_mode": "structural | realistic",
  "language": "auto | en | fr | ar"
}
```

**Response**

```json
{
  "anonymized_text": "<<PERSON:K7D2QH>> from <<ORG:1B9M3X>> called me.",
  "mapping": {
    "token_to_original": {
      "<<PERSON:K7D2QH>>": "John Doe",
      "<<ORG:1B9M3X>>": "Acme Corp"
    },
    "token_to_fake": {
      "<<PERSON:K7D2QH>>": "Mark Lewis",
      "<<ORG:1B9M3X>>": "Zeta Labs"
    },
    "fake_to_token": {
      "Mark Lewis": "<<PERSON:K7D2QH>>",
      "Zeta Labs": "<<ORG:1B9M3X>>"
    },
    "meta": {
      "session_id": "custA-xyz",
      "template_id": "default-pii-v1",
      "template_version": 3,
      "render_mode": "realistic"
    }
  }
}
```

**Rules**

* Always return `token_to_original`.
* If `render_mode="structural"`:

  * `token_to_fake` and `fake_to_token` may be omitted.
* Do not log raw text or extracted entities by default (see Ops section).

---

### 4.2 `POST /v2/deanonymize`

**Request**

```json
{
  "text": "Hello <<PERSON:K7D2QH>>.",
  "mapping": {
    "token_to_original": {
      "<<PERSON:K7D2QH>>": "John Doe"
    },
    "fake_to_token": {
      "Mark Lewis": "<<PERSON:K7D2QH>>"
    }
  }
}
```

**Response**

```json
{ "text": "Hello John Doe." }
```

**Rules**

* If text contains tokens, use `token_to_original`.
* Else, if text contains realistic fakes, use `fake_to_token` then `token_to_original`.
* No LLM calls.

---

### 4.3 Template endpoints

* `GET /v2/templates`
  Returns list of available templates with `template_id`, `version`, and description.

* `GET /v2/templates/{template_id}`
  Returns the active template JSON (or versioned retrieval if implemented).

* `POST /v2/templates/validate`
  Validates a template JSON against schema + sanity rules.

Templates are loaded from a mounted directory (see Docker section).

---

## 5) Template JSON specification

Templates are the dynamic configuration layer: **entity taxonomy + prompt compilation + canonicalization + replacement policy**.

### 5.1 Minimal template

```json
{
  "template_id": "default-pii-v1",
  "version": 3,
  "description": "Baseline PII",
  "entities": [
    {
      "id": "PERSON",
      "enabled": true,
      "instructions": "Detect full names of real people.",
      "examples": {
        "positive": ["John Doe", "Dr. Amina El Fassi"],
        "negative": ["Monday", "May", "Paris Agreement"]
      }
    },
    {
      "id": "ORG",
      "enabled": true,
      "instructions": "Detect organizations and institutions."
    }
  ],
  "llm": {
    "model": "llama3.1",
    "system": "Return JSON only. No extra keys.",
    "max_entities": 200
  },
  "canon": {
    "unicode_normalize": "NFKC",
    "collapse_whitespace": true,
    "casefold": true,
    "strip_outer_punct": true
  },
  "replacement": {
    "placeholder_format": "<<{ENTITY}:{ID}>>",
    "pseudonym": {
      "mode": "session_stable",
      "id_len": 6,
      "secret_env": "PSEUDONYM_SECRET",
      "providers": {
        "PERSON": "faker.person",
        "ORG": "faker.company"
      }
    }
  }
}
```

### 5.2 Template rules

* Entity `id` values define the taxonomy and must be stable.
* Adding new domain-sensitive entities is done by appending to `entities`.
* Example lists are optional but recommended for ambiguity control.
* Keep templates compact for latency.

---

## 6) LLM output schema (strict, atomic)

LLM must output:

```json
{
  "entities": [
    { "entity_id": "PERSON", "text": "John Doe" },
    { "entity_id": "ORG", "text": "Acme Corp" }
  ]
}
```

**Constraints**

* Only enabled template entity IDs are allowed.
* No offsets, confidence, rationale, or additional keys.
* If entity is not present in text, do not invent it.

---

## 7) Prompting policy (concise, low-latency)

### Goals

* Extraction only (atomic)
* JSON-only, schema-validated
* No reasoning text
* Short prompts for low latency

### Message structure

Use exactly:

* **System message (fixed)**
* **User message (template-compiled)**

#### System message (fixed)

Must enforce:

* Output JSON only, matching schema.
* Only extract from provided entity definitions.
* No offsets, no confidence, no rationale, no explanations.
* Do not hallucinate entities not present.

#### User message (compiled)

Blocks in order:

1. **Task header (1–2 lines)**

* “Extract sensitive entities from TEXT using ENTITY DEFINITIONS.”

2. **Entity definitions**
   For each enabled entity:

* `ID: instructions` (keep 1–2 lines)
* Optional examples (tiny caps):

  * max 2–3 positive and 2–3 negative

3. **Output constraints**

* Only key: `entities`
* Entity object shape: `{ "entity_id": "<ID>", "text": "<EXACT_SURFACE_FORM>" }`

4. **TEXT**

* Provide the raw text (or chunk text).

### Chunking prompt policy

* For large texts, include `CHUNK i/N` and chunk content only.
* Cache the compiled “entity definitions” string per `template_id@version` to reduce overhead.

---

## 8) Tokenization & session-stable IDs

### 8.1 Canonicalization `canon(text)`

Apply template `canon` settings:

* unicode normalize (e.g. NFKC)
* trim + collapse whitespace
* optional casefold
* optional strip outer punctuation

### 8.2 Deterministic token id

* `digest = HMAC_SHA256(PSEUDONYM_SECRET, session_id + "|" + entity_id + "|" + canon(text))`
* `TOKEN_ID = base32(digest)[:id_len]` (default 6)
* token: `<<{ENTITY_ID}:{TOKEN_ID}>>`

### 8.3 Collision handling (deterministic)

If two different canon strings produce the same token id within session/entity:

* retry deterministically using suffix:

  * `...|#1`, `...|#2`, etc.
* first unused id wins

---

## 9) Realistic rendering (optional) with uniqueness

If `render_mode="realistic"`:

* Deterministically generate fake values per token using a seed derived from the same HMAC basis.
* Enforce uniqueness per session/entity:

  * if generated fake is already used for that entity type in this request, re-roll deterministically with suffix counters.
* Return:

  * `token_to_fake` and `fake_to_token`

**Important**
Even in realistic mode, placeholders remain the canonical reversible anchor.

---

## 10) Deterministic span resolver (no offsets)

### Requirements

* handle repeated entities
* handle overlaps and substrings
* handle punctuation/casing variation (as permitted by normalization)
* deterministic output

### Recommended resolver

1. Build normalized **search view** of the full text and keep a mapping to original indices.
2. Normalize entity strings similarly.
3. Use **multi-pattern matching** (Aho–Corasick / trie) to find candidates.
4. Map matches back to original spans.
5. Resolve overlaps deterministically:

   * longer wins, then earlier start
   * optional template-driven type priority
6. Replace spans in a single pass (right-to-left or span list approach).

---

## 11) Chunking strategy (latency-first)

### Default

* Small text → single LLM call
* Large text → chunk by paragraph/sentence boundaries:

  * target size (chars/tokens)
  * bounded concurrency
  * merge/dedupe extracted entities
  * single global transform pass

### Concurrency control

* Dev 8GB GPU: start with concurrency = 1 (or 2 if stable)
* Server: tune to meet p95 latency goals

---

## 12) Service module boundaries (recommended)

The codegen agent should implement equivalent structure:

```
app/
  main.py                         # FastAPI app, routing, lifespan
  config.py                       # env config (ollama url, secrets, concurrency, chunking)
  schemas/                        # Pydantic models: API, template, LLM output
  templates/
    loader.py                     # load JSON templates from disk
    registry.py                   # template lookup + caching + version selection
    validator.py                  # schema validation + sanity checks
    compiler.py                   # prompt compilation from template
  llm/
    ollama_client.py              # schema-enforced chat calls
    inference_queue.py            # bounded concurrency, latency-first
  pipeline/
    detector.py                   # chunking + extraction + merge
    normalizer.py                 # canon/dedupe entity list
    span_resolver.py              # deterministic multi-pattern matching + overlap resolution
    tokenization.py               # placeholder insertion + token_to_original
    rendering.py                  # deterministic fake generation + uniqueness
    deanonymizer.py               # reverse mapping application
  utils/
    hashing.py                    # HMAC + base32
    text_norm.py                  # canon + search view mapping
```

---

## 13) Docker & Docker Compose (persistent templates + persistent models)

### 13.1 Services (recommended, not overengineered)

* `api`: FastAPI service
* `ollama`: Ollama runtime hosting the Llama model

This separation is the right level:

* isolates model runtime & GPU usage
* persists model cache independently
* allows restarting API without model churn
* easy future scaling of API replicas

### 13.2 Persistence strategy

**Persist**

1. Templates (bind mount):

* `./templates` → `/app/templates` (read-only)

2. Ollama model cache (named volume):

* `ollama_models:/root/.ollama`

**Do not persist (by default)**

* Reverse mappings (contain sensitive originals)
* Users provide mappings back when deanonymizing

Optional future: add encrypted storage only if explicitly required.

### 13.3 Compose behavior policy

* Shared docker network for API ↔ Ollama
* Healthchecks:

  * `ollama` health endpoint
  * `api` health endpoint
* GPU access for `ollama`:

  * enable NVIDIA runtime / device reservations as appropriate
* Model availability:

  * either pre-pull model in a one-time init step, or allow Ollama to pull on first use (pre-pull is recommended to reduce first-request latency)

### 13.4 Required environment variables

API:

* `TEMPLATES_DIR=/app/templates`
* `PSEUDONYM_SECRET=...`  (**required**)
* `DEFAULT_TEMPLATE_ID=default-pii-v1`
* `OLLAMA_BASE_URL=http://ollama:11434`
* `LLM_MODEL=llama3.1`
* `LLM_CONCURRENCY=1` (dev default)
* `CHUNKING_ENABLED=true`
* `CHUNK_CHAR_TARGET=5000` (tunable)
* `LOG_SENSITIVE=false` (default)

Ollama:

* model pulled to `ollama_models` volume

---

## 14) Caching strategy (safe + minimal)

Latency-first caching without persistent sensitive storage:

1. **Template compilation cache** (in-memory)

* key: `template_id@version`
* value: compiled prompt strings + enabled entity ids

2. **Warm model**

* keep Ollama running; avoid reload churn

Optional (off by default):

* short-lived in-memory extraction cache keyed by `(template_id@version, session_id, chunk_hash)` with small TTL
  (Use only if policy allows; avoid disk persistence.)

---

## 15) Operational policy (don’t leak secrets)

### Logging

Default logs:

* request id
* template id/version
* durations (LLM time, span resolution time)
* chunk counts
* number of entities found

Do **not** log:

* raw input text
* extracted entities
* mappings

Provide a guarded local-debug mode if needed.

---

## 16) Acceptance tests (must implement)

### Correctness

* `deanonymize(anonymize(text)) == text` (structural)
* For realistic mode: `deanonymize(realistic_output, mapping) == original`
* No token collisions in `token_to_original`

### Session stability

* same request repeated with same `session_id` yields same tokens
* realistic mode yields same fakes for same entities in same session

### Overlaps & repeats

* longer entity wins (“John Doe” vs “John”)
* repeated occurrences replaced consistently

### Unicode & multilingual

* Arabic/French example names
* unicode normalization does not break reconstruction

### Large text chunking

* chunking triggers
* merged results replaced correctly globally

### Strict JSON enforcement

* invalid LLM output yields deterministic error handling
* template validation rejects malformed templates

---

## 17) Implementation deliverables (for the agent)

The codegen agent must produce:

1. FastAPI endpoints:

* `/v2/anonymize`
* `/v2/deanonymize`
* `/v2/templates`, `/v2/templates/{id}`, `/v2/templates/validate`

2. Template system:

* load templates from volume
* validate against schema
* compile prompts

3. LLM integration:

* Ollama chat wrapper
* schema-enforced JSON output
* bounded concurrency queue

4. Deterministic pipeline:

* chunking + merge + dedupe
* span resolver (multi-pattern match + overlap resolution)
* placeholder token insertion + mapping
* deterministic realistic rendering with uniqueness

5. Dockerization:

* Dockerfile for API
* docker-compose with `api` + `ollama`
* persistent volumes: templates (bind) and `ollama_models` (named)

6. Tests implementing acceptance criteria.

---

## 18) Usage examples

### Structural mode

Input:

* `"John Doe from Acme Corp"`
  Output:
* `"<<PERSON:K7D2QH>> from <<ORG:1B9M3X>>"`
  Mapping:
* `token_to_original` sufficient to reverse.

### Realistic mode

Output:

* `"Mark Lewis from Zeta Labs"`
  Mapping includes:
* `token_to_original`
* `token_to_fake`
* `fake_to_token`

### Deanonymize

Input:

* realistic or tokenized text + mapping
  Output:
* original text restored exactly

---

## 19) Recommended build order

1. Template schema + loader + registry + validation
2. Prompt compiler + strict LLM output schema enforcement
3. Span resolver + placeholder tokenization + deanonymize
4. Session-stable token id + realistic rendering + uniqueness
5. Chunking + inference queue tuning for dev GPU
6. Docker compose + healthchecks + tests

---

### Final note

**Persist templates and Ollama models. Do not persist mappings by default.**
This keeps the system safer, simpler, and aligned with your “user provides mapping” workflow while remaining scalable and session-stable.
