# PII Anonymizer v2

## Quick Start

### Prerequisites
- Docker and Docker Compose
- GPU (optional, for faster inference)

### Setup

1. **Set the pseudonym secret** (required for deterministic token generation):
   ```bash
   export PSEUDONYM_SECRET="your-very-secret-key-min-32-chars"
   ```

2. **Start services**:
   ```bash
   docker-compose up --build
   ```

3. **Pull Ollama model** (one-time, first run):
   ```bash
   docker exec pii-anonymizer-ollama ollama pull llama3.1
   ```

### API Usage

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Anonymize Text
```bash
curl -X POST http://localhost:8000/v2/anonymize \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-123",
    "template_id": "default-pii-v1",
    "text": "John Doe from Acme Corp called me at 555-1234.",
    "render_mode": "structural"
  }'
```

Response (structural mode):
```json
{
  "anonymized_text": "<<PERSON:K7D2QH>> from <<ORG:1B9M3X>> called me at <<PHONE:XYZ123>>.",
  "mapping": {
    "token_to_original": {
      "<<PERSON:K7D2QH>>": "John Doe",
      "<<ORG:1B9M3X>>": "Acme Corp",
      "<<PHONE:XYZ123>>": "555-1234"
    },
    "meta": { ... }
  }
}
```

#### Deanonymize Text
```bash
curl -X POST http://localhost:8000/v2/deanonymize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "<<PERSON:K7D2QH>> from <<ORG:1B9M3X>> called me.",
    "mapping": {
      "token_to_original": {
        "<<PERSON:K7D2QH>>": "John Doe",
        "<<ORG:1B9M3X>>": "Acme Corp"
      }
    }
  }'
```

#### List Templates
```bash
curl http://localhost:8000/v2/templates
```

#### Get Template
```bash
curl http://localhost:8000/v2/templates/default-pii-v1
```

#### Validate Template
```bash
curl -X POST http://localhost:8000/v2/templates/validate \
  -H "Content-Type: application/json" \
  -d '{...template JSON...}'
```

## Configuration

Environment variables (in `docker-compose.yml`):

- `PSEUDONYM_SECRET` (required): Secret for deterministic token generation
- `TEMPLATES_DIR`: Path to templates directory (default: `/app/templates`)
- `OLLAMA_BASE_URL`: Ollama server URL (default: `http://ollama:11434`)
- `LLM_MODEL`: Ollama model name (default: `llama3.1`)
- `LLM_CONCURRENCY`: Max concurrent LLM calls (default: 1)
- `CHUNKING_ENABLED`: Enable text chunking for large texts (default: true)
- `CHUNK_CHAR_TARGET`: Target chunk size in characters (default: 5000)
- `LOG_SENSITIVE`: Whether to log sensitive data (default: false)
- `TOKEN_ID_LEN`: Length of token IDs (default: 6)

## Project Structure

```
app/
  main.py                      # FastAPI app and endpoints
  config.py                    # Configuration from env
  schemas/
    models.py                  # Pydantic models
  templates/
    loader.py                  # Load templates from disk
    registry.py                # Template caching and lookup
    validator.py               # Template validation
    compiler.py                # Prompt compilation
  llm/
    ollama_client.py           # Ollama API wrapper
    inference_queue.py         # Bounded concurrency queue
  pipeline/
    detector.py                # Main anonymization pipeline
    normalizer.py              # Entity deduplication
    span_resolver.py           # Deterministic span resolution
    tokenization.py            # Placeholder insertion
    rendering.py               # Realistic fake generation
    deanonymizer.py            # Reversal logic
    chunker.py                 # Text chunking
  utils/
    hashing.py                 # HMAC token generation
    text_norm.py               # Text normalization
tests/
  test_acceptance.py           # Acceptance tests
templates/
  default-pii-v1.json          # Default template
```

## Tests

Run acceptance tests:
```bash
pip install -r requirements.txt
pytest tests/test_acceptance.py -v
```

## Architecture

**Three-plane pipeline:**

1. **Detection (LLM)**: Extract entities using Ollama + Llama model
2. **Transformation (Deterministic)**: Resolve spans, insert placeholders, generate session-stable tokens
3. **Deanonymization**: Reverse mappings back to originals

**Key features:**

- **Lossless reversal**: Exact reconstruction from placeholders
- **Session-stable tokens**: Same entity in same session → same token
- **No LLM offsets**: Deterministic span resolution in code
- **Realistic rendering**: Optional fake names while keeping tokens canonical
- **Chunking support**: Automatic chunking for large texts
- **Docker persistence**: Templates (bind mount) + Ollama models (named volume)

## Production Considerations

1. **PSEUDONYM_SECRET**: Use a strong, unique secret in production
2. **GPU support**: Set `OLLAMA_NUM_GPU=1` if GPU available
3. **Concurrency**: Tune `LLM_CONCURRENCY` for your GPU/CPU
4. **Logging**: Set `LOG_SENSITIVE=false` (never log raw text or mappings)
5. **Mapping storage**: User is responsible for storing mappings securely
6. **Model selection**: Customize `LLM_MODEL` for your inference needs

## API Versioning

This implementation uses `v2` endpoints per spec:
- `/v2/anonymize`
- `/v2/deanonymize`
- `/v2/templates`

## License

Part of the airllm project.
