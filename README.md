# Dietitian AI API

FastAPI service that drafts nutrition recommendations with Ollama and keeps them pending dietitian review.

## Run locally

Start the API and Ollama:

```powershell
docker compose up --build -d
docker compose exec ollama ollama pull llama3.2
```

The model download is several gigabytes and is stored in the `ollama_data` volume. The API uses `llama3.2` by default. To use another model, set `OLLAMA_MODEL` in a `.env` file and pull that model from the Ollama container.

Open `http://localhost:8000/docs` to try the API. Health check: `GET /health`.

## AI endpoints

`POST /api/v1/ai/generate-plan` accepts the client identifier, goal, and optional allergy/condition context:

```json
{
  "client_id": "client-123",
  "dietary_goal": "More fiber and balanced meals",
  "allergies": [
    {"allergen": "peanuts", "severity": "severe"}
  ],
  "medical_conditions": [
    {"condition_name": "diabetes"}
  ]
}
```

Other AI workflow endpoints:

- `GET /api/v1/ai/recommendations/pending` lists recommendations awaiting review.
- `PATCH /api/v1/ai/recommendations/{recommendation_id}` records a review action and optional notes.

Recommendations remain `PENDING_REVIEW` when generated. They are suggestions for dietitian review, not a substitute for clinical judgment.

## Hosting

Build and deploy `Dockerfile.api`. Configure `OLLAMA_BASE_URL` to an Ollama server reachable from the API container and `OLLAMA_MODEL` to a model installed on that server. The server must expose the port provided by the hosting platform through `PORT` (defaults to `8000`).

Recommendations are currently stored in process memory, so they are lost when the API restarts and are not shared across multiple API instances. Add persistent storage and protect the endpoints with authentication/rate limiting before exposing this service publicly.
