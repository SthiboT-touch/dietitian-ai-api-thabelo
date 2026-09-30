from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

app = FastAPI(
    title="Dietitian AI API",
    description="AI-generated nutrition recommendations for dietitian review.",
    version="1.0.0",
    docs_url="/docs"  # Keeps Swagger UI at /docs
)

# --- IN-MEMORY RECOMMENDATIONS ---
ai_recommendations_db: Dict[str, dict] = {}

# --- PYDANTIC SCHEMAS ---

class AllergyEntry(BaseModel):
    allergen: str
    severity: str

class MedicalConditionEntry(BaseModel):
    condition_name: str
    icd_code: Optional[str] = None

class AIGenerateRequest(BaseModel):
    client_id: str
    dietary_goal: str
    allergies: List[AllergyEntry] = Field(default_factory=list)
    medical_conditions: List[MedicalConditionEntry] = Field(default_factory=list)

class RecommendationReview(BaseModel):
    action: str  # APPROVED, MODIFIED, REJECTED
    notes: Optional[str] = None


# --- ROUTE HANDLERS ---

# Redirect root '/' to '/docs' so visiting the site opens Swagger UI directly
@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs")

# Health check route that appears in Swagger UI under "Health Check"
@app.get("/health", tags=["Health Check"])
def read_root():
    return {"status": "online", "message": "Dietitian API Service Running..."}


# --- AI ENGINE & HUMAN-IN-THE-LOOP WORKFLOW ---

def generate_llama_plan(dietary_goal: str, client_profile: Optional[dict] = None) -> List[str]:
    profile = client_profile or {}
    context = {
        "dietary_goal": dietary_goal,
        "allergies": profile.get("allergies", []),
        "medical_conditions": profile.get("conditions", []),
    }
    request_body = json.dumps({
        "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
        "stream": False,
        "keep_alive": "30m",
        "format": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 5,
                }
            },
            "required": ["plan"],
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "Draft a short, general nutrition plan for review by a licensed dietitian. "
                    "Respect all listed allergies. Do not diagnose, prescribe treatment, "
                    "or make claims that the plan treats a medical condition. "
                    "Reply with exactly 3 to 5 short plain-text suggestions, one sentence each "
                    "(e.g. 'Breakfast: oatmeal with berries'). Do not nest days, meals, or "
                    "objects inside the list -- each list item must be a single short string."
                ),
            },
            {"role": "user", "content": json.dumps(context)},
        ],
    }).encode("utf-8")
    request = Request(
        f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')}/api/chat",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=280) as response:
            result = json.loads(response.read().decode("utf-8"))
        generated = json.loads(result["message"]["content"])
        plan = generated.get("plan")
        if not isinstance(plan, list) or not plan or not all(isinstance(item, str) for item in plan):
            raise ValueError("The model returned an invalid plan")
        return plan
    except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Llama service unavailable or returned an invalid response. Check Ollama and the configured model.",
        ) from exc

@app.post("/api/v1/ai/generate-plan", tags=["6. AI Engine & HITL Workflow"])
def trigger_ai_plan(data: AIGenerateRequest):
    client_profile = {
        "allergies": [allergy.model_dump() for allergy in data.allergies],
        "conditions": [condition.model_dump() for condition in data.medical_conditions],
    }
    generated_plan = generate_llama_plan(data.dietary_goal, client_profile)
    rec_id = f"rec_{uuid.uuid4().hex[:8]}"
    rec_data = {
        "recommendation_id": rec_id,
        "client_id": data.client_id,
        "dietary_goal": data.dietary_goal,
        "status": "PENDING_REVIEW",
        "generated_plan": generated_plan
    }
    ai_recommendations_db[rec_id] = rec_data
    return {"message": "AI recommendation drafted", "recommendation": rec_data}

@app.get("/api/v1/ai/recommendations/pending", tags=["6. AI Engine & HITL Workflow"])
def get_pending_recommendations():
    pending = [r for r in ai_recommendations_db.values() if r["status"] == "PENDING_REVIEW"]
    return pending

@app.patch("/api/v1/ai/recommendations/{recommendation_id}", tags=["6. AI Engine & HITL Workflow"])
def review_ai_recommendation(recommendation_id: str, data: RecommendationReview):
    if recommendation_id not in ai_recommendations_db:
        raise HTTPException(status_code=404, detail="Recommendation ID not found")
    ai_recommendations_db[recommendation_id]["status"] = data.action
    if data.notes:
        ai_recommendations_db[recommendation_id]["review_notes"] = data.notes
    return {"message": f"Recommendation status updated to {data.action}", "recommendation": ai_recommendations_db[recommendation_id]}
