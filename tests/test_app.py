import json
from io import BytesIO
from urllib.error import URLError
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app


client = TestClient(app.app)


@pytest.fixture(autouse=True)
def clear_in_memory_data():
    app.ai_recommendations_db.clear()
    yield
    app.ai_recommendations_db.clear()


def ollama_response(plan):
    body = {"message": {"content": json.dumps({"plan": plan})}}
    return BytesIO(json.dumps(body).encode("utf-8"))


def test_generate_plan_returns_pending_recommendation():
    with patch("app.urlopen", return_value=ollama_response(["Breakfast: oatmeal"])):
        response = client.post(
            "/api/v1/ai/generate-plan",
            json={"client_id": "client-1", "dietary_goal": "More fiber"},
        )

    assert response.status_code == 200
    recommendation = response.json()["recommendation"]
    assert recommendation["generated_plan"] == ["Breakfast: oatmeal"]
    assert recommendation["status"] == "PENDING_REVIEW"


def test_generate_plan_forwards_allergies_and_conditions():
    with patch("app.urlopen", return_value=ollama_response(["Balanced meals"])) as mock_urlopen:
        response = client.post(
            "/api/v1/ai/generate-plan",
            json={
                "client_id": "client-2",
                "dietary_goal": "Balanced meals",
                "allergies": [{"allergen": "peanuts", "severity": "severe"}],
                "medical_conditions": [{"condition_name": "diabetes"}],
            },
        )

    request_body = json.loads(mock_urlopen.call_args.args[0].data.decode("utf-8"))
    context = json.loads(request_body["messages"][1]["content"])
    assert response.status_code == 200
    assert context["allergies"] == [{"allergen": "peanuts", "severity": "severe"}]
    assert context["medical_conditions"] == [{"condition_name": "diabetes", "icd_code": None}]


def test_generate_plan_returns_503_when_ollama_is_unreachable():
    with patch("app.urlopen", side_effect=URLError("Ollama unavailable")):
        response = client.post(
            "/api/v1/ai/generate-plan",
            json={"client_id": "client-3", "dietary_goal": "More vegetables"},
        )

    assert response.status_code == 503


def test_generate_plan_returns_503_for_malformed_model_response():
    malformed_response = BytesIO(json.dumps({"message": {"content": "not json"}}).encode("utf-8"))
    with patch("app.urlopen", return_value=malformed_response):
        response = client.post(
            "/api/v1/ai/generate-plan",
            json={"client_id": "client-4", "dietary_goal": "More vegetables"},
        )

    assert response.status_code == 503


def test_health_endpoint_reports_online():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_non_ai_food_endpoint_is_removed():
    response = client.get("/api/v1/foods")

    assert response.status_code == 404


def test_pending_recommendations_only_returns_pending_items():
    app.ai_recommendations_db.update({
        "rec-pending": {"recommendation_id": "rec-pending", "status": "PENDING_REVIEW"},
        "rec-approved": {"recommendation_id": "rec-approved", "status": "APPROVED"},
    })

    response = client.get("/api/v1/ai/recommendations/pending")

    assert response.status_code == 200
    assert [item["recommendation_id"] for item in response.json()] == ["rec-pending"]


def test_review_recommendation_updates_status_and_notes():
    app.ai_recommendations_db["rec-review"] = {
        "recommendation_id": "rec-review",
        "status": "PENDING_REVIEW",
    }

    response = client.patch(
        "/api/v1/ai/recommendations/rec-review",
        json={"action": "APPROVED", "notes": "Reviewed by dietitian"},
    )

    assert response.status_code == 200
    assert response.json()["recommendation"]["status"] == "APPROVED"
    assert response.json()["recommendation"]["review_notes"] == "Reviewed by dietitian"
