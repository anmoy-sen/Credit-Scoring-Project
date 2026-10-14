"""Automated integration and unit tests for Credit Scoring FastAPI service."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE_APPLICANT = {
    "LIMIT_BAL": 100000.0,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 32,
    "PAY_1": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 25000.0,
    "BILL_AMT2": 24000.0,
    "BILL_AMT3": 23000.0,
    "BILL_AMT4": 20000.0,
    "BILL_AMT5": 19000.0,
    "BILL_AMT6": 18000.0,
    "PAY_AMT1": 2500.0,
    "PAY_AMT2": 2500.0,
    "PAY_AMT3": 2000.0,
    "PAY_AMT4": 2000.0,
    "PAY_AMT5": 1800.0,
    "PAY_AMT6": 1800.0
}


def test_root_endpoint():
    """Verify welcome and metadata endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["docs_url"] == "/docs"


def test_health_check():
    """Verify health endpoint confirms model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_single_valid():
    """Verify single applicant scoring returns valid credit score and decision."""
    response = client.post("/predict?threshold=0.40", json=SAMPLE_APPLICANT)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["probability_of_default"] <= 1.0
    assert 300 <= data["credit_score"] <= 850
    assert data["decision"] in ["APPROVE", "DECLINE"]
    assert data["threshold_used"] == 0.40
    assert "Risk" in data["risk_tier"]


def test_predict_single_invalid_age():
    """Verify schema boundary rejection when age is under 18."""
    bad_payload = SAMPLE_APPLICANT.copy()
    bad_payload["AGE"] = 16  # Must be >= 18
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_single_invalid_limit():
    """Verify schema rejection on negative credit limit."""
    bad_payload = SAMPLE_APPLICANT.copy()
    bad_payload["LIMIT_BAL"] = -1000.0
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_batch_valid():
    """Verify batch endpoint scores all records and summarizes counts."""
    payload = {
        "applicants": [SAMPLE_APPLICANT, SAMPLE_APPLICANT]
    }
    response = client.post("/predict/batch?threshold=0.50", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert (data["approved_count"] + data["declined_count"]) == 2
    assert len(data["results"]) == 2


def test_predict_batch_empty():
    """Verify empty batch is rejected with 400 Bad Request."""
    response = client.post("/predict/batch", json={"applicants": []})
    assert response.status_code == 400


def test_predict_csv_upload():
    """Verify CSV upload file endpoint."""
    import io
    import pandas as pd
    df = pd.DataFrame([SAMPLE_APPLICANT, SAMPLE_APPLICANT])
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    
    files = {"file": ("test_applicants.csv", io.BytesIO(csv_bytes), "text/csv")}
    response = client.post("/predict/csv?threshold=0.40", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_rows"] == 2
    assert "average_credit_score" in data
    assert len(data["records"]) == 2
