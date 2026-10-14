"""Tests for TreeSHAP explainability engine and regulatory adverse action endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from src.predict import load_pipeline
from src.explainability import CreditExplainer

client = TestClient(app)

SAMPLE_APPLICANT = {
    "LIMIT_BAL": 80000.0,
    "SEX": 1,
    "EDUCATION": 2,
    "MARRIAGE": 2,
    "AGE": 29,
    "PAY_1": 1,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 65000.0,
    "BILL_AMT2": 62000.0,
    "BILL_AMT3": 58000.0,
    "BILL_AMT4": 55000.0,
    "BILL_AMT5": 50000.0,
    "BILL_AMT6": 48000.0,
    "PAY_AMT1": 2500.0,
    "PAY_AMT2": 2500.0,
    "PAY_AMT3": 2000.0,
    "PAY_AMT4": 2000.0,
    "PAY_AMT5": 2000.0,
    "PAY_AMT6": 2000.0
}


def test_credit_explainer_direct():
    """Verify CreditExplainer correctly extracts adverse and protective factors."""
    pipeline = load_pipeline()
    explainer = CreditExplainer(pipeline)
    explanation = explainer.explain_applicant(SAMPLE_APPLICANT, top_k=3)

    assert "expected_base_value" in explanation
    assert "adverse_factors" in explanation
    assert "protective_factors" in explanation

    adverse = explanation["adverse_factors"]
    protective = explanation["protective_factors"]

    for item in adverse:
        assert item["shap_impact"] > 0
        assert item["direction"] == "INCREASES_RISK"
        assert "label" in item

    for item in protective:
        assert item["shap_impact"] < 0
        assert item["direction"] == "DECREASES_RISK"
        assert "label" in item


def test_predict_with_explain_flag():
    """Verify POST /predict returns SHAP adverse factors when explain=true."""
    response = client.post("/predict?explain=true", json=SAMPLE_APPLICANT)
    assert response.status_code == 200
    data = response.json()

    assert data["adverse_factors"] is not None
    assert data["protective_factors"] is not None
    assert len(data["adverse_factors"]) > 0


def test_adverse_action_endpoint():
    """Verify POST /explain returns FCRA/ECOA compliant adverse action notice."""
    response = client.post("/explain?threshold=0.40", json=SAMPLE_APPLICANT)
    assert response.status_code == 200
    data = response.json()

    assert data["decision"] in ["APPROVE", "DECLINE"]
    assert "primary_adverse_reasons" in data
    assert "positive_factors" in data
    assert "Equal Credit Opportunity Act" in data["regulatory_statement"]
    assert len(data["primary_adverse_reasons"]) > 0
