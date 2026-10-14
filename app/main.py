"""Production FastAPI Application for Credit Scoring & Risk Decisioning.

Exposes RESTful endpoints for:
- Health and model readiness checks
- Real-time single applicant scoring & decisioning
- High-throughput batch scoring
- CSV file upload batch inference
"""

import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.schemas import (
    CreditApplicantInput,
    CreditScoreResponse,
    BatchApplicantsInput,
    BatchCreditScoreResponse,
    HealthCheckResponse
)
from src.predict import (
    load_pipeline,
    score_applicant,
    score_batch,
    CHAMPION_PIPELINE_PATH
)

API_VERSION = "1.0.0"


def get_pipeline():
    """Retrieves cached pipeline from app state or loads on demand."""
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        try:
            pipeline = load_pipeline(CHAMPION_PIPELINE_PATH)
            app.state.pipeline = pipeline
        except Exception as exc:
            print(f"[MODEL LOAD WARNING] Failed to load pipeline: {exc}")
            return None
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads model pipeline at startup and manages service lifecycle."""
    print(f"[STARTUP] Initializing Credit Scoring API v{API_VERSION}...")
    pipeline = get_pipeline()
    if pipeline is not None:
        print(f"[STARTUP] Successfully loaded champion pipeline from {CHAMPION_PIPELINE_PATH}")
    else:
        print(f"[STARTUP WARNING] Champion pipeline could not be loaded on startup.")
    yield
    print("[SHUTDOWN] Service shutting down.")


app = FastAPI(
    title="Credit Scoring & Risk Underwriting API",
    description="""
    Production machine learning inference service for credit card default risk assessment.
    
    Provides:
    - **Default Probability Estimation**: Calibrated risk probabilities.
    - **FICO-style Credit Scoring**: Translates default risk to standard 300-850 scores.
    - **Tiered Underwriting Recommendations**: Auto-Approve, Manual Review, or Decline.
    """,
    version=API_VERSION,
    lifespan=lifespan
)

# Enable CORS for frontend clients (e.g. Streamlit, React, mobile)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint with service information and documentation links."""
    return {
        "service": "Credit Scoring & Risk Underwriting API",
        "version": API_VERSION,
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"],
    summary="Health and model readiness check"
)
async def health_check():
    """Verifies service health and confirms the champion model pipeline is in memory."""
    pipeline = get_pipeline()
    is_loaded = pipeline is not None
    return HealthCheckResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        pipeline_path=str(CHAMPION_PIPELINE_PATH),
        api_version=API_VERSION
    )


@app.post(
    "/predict",
    response_model=CreditScoreResponse,
    tags=["Scoring"],
    summary="Score a single applicant"
)
async def predict_single(
    applicant: CreditApplicantInput,
    threshold: float = Query(
        0.50,
        ge=0.01,
        le=0.99,
        description="Decision threshold for classifying default risk (default: 0.50)."
    )
):
    """Evaluates a single credit applicant payload and produces risk score and decision."""
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded. Please ensure models/champion_pipeline.pkl exists."
        )

    try:
        raw_dict = applicant.model_dump()
        result = score_applicant(raw_dict, pipeline=pipeline, threshold=threshold)
        return CreditScoreResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchCreditScoreResponse,
    tags=["Scoring"],
    summary="Score multiple applicants in a single payload"
)
async def predict_batch(
    payload: BatchApplicantsInput,
    threshold: float = Query(
        0.50,
        ge=0.01,
        le=0.99,
        description="Decision threshold for classifying default risk."
    )
):
    """High-throughput batch evaluation of applicant list."""
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded."
        )

    if not payload.applicants:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch list must contain at least one applicant record."
        )

    try:
        records = [a.model_dump() for a in payload.applicants]
        df_batch = pd.DataFrame(records)
        scored_df = score_batch(df_batch, pipeline=pipeline, threshold=threshold)

        results: List[CreditScoreResponse] = []
        approved_count = 0
        declined_count = 0

        for _, row in scored_df.iterrows():
            res = CreditScoreResponse(
                probability_of_default=float(row["probability_of_default"]),
                credit_score=int(row["credit_score"]),
                risk_tier=str(row["risk_tier"]),
                decision=str(row["decision"]),
                threshold_used=threshold
            )
            results.append(res)
            if res.decision == "APPROVE":
                approved_count += 1
            else:
                declined_count += 1

        return BatchCreditScoreResponse(
            total_records=len(results),
            approved_count=approved_count,
            declined_count=declined_count,
            results=results
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(exc)}"
        )


@app.post(
    "/predict/csv",
    tags=["Scoring"],
    summary="Upload a CSV file for batch portfolio scoring"
)
async def predict_csv(
    file: UploadFile = File(..., description="CSV file containing raw applicant columns"),
    threshold: float = Query(0.50, ge=0.01, le=0.99)
):
    """Uploads a CSV file of applicants and returns the scored portfolio summary and rows."""
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded."
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files (.csv) are supported."
        )

    try:
        content = await file.read()
        df_raw = pd.read_csv(io.BytesIO(content))
        
        # If target column exists, drop it before inference
        if "default" in df_raw.columns:
            df_features = df_raw.drop(columns=["default"])
        else:
            df_features = df_raw

        scored_df = score_batch(df_features, pipeline=pipeline, threshold=threshold)

        summary = {
            "filename": file.filename,
            "total_rows": len(scored_df),
            "approved_count": int((scored_df["decision"] == "APPROVE").sum()),
            "declined_count": int((scored_df["decision"] == "DECLINE").sum()),
            "average_credit_score": round(float(scored_df["credit_score"].mean()), 1),
            "average_default_prob": round(float(scored_df["probability_of_default"].mean()), 4),
            "records": scored_df.to_dict(orient="records")
        }
        return JSONResponse(content=summary)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process CSV file: {str(exc)}"
        )
