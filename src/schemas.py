"""Pydantic schemas and data contracts for Credit Scoring serving.

Defines input validation contracts with field constraints, output response models,
and rich OpenAPI schema examples for documentation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class CreditApplicantInput(BaseModel):
    """Raw credit applicant feature payload."""

    LIMIT_BAL: float = Field(
        ...,
        gt=0,
        description="Amount of given credit (in NT dollars), including individual and family credit.",
        json_schema_extra={"example": 50000.0}
    )
    SEX: int = Field(
        ...,
        ge=1,
        le=2,
        description="Gender (1 = male, 2 = female).",
        json_schema_extra={"example": 2}
    )
    EDUCATION: int = Field(
        ...,
        ge=0,
        le=6,
        description="Education (1 = graduate school, 2 = university, 3 = high school, 0/4/5/6 = others).",
        json_schema_extra={"example": 2}
    )
    MARRIAGE: int = Field(
        ...,
        ge=0,
        le=3,
        description="Marital status (1 = married, 2 = single, 3 = divorce, 0 = others).",
        json_schema_extra={"example": 1}
    )
    AGE: int = Field(
        ...,
        ge=18,
        le=100,
        description="Age in years (18-100).",
        json_schema_extra={"example": 35}
    )

    # Repayment status over past 6 months (September to April)
    PAY_1: int = Field(
        ...,
        ge=-2,
        le=8,
        description="Repayment status in September (-2=no consumption, -1=paid in full, 0=revolving, 1-8=months delayed).",
        json_schema_extra={"example": 0}
    )
    PAY_2: int = Field(..., ge=-2, le=8, description="Repayment status in August.", json_schema_extra={"example": 0})
    PAY_3: int = Field(..., ge=-2, le=8, description="Repayment status in July.", json_schema_extra={"example": 0})
    PAY_4: int = Field(..., ge=-2, le=8, description="Repayment status in June.", json_schema_extra={"example": 0})
    PAY_5: int = Field(..., ge=-2, le=8, description="Repayment status in May.", json_schema_extra={"example": 0})
    PAY_6: int = Field(..., ge=-2, le=8, description="Repayment status in April.", json_schema_extra={"example": 0})

    # Bill statement amounts
    BILL_AMT1: float = Field(..., description="Bill statement amount in September (NT dollars).", json_schema_extra={"example": 25000.0})
    BILL_AMT2: float = Field(..., description="Bill statement amount in August.", json_schema_extra={"example": 24000.0})
    BILL_AMT3: float = Field(..., description="Bill statement amount in July.", json_schema_extra={"example": 22000.0})
    BILL_AMT4: float = Field(..., description="Bill statement amount in June.", json_schema_extra={"example": 21000.0})
    BILL_AMT5: float = Field(..., description="Bill statement amount in May.", json_schema_extra={"example": 19000.0})
    BILL_AMT6: float = Field(..., description="Bill statement amount in April.", json_schema_extra={"example": 18000.0})

    # Previous payment amounts
    PAY_AMT1: float = Field(..., ge=0, description="Amount paid in September (NT dollars).", json_schema_extra={"example": 2000.0})
    PAY_AMT2: float = Field(..., ge=0, description="Amount paid in August.", json_schema_extra={"example": 2000.0})
    PAY_AMT3: float = Field(..., ge=0, description="Amount paid in July.", json_schema_extra={"example": 1800.0})
    PAY_AMT4: float = Field(..., ge=0, description="Amount paid in June.", json_schema_extra={"example": 1500.0})
    PAY_AMT5: float = Field(..., ge=0, description="Amount paid in May.", json_schema_extra={"example": 1500.0})
    PAY_AMT6: float = Field(..., ge=0, description="Amount paid in April.", json_schema_extra={"example": 1500.0})

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
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
        }
    )


class CreditScoreResponse(BaseModel):
    """Credit scoring prediction response."""

    probability_of_default: float = Field(..., description="Estimated probability of default (0.0 to 1.0).")
    credit_score: int = Field(..., description="FICO-equivalent credit score ranging between 300 and 850.")
    risk_tier: str = Field(..., description="Categorical risk tier (Low Risk, Moderate Risk, High Risk).")
    decision: str = Field(..., description="Underwriting decision: APPROVE or DECLINE.")
    threshold_used: float = Field(..., description="Decision threshold applied to default probability.")


class BatchApplicantsInput(BaseModel):
    """Batch input wrapper for scoring multiple applicants."""

    applicants: List[CreditApplicantInput]


class BatchCreditScoreResponse(BaseModel):
    """Batch response containing scores for all submitted applicants."""

    total_records: int
    approved_count: int
    declined_count: int
    results: List[CreditScoreResponse]


class HealthCheckResponse(BaseModel):
    """API health and model metadata response."""

    status: str
    model_loaded: bool
    pipeline_path: str
    api_version: str
