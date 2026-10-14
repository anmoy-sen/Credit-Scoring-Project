"""Inference and scoring module for the Credit Scoring system.

Loads the champion pipeline artifact and exposes convenient methods for
single-record, batch, and dictionary predictions, computing:
- Probability of default
- FICO-style credit score (300 - 850)
- Risk tier category (Low, Medium, High)
- Binary approval / decline recommendation
"""

from pathlib import Path
from typing import Dict, Any, Union, List
import joblib
import numpy as np
import pandas as pd

from src.evaluation import probability_to_credit_score

CHAMPION_PIPELINE_PATH = Path("models/champion_pipeline.pkl")


def load_pipeline(path: Union[str, Path] = CHAMPION_PIPELINE_PATH):
    """Loads the fitted end-to-end pipeline."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Champion pipeline not found at: {p.resolve()}. Please train first.")
    return joblib.load(p)


def get_risk_tier(prob_default: float) -> str:
    """Classifies risk level based on default probability."""
    if prob_default < 0.15:
        return "Low Risk (Auto-Approve)"
    elif prob_default < 0.35:
        return "Moderate Risk (Manual Review)"
    else:
        return "High Risk (Decline)"


def score_applicant(
    data: Union[Dict[str, Any], pd.DataFrame],
    pipeline=None,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """Scores a single applicant from raw financial inputs.

    Args:
        data: Dict of raw feature key-values or single-row DataFrame.
        pipeline: Fitted scikit-learn pipeline (loads champion if None).
        threshold: Decision threshold for binary classification.

    Returns:
        Dict containing default probability, credit score, risk tier, and recommendation.
    """
    if pipeline is None:
        pipeline = load_pipeline()

    if isinstance(data, dict):
        df_input = pd.DataFrame([data])
    else:
        df_input = data.copy()

    prob = float(pipeline.predict_proba(df_input)[0, 1])
    score = int(probability_to_credit_score(np.array([prob]))[0])
    is_default = int(prob >= threshold)
    tier = get_risk_tier(prob)

    return {
        "probability_of_default": round(prob, 4),
        "credit_score": score,
        "risk_tier": tier,
        "decision": "DECLINE" if is_default == 1 else "APPROVE",
        "threshold_used": threshold
    }


def score_batch(
    df: pd.DataFrame,
    pipeline=None,
    threshold: float = 0.50
) -> pd.DataFrame:
    """Scores a batch of applicants in a DataFrame."""
    if pipeline is None:
        pipeline = load_pipeline()

    df_out = df.copy()
    probas = pipeline.predict_proba(df)[:, 1]
    scores = probability_to_credit_score(probas)
    tiers = [get_risk_tier(p) for p in probas]
    decisions = ["DECLINE" if p >= threshold else "APPROVE" for p in probas]

    df_out["probability_of_default"] = np.round(probas, 4)
    df_out["credit_score"] = scores
    df_out["risk_tier"] = tiers
    df_out["decision"] = decisions
    return df_out
