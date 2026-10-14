"""Explainable AI (XAI) and Regulatory Adverse Action module.

Uses TreeSHAP to decompose credit risk predictions into:
- Individual feature attributions (SHAP values)
- Adverse Action Reasons (ECOA / FCRA compliant factors for credit denial)
- Positive Credit Drivers (justifications for approval)
- Global portfolio feature importance
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

FEATURE_LABEL_MAP: Dict[str, str] = {
    "num__LIMIT_BAL": "Given Credit Line Limit",
    "num__AGE": "Applicant Age",
    "num__PAY_1": "Most Recent Payment Delay (September)",
    "num__PAY_2": "Payment Delay in August",
    "num__PAY_3": "Payment Delay in July",
    "num__PAY_4": "Payment Delay in June",
    "num__PAY_5": "Payment Delay in May",
    "num__PAY_6": "Payment Delay in April",
    "num__late_payment_count": "Total Delinquent Months in Past 6 Months",
    "num__severe_late_count": "Severe Delinquencies Count (Delay >= 3 Mos)",
    "num__max_delay": "Peak Repayment Delay Status",
    "num__avg_pay_status": "Average Repayment Delay Index",
    "num__PAY_RATIO_1": "Recent Payment-to-Bill Ratio (September)",
    "num__PAY_RATIO_2": "Payment-to-Bill Ratio (August)",
    "num__PAY_RATIO_3": "Payment-to-Bill Ratio (July)",
    "num__PAY_RATIO_4": "Payment-to-Bill Ratio (June)",
    "num__PAY_RATIO_5": "Payment-to-Bill Ratio (May)",
    "num__PAY_RATIO_6": "Payment-to-Bill Ratio (April)",
    "num__avg_payment_ratio": "Average Payment-to-Bill Ratio",
    "num__avg_bill": "Average Monthly Statement Balance",
    "num__max_bill": "Peak Monthly Statement Balance",
    "num__avg_payment": "Average Monthly Payment Made",
    "num__max_payment": "Peak Payment Amount Made",
    "num__avg_utilization": "Credit Limit Utilization Rate",
    "cat__SEX_1": "Gender: Male",
    "cat__SEX_2": "Gender: Female",
    "cat__EDUCATION_1": "Education: Graduate School",
    "cat__EDUCATION_2": "Education: University",
    "cat__EDUCATION_3": "Education: High School",
    "cat__EDUCATION_4": "Education: Other / Vocational",
    "cat__EDUCATION_5": "Education: Other",
    "cat__EDUCATION_6": "Education: Other",
    "cat__EDUCATION_0": "Education: Unknown",
    "cat__MARRIAGE_1": "Marital Status: Married",
    "cat__MARRIAGE_2": "Marital Status: Single",
    "cat__MARRIAGE_3": "Marital Status: Divorced / Other",
    "cat__MARRIAGE_0": "Marital Status: Unknown"
}


def clean_feature_label(raw_col: str) -> str:
    """Returns human-friendly label for a transformed pipeline feature column."""
    if raw_col in FEATURE_LABEL_MAP:
        return FEATURE_LABEL_MAP[raw_col]
    clean = raw_col.replace("num__", "").replace("cat__", "").replace("_", " ")
    return clean.title()


class CreditExplainer:
    """TreeSHAP explainer for credit scoring pipelines."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.fe = pipeline.named_steps["feature_engineer"]
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]

        # Cache transformed feature names
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.classifier)

    def _extract_shap_vector(self, shap_values: np.ndarray, sample_idx: int = 0) -> np.ndarray:
        """Extracts class 1 (default risk) attributions from SHAP result array."""
        arr = np.array(shap_values)
        if arr.ndim == 3:
            # Shape (N, F, 2): Take class 1 (default)
            return arr[sample_idx, :, 1]
        elif arr.ndim == 2:
            return arr[sample_idx, :]
        elif arr.ndim == 1:
            return arr
        else:
            raise ValueError(f"Unexpected SHAP values shape: {arr.shape}")

    def explain_applicant(
        self,
        applicant_data: Union[Dict[str, Any], pd.DataFrame],
        top_k: int = 4
    ) -> Dict[str, Any]:
        """Generates adverse action reason codes and positive drivers for an applicant.

        Args:
            applicant_data: Dict of raw applicant features or 1-row DataFrame.
            top_k: Number of top adverse and protective factors to return.

        Returns:
            Dict containing adverse factors, protective factors, base value, and all attributions.
        """
        if isinstance(applicant_data, dict):
            df_raw = pd.DataFrame([applicant_data])
        else:
            df_raw = applicant_data.copy()

        # Transform through pipeline steps prior to classifier
        X_fe = self.fe.transform(df_raw)
        X_trans = self.preprocessor.transform(X_fe)

        # Compute SHAP values
        raw_shap = self.explainer.shap_values(X_trans)
        shap_vec = self._extract_shap_vector(raw_shap, sample_idx=0)

        # Baseline value (expected value)
        expected_val = self.explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            expected_val = float(expected_val[1] if len(expected_val) > 1 else expected_val[0])
        else:
            expected_val = float(expected_val)

        # Map each feature to its impact
        all_attributions = []
        for i, col in enumerate(self.feature_names):
            val_in_fe = None
            raw_feature_name = col.replace("num__", "").replace("cat__", "")
            # Try to get raw/engineered value for context
            if raw_feature_name in X_fe.columns:
                val_in_fe = X_fe.iloc[0][raw_feature_name]
                if isinstance(val_in_fe, (np.floating, float)):
                    val_in_fe = round(float(val_in_fe), 2)

            impact = float(shap_vec[i])
            all_attributions.append({
                "feature": col,
                "label": clean_feature_label(col),
                "value": val_in_fe,
                "shap_impact": round(impact, 4),
                "direction": "INCREASES_RISK" if impact > 0 else "DECREASES_RISK"
            })

        # Adverse factors: Highest positive SHAP values (pushing towards default)
        adverse_factors = sorted(
            [a for a in all_attributions if a["shap_impact"] > 0],
            key=lambda x: x["shap_impact"],
            reverse=True
        )[:top_k]

        # Protective factors: Most negative SHAP values (pulling towards non-default)
        protective_factors = sorted(
            [a for a in all_attributions if a["shap_impact"] < 0],
            key=lambda x: x["shap_impact"]
        )[:top_k]

        return {
            "expected_base_value": round(expected_val, 4),
            "adverse_factors": adverse_factors,
            "protective_factors": protective_factors,
            "all_attributions": sorted(all_attributions, key=lambda x: abs(x["shap_impact"]), reverse=True)
        }

    def generate_waterfall_plot(
        self,
        applicant_data: Union[Dict[str, Any], pd.DataFrame],
        top_n: int = 10,
        save_path: Optional[str] = None
    ) -> None:
        """Generates a clean horizontal bar chart of the top feature attributions."""
        explanation = self.explain_applicant(applicant_data, top_k=top_n)
        attrs = explanation["all_attributions"][:top_n]

        labels = [a["label"] for a in reversed(attrs)]
        impacts = [a["shap_impact"] for a in reversed(attrs)]
        colors = ["#d62728" if x > 0 else "#2ca02c" for x in impacts]

        plt.figure(figsize=(9, 5))
        plt.barh(labels, impacts, color=colors, alpha=0.85)
        plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
        plt.xlabel("SHAP Impact on Default Risk (Red = Increases Risk, Green = Reduces Risk)")
        plt.title("Credit Risk Decision Attribution (TreeSHAP)")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
