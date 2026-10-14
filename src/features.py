"""Feature engineering and transformation module for Credit Scoring.

Defines a custom scikit-learn compatible transformer that converts raw
credit history data into engineered risk features suitable for production serving.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Raw input columns expected in production ingestion
RAW_FEATURE_COLUMNS: List[str] = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

CATEGORICAL_FEATURES: List[str] = ["SEX", "EDUCATION", "MARRIAGE"]

NUMERIC_FEATURES: List[str] = [
    "LIMIT_BAL", "AGE", "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "late_payment_count", "severe_late_count", "max_delay", "avg_pay_status",
    "PAY_RATIO_1", "PAY_RATIO_2", "PAY_RATIO_3", "PAY_RATIO_4", "PAY_RATIO_5", "PAY_RATIO_6",
    "avg_payment_ratio", "avg_bill", "max_bill",
    "avg_payment", "max_payment", "avg_utilization"
]

DROP_AFTER_ENGINEERING: List[str] = [
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]


class CreditFeatureTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for credit scoring feature engineering.
    
    Transforms raw applicant financial records into behavioral risk indices:
    - Delinquency aggregations (late count, severe late count, max/avg payment status)
    - Payment-to-bill ratios
    - Aggregated bills and payments
    - Credit utilization ratio
    """

    def __init__(self, drop_raw_amounts: bool = True):
        self.drop_raw_amounts = drop_raw_amounts
        self.pay_cols = [f"PAY_{i}" for i in range(1, 7)]
        self.bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
        self.pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
        self.ratio_cols = [f"PAY_RATIO_{i}" for i in range(1, 7)]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit method (stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw input DataFrame into engineered features."""
        # Work on a copy to avoid mutating external DataFrames
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=RAW_FEATURE_COLUMNS)
        else:
            X_df = X.copy()

        # 1. Delinquency and Delay Indicators
        X_df["late_payment_count"] = (X_df[self.pay_cols] >= 1).sum(axis=1)
        X_df["severe_late_count"] = (X_df[self.pay_cols] >= 3).sum(axis=1)
        X_df["max_delay"] = X_df[self.pay_cols].max(axis=1)
        X_df["avg_pay_status"] = X_df[self.pay_cols].mean(axis=1)

        # 2. Payment-to-Bill Ratios (safe division with absolute bill handling)
        for i in range(1, 7):
            X_df[f"PAY_RATIO_{i}"] = (
                X_df[f"PAY_AMT{i}"].clip(lower=0) /
                (X_df[f"BILL_AMT{i}"].abs() + 1.0)
            )
        X_df["avg_payment_ratio"] = X_df[self.ratio_cols].mean(axis=1)

        # 3. Bill & Payment Aggregations
        X_df["avg_bill"] = X_df[self.bill_cols].mean(axis=1)
        X_df["max_bill"] = X_df[self.bill_cols].max(axis=1)
        X_df["avg_payment"] = X_df[self.pay_amt_cols].mean(axis=1)
        X_df["max_payment"] = X_df[self.pay_amt_cols].max(axis=1)

        # 4. Credit Utilization Ratio
        X_df["avg_utilization"] = (
            X_df["avg_bill"].abs() /
            (X_df["LIMIT_BAL"].clip(lower=0) + 1.0)
        )

        # 5. Clean / Drop redundant raw columns if configured
        if self.drop_raw_amounts:
            cols_to_drop = [c for c in DROP_AFTER_ENGINEERING if c in X_df.columns]
            X_df.drop(columns=cols_to_drop, inplace=True)

        return X_df
