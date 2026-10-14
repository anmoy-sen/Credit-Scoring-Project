"""Pipeline construction module for the Credit Scoring system.

Composes the custom feature engineer, the ColumnTransformer (imputation,
scaling, one-hot encoding), and candidate ML estimators into a unified,
deployable scikit-learn Pipeline.
"""

from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator

from src.features import (
    CreditFeatureTransformer,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)


def create_preprocessor_pipeline() -> ColumnTransformer:
    """Builds the numeric and categorical ColumnTransformer."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def create_full_pipeline(
    model: Optional[BaseEstimator] = None,
    drop_raw_amounts: bool = True
) -> Pipeline:
    """Creates a unified Pipeline from raw inputs to prediction.

    Pipeline structure:
      1. feature_engineer (CreditFeatureTransformer)
      2. preprocessor (ColumnTransformer)
      3. classifier (optional estimator)
    """
    steps = [
        ("feature_engineer", CreditFeatureTransformer(drop_raw_amounts=drop_raw_amounts)),
        ("preprocessor", create_preprocessor_pipeline())
    ]

    if model is not None:
        steps.append(("classifier", model))

    return Pipeline(steps=steps)
