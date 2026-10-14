"""Model definitions and factories for Credit Scoring.

Supports:
- Random Forest
- XGBoost
- LightGBM
- CatBoost
"""

from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_tree_models(params: Optional[Dict[str, Any]] = None, random_state: int = 42) -> Dict[str, Any]:
    """Returns a dictionary of initialized candidate models.
    
    Args:
        params: Optional dictionary of hyperparameters per model (e.g., loaded from params.yaml).
        random_state: Random state seed.
    """
    rf_params = params.get("random_forest", {}) if params else {}
    xgb_params = params.get("xgboost", {}) if params else {}
    lgbm_params = params.get("lightgbm", {}) if params else {}
    cat_params = params.get("catboost", {}) if params else {}

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 100),
            max_depth=rf_params.get("max_depth", 8),
            min_samples_split=rf_params.get("min_samples_split", 20),
            min_samples_leaf=rf_params.get("min_samples_leaf", 10),
            class_weight=rf_params.get("class_weight", "balanced"),
            random_state=rf_params.get("random_state", random_state),
            n_jobs=rf_params.get("n_jobs", -1)
        ),
        "XGBoost": XGBClassifier(
            n_estimators=xgb_params.get("n_estimators", 150),
            max_depth=xgb_params.get("max_depth", 6),
            learning_rate=xgb_params.get("learning_rate", 0.05),
            scale_pos_weight=xgb_params.get("scale_pos_weight", 3.5),
            eval_metric=xgb_params.get("eval_metric", "logloss"),
            random_state=xgb_params.get("random_state", random_state),
            n_jobs=xgb_params.get("n_jobs", -1)
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=lgbm_params.get("n_estimators", 150),
            max_depth=lgbm_params.get("max_depth", 6),
            learning_rate=lgbm_params.get("learning_rate", 0.05),
            class_weight=lgbm_params.get("class_weight", "balanced"),
            random_state=lgbm_params.get("random_state", random_state),
            verbose=lgbm_params.get("verbose", -1),
            n_jobs=lgbm_params.get("n_jobs", -1)
        ),
        "CatBoost": CatBoostClassifier(
            iterations=cat_params.get("iterations", 300),
            depth=cat_params.get("depth", 6),
            learning_rate=cat_params.get("learning_rate", 0.05),
            auto_class_weights=cat_params.get("auto_class_weights", "Balanced"),
            random_state=cat_params.get("random_state", random_state),
            verbose=cat_params.get("verbose", 0)
        )
    }

    return models


def train_model(model: Any, X_train: Any, y_train: Any) -> Any:
    """Fits the model to the training data."""
    model.fit(X_train, y_train)
    return model