"""End-to-end model training, evaluation, and champion selection script.

Orchestrates:
1. Ingestion of raw dataset.
2. Train/Test stratified splitting.
3. Unified pipeline creation (feature engineering + preprocessing + classifier).
4. Multi-model training (Random Forest, XGBoost, LightGBM, CatBoost).
5. Evaluation with financial cost metrics and threshold optimization.
6. Champion model selection and deployment artifact persistence.
7. DVC metrics and evaluation plots export.
"""

import sys
import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.models import get_tree_models
from src.pipeline import create_full_pipeline
from src.evaluation import (
    evaluate_classification,
    find_optimal_threshold,
    plot_confusion_matrix,
    plot_roc_curves,
    save_metrics_report
)


def load_params(params_path: str = "params.yaml") -> dict:
    """Loads parameters from YAML config."""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 60)
    print("  CREDIT SCORING PIPELINE - TRAINING & EVALUATION RUN")
    print("=" * 60)

    # 1. Load parameters
    params = load_params("params.yaml")
    data_cfg = params.get("data", {})
    model_cfg = params.get("model", {})
    hparams = params.get("hyperparameters", {})

    raw_path = data_cfg.get("raw_path", "data/raw/credit.csv")
    test_size = data_cfg.get("test_size", 0.2)
    random_state = data_cfg.get("random_state", 42)
    cost_fp = model_cfg.get("cost_fp", 1.0)
    cost_fn = model_cfg.get("cost_fn", 5.0)

    # 2. Ingest Data
    print(f"\n[1/6] Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f"      Rows: {len(df):,} (dropped {initial_rows - len(df)} duplicates)")

    if "default" not in df.columns:
        raise ValueError("Target column 'default' not found in raw data.")

    X = df.drop(columns=["default"])
    y = df["default"]

    # 3. Stratified Split
    print(f"\n[2/6] Splitting data (test_size={test_size}, stratify=y, seed={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"      Train set: {len(X_train):,} samples | Test set: {len(X_test):,} samples")

    # Save processed splits for reproducibility
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

    # 4. Instantiate Models
    print("\n[3/6] Initializing models (Random Forest, XGBoost, LightGBM, CatBoost)...")
    models = get_tree_models(params=hparams, random_state=random_state)

    plots_dir = Path("reports/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    fitted_pipelines = {}
    predicted_probas = {}

    # 5. Train & Evaluate each Candidate Model
    print("\n[4/6] Training and evaluating full end-to-end pipelines...")
    for model_name, classifier in models.items():
        print(f"\n---> Training Candidate: {model_name}")
        pipeline = create_full_pipeline(model=classifier)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        predicted_probas[model_name] = y_proba

        metrics = evaluate_classification(
            y_true=y_test.values,
            y_pred=y_pred,
            y_pred_proba=y_proba,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            model_name=model_name
        )

        opt_thresh = find_optimal_threshold(
            y_true=y_test.values,
            y_pred_proba=y_proba,
            cost_fp=cost_fp,
            cost_fn=cost_fn
        )
        metrics["Optimal_Threshold"] = opt_thresh["threshold"]
        metrics["Optimal_Cost"] = opt_thresh["cost"]
        metrics["Optimal_Recall"] = opt_thresh["recall"]

        results[model_name] = metrics

        # Plot individual confusion matrix
        cm_filename = plots_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plot_confusion_matrix(y_test.values, y_pred, model_name=model_name, save_path=str(cm_filename))

    # Overlaid ROC curves
    plot_roc_curves(predicted_probas, y_test.values, save_path=str(plots_dir / "roc_curves.png"))

    # 6. Select Champion Model
    print("\n[5/6] Comparing models and selecting Champion...")
    comparison_df = pd.DataFrame(results).T
    print("\nModel Comparison Table:")
    print(comparison_df[["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1-Score", "Financial_Cost", "Optimal_Threshold"]])

    # Champion rule: Highest ROC-AUC
    champion_name = comparison_df["ROC-AUC"].idxmax()
    champion_pipeline = fitted_pipelines[champion_name]
    print(f"\n[CHAMPION] Champion Model Selected: {champion_name} (ROC-AUC: {comparison_df.loc[champion_name, 'ROC-AUC']:.4f})")

    # Save artifacts
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    champion_path = models_dir / "champion_pipeline.pkl"
    joblib.dump(champion_pipeline, champion_path)
    print(f"   Saved Champion Pipeline to: {champion_path}")

    # Also save comparison report
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(reports_dir / "model_comparison.csv")

    # Serialize DVC metrics JSON
    dvc_metrics = {
        "champion": {
            "name": champion_name,
            **results[champion_name]
        },
        "all_models": results
    }
    save_metrics_report(dvc_metrics, save_path="reports/metrics.json")
    print("   Saved DVC metrics to: reports/metrics.json")

    print("\n[6/6] Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
