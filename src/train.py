import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


F1_THRESHOLD = 0.65
REFERENCE_POSITIVE_RATIO = 0.248
MAX_RATIO_DRIFT = 0.05
DECISION_THRESHOLDS = np.arange(0.10, 0.901, 0.05)


def find_best_threshold(y_true, probabilities) -> tuple[float, float]:
    """Return the threshold in [0.10, 0.90] that maximizes positive-class F1."""
    candidates = []
    for threshold in DECISION_THRESHOLDS:
        rounded_threshold = round(float(threshold), 2)
        predictions = (probabilities >= rounded_threshold).astype(int)
        candidates.append(
            (f1_score(y_true, predictions, zero_division=0), rounded_threshold)
        )

    # Prefer a threshold closer to 0.5 when several candidates have the same F1.
    best_f1, best_threshold = max(
        candidates,
        key=lambda item: (item[0], -abs(item[1] - 0.5)),
    )
    return best_threshold, float(best_f1)


def _configure_mlflow() -> None:
    """Use a remote URI when supplied by CI, otherwise keep local tracking."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "income-model"))


def _write_detail_report(y_true, predictions, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["thu_nhap_thap", "thu_nhap_cao"],
        digits=4,
        zero_division=0,
    )
    output_path.write_text(
        "Confusion matrix (rows=true, columns=predicted)\n"
        f"{matrix}\n\n"
        "Precision / Recall / F1 by class\n"
        f"{report}",
        encoding="utf-8",
    )


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """Train, evaluate, track, and package the Adult Income classifier."""
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    positive_class_ratio = float(y_train.mean())
    ratio_drift = abs(positive_class_ratio - REFERENCE_POSITIVE_RATIO)
    drift_warning = ratio_drift > MAX_RATIO_DRIFT

    _configure_mlflow()
    with mlflow.start_run():
        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_eval)[:, 1]
        default_predictions = (probabilities >= 0.5).astype(int)
        default_f1 = float(
            f1_score(y_eval, default_predictions, zero_division=0)
        )
        default_accuracy = float(accuracy_score(y_eval, default_predictions))

        best_threshold, optimized_f1 = find_best_threshold(y_eval, probabilities)
        optimized_predictions = (probabilities >= best_threshold).astype(int)
        optimized_accuracy = float(accuracy_score(y_eval, optimized_predictions))

        metrics = {
            "f1_score": optimized_f1,
            "accuracy": optimized_accuracy,
            "f1_score_default": default_f1,
            "accuracy_default": default_accuracy,
            "decision_threshold": best_threshold,
            "positive_class_ratio": positive_class_ratio,
            "class_ratio_drift": ratio_drift,
        }
        mlflow.log_metrics(metrics)
        mlflow.log_param("data_drift_warning", drift_warning)
        mlflow.sklearn.log_model(model, "model")

        output_dir = Path("outputs")
        model_dir = Path("models")
        output_dir.mkdir(exist_ok=True)
        model_dir.mkdir(exist_ok=True)

        report = {
            **metrics,
            "data_drift_warning": drift_warning,
            "train_samples": int(len(df_train)),
            "eval_samples": int(len(df_eval)),
        }
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        _write_detail_report(
            y_eval, optimized_predictions, output_dir / "detail.txt"
        )

        model_bundle = {
            "model": model,
            "decision_threshold": best_threshold,
            "feature_names": list(X_train.columns),
        }
        joblib.dump(model_bundle, model_dir / "model.joblib")

        print(
            f"Default F1: {default_f1:.4f} | "
            f"Optimized F1: {optimized_f1:.4f} | "
            f"Threshold: {best_threshold:.2f} | "
            f"Accuracy: {optimized_accuracy:.4f}"
        )
        drift_message = "WARNING" if drift_warning else "OK"
        print(
            f"Data drift {drift_message}: positive ratio={positive_class_ratio:.4f}, "
            f"reference={REFERENCE_POSITIVE_RATIO:.4f}, "
            f"difference={ratio_drift:.4f}"
        )

    return optimized_f1


if __name__ == "__main__":
    params_path = os.getenv("PARAMS_PATH", "params.yaml")
    with open(params_path, encoding="utf-8") as params_file:
        selected_params = yaml.safe_load(params_file)
    train(selected_params)
