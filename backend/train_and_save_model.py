"""
Train the Win32 malware detection LightGBM model on the real EMBER2024
parquet data and save it for the FastAPI backend to serve real predictions.

Reproduces notebooks/detection-model.ipynb exactly (same feature
engineering from the dataset's precomputed general/strings/imports JSON
columns, same LGBMClassifier hyperparameters), then additionally:
  - saves the fitted model to models/detection_model.pkl
  - saves the trained feature column order to models/feature_columns.json
  - refreshes model_results.json with real metrics

Usage:
    python backend/train_and_save_model.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "win32_data"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "detection_model.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
RESULTS_PATH = ROOT / "model_results.json"


def extract_import_features(x):
    if isinstance(x, str):
        x = json.loads(x)
    if not isinstance(x, dict):
        return {}

    features = {"num_imported_dlls": len(x)}
    features["num_imported_functions"] = sum(len(funcs) for funcs in x.values())
    features["avg_functions_per_dll"] = (
        features["num_imported_functions"] / len(x) if len(x) > 0 else 0
    )

    common_dlls = [
        "KERNEL32.dll", "ADVAPI32.dll", "USER32.dll",
        "WS2_32.dll", "SHELL32.dll", "OLE32.dll",
    ]
    for dll in common_dlls:
        features[f"has_{dll}"] = int(dll in x)
    return features


def extract_string_features(x):
    if isinstance(x, str):
        x = json.loads(x)
    return {
        "numstrings": x.get("numstrings", 0),
        "avg_string_length": x.get("avlength", 0),
        "num_printables": x.get("printables", 0),
        "string_entropy": x.get("entropy", 0),
    }


def extract_general_features(x):
    if isinstance(x, str):
        x = json.loads(x)
    return {
        "file_size": x.get("size", 0),
        "file_entropy": x.get("entropy", 0),
        "is_pe": x.get("is_pe", 0),
    }


def vectorize(df: pd.DataFrame) -> pd.DataFrame:
    imports_features = pd.DataFrame(df["imports"].apply(extract_import_features).tolist())
    strings_features = pd.DataFrame(df["strings"].apply(extract_string_features).tolist())
    general_features = pd.DataFrame(df["general"].apply(extract_general_features).tolist())
    return pd.concat([general_features, strings_features, imports_features], axis=1)


def main():
    train_path = DATA_DIR / "win32_detection_train_20pct.parquet"
    test_path = DATA_DIR / "win32_test_detection.parquet"
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run `python backend/download_data.py` first."
            )

    print(f"Loading {train_path.name} and {test_path.name} ...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    print("Vectorizing features ...")
    X_train, y_train = vectorize(train_df), train_df["label"]
    X_test, y_test = vectorize(test_df), test_df["label"]
    print(f"X_train shape: {X_train.shape}  X_test shape: {X_test.shape}")

    print("Training LGBMClassifier ...")
    model = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    results = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,  # rows/cols ordered [Benign(0), Malware(1)]
        "labels": ["Benign", "Malware"],
        "classification_report": report,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "feature_importance": dict(
            zip(X_train.columns.tolist(), model.feature_importances_.tolist())
        ),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    FEATURE_COLUMNS_PATH.write_text(json.dumps(X_train.columns.tolist(), indent=2))
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved feature columns to {FEATURE_COLUMNS_PATH}")
    print(f"Saved results to {RESULTS_PATH}")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC AUC:  {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
