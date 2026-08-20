"""
Run once (locally, not from Streamlit) to train/evaluate the detection model
and save its performance metrics to model_results.json for the Streamlit app
to display.

Usage:
    python export_model_results.py
"""

import json
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "win32_data"
OUTPUT_PATH = ROOT / "model_results.json"


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
    train_df = pd.read_parquet(DATA_DIR / "win32_detection_train_20pct.parquet")
    test_df = pd.read_parquet(DATA_DIR / "win32_test_detection.parquet")

    X_train, y_train = vectorize(train_df), train_df["label"]
    X_test, y_test = vectorize(test_df), test_df["label"]

    model = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": cm,  # rows/cols ordered [Benign(0), Malware(1)]
        "labels": ["Benign", "Malware"],
        "classification_report": report,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "feature_importance": dict(
            zip(X_train.columns.tolist(), model.feature_importances_.tolist())
        ),
    }

    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {OUTPUT_PATH}")
    print(f"Accuracy: {results['accuracy']:.4f}  ROC AUC: {results['roc_auc']:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()