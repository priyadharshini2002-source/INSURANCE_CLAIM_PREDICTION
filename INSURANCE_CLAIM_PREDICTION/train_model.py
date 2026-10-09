from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "insurance_claims.xlsx"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH, engine="openpyxl").replace("?", np.nan)
    target = "fraud_reported"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    y = df[target].astype(str).str.strip().map({"N": 0, "Y": 1})
    valid = y.notna()
    df, y = df.loc[valid].copy(), y.loc[valid].astype(int)

    # Remove identifiers and free-text location fields that are unsuitable as direct predictors.
    drop_cols = [c for c in ["fraud_reported", "policy_number", "insured_zip", "incident_location", "policy_bind_date", "incident_date"] if c in df.columns]
    X = df.drop(columns=drop_cols).copy()
    # Standardize mixed-type object columns (for example auto_model) while
    # preserving missing values for the imputation pipeline.
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].map(lambda value: str(value) if pd.notna(value) else np.nan)

    # Avoid duplicate information from raw dates; keep model features aligned at inference.
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    candidates = {
        "Random Forest": RandomForestClassifier(
            n_estimators=350, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=2,
            random_state=42
        ),
    }

    rows = []
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        row = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
        }
        rows.append(row)
        fitted[name] = (pipeline, pred)

    best_row = max(rows, key=lambda r: r["f1"])
    best_name = best_row["model"]
    best_model, best_pred = fitted[best_name]
    joblib.dump(best_model, MODEL_DIR / "insurance_fraud_model.joblib")

    report = classification_report(y_test, best_pred, labels=[0, 1], target_names=["No fraud reported", "Fraud reported"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, best_pred, labels=[0, 1]).tolist()
    metrics = {
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "target_distribution": {str(k): int(v) for k, v in df[target].astype(str).value_counts().to_dict().items()},
        "best_model": best_name,
        "model_comparison": rows,
        "classification_report": report,
        "confusion_matrix": cm,
        "features": list(X.columns),
        "positive_class": "Fraud reported (Y)",
        "note": "Educational dataset evaluation; not a guarantee of real-world performance.",
    }
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nINSURANCE CLAIM FRAUD MODEL TRAINING")
    print("=" * 45)
    print(f"Rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSelected model by F1: {best_name}")
    print(f"Saved: {MODEL_DIR / 'insurance_fraud_model.joblib'}")
    print(f"Saved: {MODEL_DIR / 'metrics.json'}")

if __name__ == "__main__":
    main()
