from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from wine_transformers import BinarizeAlcoholTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "wine_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "wine_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "wine_metrics.json"


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if path.exists():
        df = pd.read_csv(path)
        if "target" in df.columns:
            return df.drop(columns=["target"]), df["target"]
        raise ValueError("Expected a target column or an alcohol column in the dataset.")

    wine = load_wine(as_frame=True)
    df = wine.frame.copy()
    return df.drop(columns=["target"]), df["target"]


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    if "alcohol" in numeric_features:
        numeric_features = [col for col in numeric_features if col != "alcohol"]
    categorical_features = [col for col in X.columns if col not in numeric_features and col != "alcohol"]

    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if "alcohol" in X.columns:
        transformers.append(("alcohol_bin", BinarizeAlcoholTransformer(threshold=13.0), ["alcohol"]))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = LogisticRegression(max_iter=2000, random_state=42)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    X, y = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "roc_auc_ovr": roc_auc_score(y_test, y_proba, multi_class="ovr"),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
