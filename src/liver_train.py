"""Train and compare leakage-safe liver disease classifiers."""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "datasets" / "ilpd_combined.csv"
MODEL_PATH = PROJECT_DIR / "models" / "liver_model.pkl"
RESULT_DIR = PROJECT_DIR / "results" / "liver_model"
TARGET = "Selector"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    if set(df[TARGET].unique()) != {1, 2}:
        raise ValueError("Expected liver labels 1 and 2.")

    X = df.drop(columns=[TARGET])
    X = X.loc[:, ~X.columns.astype(str).str.startswith("Unnamed")]
    y = (df[TARGET] == 1).astype(int)
    return X, y


def make_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])


def make_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=5,
            learning_rate=0.04,
            loss_function="Logloss",
            verbose=False,
            random_seed=RANDOM_STATE,
            thread_count=-1,
        ),
    }


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "recall": "recall"}
    results = []
    fitted_pipelines = {}

    for name, classifier in make_models().items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("classifier", classifier),
        ])
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        results.append({
            "Model": name,
            "CV_ROC_AUC": scores["test_roc_auc"].mean(),
            "CV_Recall": scores["test_recall"].mean(),
            "CV_ROC_AUC_STD": scores["test_roc_auc"].std(),
            "CV_Recall_STD": scores["test_recall"].std(),
        })
        fitted_pipelines[name] = pipeline

    comparison = pd.DataFrame(results).sort_values(
        ["CV_Recall", "CV_ROC_AUC"],
        ascending=False,
    )
    comparison.to_csv(RESULT_DIR / "model_comparison.csv", index=False)
    best_name = comparison.iloc[0]["Model"]
    best_pipeline = fitted_pipelines[best_name].fit(X_train, y_train)

    test_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)
    test_metrics = pd.DataFrame([{
        "Model": best_name,
        "Test_ROC_AUC": roc_auc_score(y_test, test_probabilities),
        "Test_Recall": recall_score(y_test, test_predictions),
    }])
    test_metrics.to_csv(RESULT_DIR / "test_metrics.csv", index=False)
    with open(RESULT_DIR / "classification_report.txt", "w", encoding="utf-8") as report:
        report.write(classification_report(
            y_test,
            test_predictions,
            target_names=["Non-Liver Patient", "Liver Patient"],
            zero_division=0,
        ))

    # Save the bare fitted pipeline: app.py can call predict/predict_proba directly.
    joblib.dump(best_pipeline, MODEL_PATH)

    print("\nCross-validation comparison:")
    print(comparison.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(test_metrics.to_string(index=False))
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
