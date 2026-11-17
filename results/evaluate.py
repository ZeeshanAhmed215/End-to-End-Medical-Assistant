"""Evaluate all saved medical assistant models on their held-out test sets."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "datasets"
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_binary_split_dataset(filename, target):
    """Recreate a stratified holdout used by a binary training script."""
    df = pd.read_csv(DATASET_DIR / filename)
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def load_cancer_test_set():
    _, X_test, _, y_test = load_binary_split_dataset("cancer.csv", "target")
    return X_test, y_test


def load_heart_test_set():
    _, X_test, _, y_test = load_binary_split_dataset("clean_heart.csv", "target")
    return X_test, y_test


def load_diabetes_test_set():
    df = pd.read_csv(DATASET_DIR / "diabetes_testing.csv")
    return df.drop(columns=["target"]), df["target"]


def load_general_test_set():
    df = pd.read_csv(DATASET_DIR / "general.csv")
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
    df = df.drop_duplicates()

    target = "yellow_crust_ooze"
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(df[target])
    X = df.drop(columns=[target, "target"], errors="ignore")

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_test, y_test, y_encoder


def load_liver_test_set():
    df = pd.read_csv(DATASET_DIR / "ilpd_combined.csv")
    target = "Selector"
    df = df.dropna(subset=[target]).copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(int)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    X = df.drop(columns=[target])
    y = (df[target] == 1).astype(int)
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_test, y_test


def specificity_by_class(y_true, y_pred, classes):
    """Return one-vs-rest specificity for each class."""
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    specificities = []
    for index in range(len(classes)):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        true_negative = matrix.sum() - true_positive - false_positive - false_negative
        denominator = true_negative + false_positive
        specificities.append(
            true_negative / denominator if denominator else np.nan
        )
    return np.asarray(specificities, dtype=float)


def class_auc_metrics(y_true, probabilities, classes):
    """Calculate one-vs-rest ROC-AUC and PR-AUC values."""
    roc_values = []
    pr_values = []
    for index, class_value in enumerate(classes):
        binary_target = (np.asarray(y_true) == class_value).astype(int)
        class_probability = probabilities[:, index]
        if np.unique(binary_target).size == 2:
            roc_values.append(roc_auc_score(binary_target, class_probability))
            pr_values.append(average_precision_score(binary_target, class_probability))
        else:
            roc_values.append(np.nan)
            pr_values.append(np.nan)
    return np.asarray(roc_values), np.asarray(pr_values)


def evaluate_model(name, model, X_test, y_test, class_labels):
    """Evaluate one model and return summary and detailed class-level rows."""
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = np.asarray(model.classes_)
    class_labels = [str(label) for label in class_labels]

    specificities = specificity_by_class(y_test, y_pred, classes)
    class_roc_auc, class_pr_auc = class_auc_metrics(
        y_test,
        probabilities,
        classes,
    )
    is_binary = len(classes) == 2
    average = "binary" if is_binary else "macro"
    positive_class = 1 if is_binary and 1 in classes else classes[-1]
    positive_index = int(np.where(classes == positive_class)[0][0])

    if is_binary:
        precision = precision_score(y_test, y_pred, pos_label=positive_class, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=positive_class, zero_division=0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        specificity = specificities[positive_index]
        roc_auc = class_roc_auc[positive_index]
        pr_auc = class_pr_auc[positive_index]
    else:
        precision = precision_score(y_test, y_pred, average=average, zero_division=0)
        recall = recall_score(y_test, y_pred, average=average, zero_division=0)
        f1 = recall_score(y_test, y_pred, average=average, zero_division=0)
        report = classification_report(
            y_test,
            y_pred,
            labels=classes,
            target_names=class_labels,
            output_dict=True,
            zero_division=0,
        )
        f1 = report["macro avg"]["f1-score"]
        specificity = float(np.nanmean(specificities))
        roc_auc = float(np.nanmean(class_roc_auc))
        pr_auc = float(np.nanmean(class_pr_auc))

    report_text = classification_report(
        y_test,
        y_pred,
        labels=classes,
        target_names=class_labels,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=classes,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    (RESULT_DIR / f"{name}_classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    detailed_rows = []
    for index, label in enumerate(class_labels):
        class_mask = np.asarray(y_test) == classes[index]
        detailed_rows.append({
            "Model": name,
            "Class": label,
            "Precision": precision_score(
                y_test,
                y_pred,
                labels=classes,
                average=None,
                zero_division=0,
            )[index],
            "Recall": recall_score(
                y_test,
                y_pred,
                labels=classes,
                average=None,
                zero_division=0,
            )[index],
            "Specificity": specificities[index],
            "F1-Score": report_dict[label]["f1-score"],
            "ROC-AUC": class_roc_auc[index],
            "PR-AUC": class_pr_auc[index],
            "Support": int(class_mask.sum()),
        })

    summary = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Test Samples": len(y_test),
    }
    return summary, detailed_rows, confusion_matrix(y_test, y_pred, labels=classes), class_labels


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    cancer_model = joblib.load(MODEL_DIR / "cancer_random_forest.pkl")
    diabetes_model = joblib.load(MODEL_DIR / "diabetes_random_forest.pkl")
    heart_model = joblib.load(MODEL_DIR / "heart_random_forest.pkl")
    general_model, general_features, general_encoder = joblib.load(
        MODEL_DIR / "general_model.pkl"
    )
    liver_model = joblib.load(MODEL_DIR / "liver_model.pkl")

    cancer_X, cancer_y = load_cancer_test_set()
    diabetes_X, diabetes_y = load_diabetes_test_set()
    heart_X, heart_y = load_heart_test_set()
    general_X, general_y, general_encoder_from_data = load_general_test_set()
    liver_X, liver_y = load_liver_test_set()

    if list(general_X.columns) != list(general_features):
        raise ValueError("General test features do not match the saved model schema.")
    if not np.array_equal(
        general_encoder.classes_,
        general_encoder_from_data.classes_,
    ):
        raise ValueError("General label encoding does not match the saved model.")

    evaluations = [
        ("cancer", cancer_model, cancer_X, cancer_y, cancer_model.classes_),
        ("diabetes", diabetes_model, diabetes_X, diabetes_y, diabetes_model.classes_),
        ("heart", heart_model, heart_X, heart_y, heart_model.classes_),
        (
            "general",
            general_model,
            general_X,
            general_y,
            general_encoder.classes_,
        ),
        ("liver", liver_model, liver_X, liver_y, liver_model.classes_),
    ]

    summaries = []
    detailed_rows = []
    confusion_matrices = []
    for name, model, X_test, y_test, labels in evaluations:
        summary, rows, matrix, class_labels = evaluate_model(
            name,
            model,
            X_test,
            y_test,
            labels,
        )
        summaries.append(summary)
        detailed_rows.extend(rows)
        confusion_matrices.append((name, matrix, class_labels))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RESULT_DIR / "models_performance_summary.csv", index=False)
    pd.DataFrame(detailed_rows).to_csv(
        RESULT_DIR / "evaluation_metrics.csv",
        index=False,
    )

    columns = 2
    rows = (len(confusion_matrices) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(14, rows * 5))
    axes = np.atleast_1d(axes).ravel()
    for axis, (name, matrix, labels) in zip(axes, confusion_matrices):
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        axis.set_title(f"{name.title()} confusion matrix")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(range(len(labels)), labels, rotation=90)
        axis.set_yticks(range(len(labels)), labels)
        threshold = matrix.max() / 2 if matrix.size else 0
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    matrix[row_index, column_index],
                    ha="center",
                    va="center",
                    color="white" if matrix[row_index, column_index] > threshold else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes[len(confusion_matrices):]:
        axis.axis("off")
    figure.suptitle("Medical Assistant Model Confusion Matrices")
    figure.tight_layout()
    figure.savefig(RESULT_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(figure)

    print("=" * 110)
    print("MEDICAL ASSISTANT MODEL PERFORMANCE SUMMARY")
    print("=" * 110)
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("=" * 110)
    print(f"Saved artifacts to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
