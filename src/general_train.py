
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def run_pipeline():

    csv_path = r"D:\End-to-End-Medical-Assistant\datasets\general.csv"

    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # 1. Load dataset
    print("1. Loading dataset...")
    df = pd.read_csv(csv_path)

    # Remove unnamed columns
    df = df.loc[
        :, ~df.columns.astype(str).str.contains("^Unnamed")
    ]

    target_col = "yellow_crust_ooze"

    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        return

    # 2. Remove exact duplicate rows
    print(f"Original dataset shape: {df.shape}")

    df = df.drop_duplicates()

    print(f"Shape after removing duplicates: {df.shape}")

    # 3. Separate features and target
    y_raw = df[target_col]

    X = df.drop(
        columns=[target_col, "target"],
        errors="ignore"
    )

    # 4. Encode disease labels
    print("2. Encoding disease labels...")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # 5. Train/test split
    print("3. Splitting dataset (80/20)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # 6. Train Random Forest
    print("4. Training Random Forest Classifier...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Training completed successfully!")

    # 7. Predictions
    print("5. Evaluating model...")

    test_preds = model.predict(X_test)

    test_acc = accuracy_score(y_test, test_preds)

    # Generate classification report
    report = classification_report(
        y_test,
        test_preds,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    )

    print("\n========================================")
    print("       TRUE EVALUATION RESULTS")
    print("========================================")

    print(f"Test Set Accuracy: {test_acc * 100:.2f}%")

    print("\nDetailed Classification Report:")
    print(report)

    # ==========================================
    # 8. SAVE CLASSIFICATION REPORT
    # ==========================================

    report_path = "results/classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MEDICAL DISEASE MODEL - EVALUATION REPORT\n")
        f.write("=" * 55 + "\n\n")

        f.write(f"Model: Random Forest Classifier\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Testing samples: {len(X_test)}\n")
        f.write(f"Number of diseases: {len(label_encoder.classes_)}\n")
        f.write(f"Test accuracy: {test_acc * 100:.2f}%\n\n")

        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 55 + "\n")
        f.write(report)

    print(f"Classification report saved: {report_path}")

    # ==========================================
    # 9. SAVE EVALUATION METRICS AS CSV
    # ==========================================

    report_dict = classification_report(
        y_test,
        test_preds,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    metric_rows = []

    for class_name, values in report_dict.items():

        if isinstance(values, dict):
            metric_rows.append({
                "class": class_name,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1_score": values["f1-score"],
                "support": values["support"]
            })

        else:
            metric_rows.append({
                "class": class_name,
                "precision": None,
                "recall": None,
                "f1_score": values,
                "support": len(y_test)
            })

    metrics_df = pd.DataFrame(metric_rows)

    metrics_path = "results/evaluation_metrics.csv"

    metrics_df.to_csv(metrics_path, index=False)

    print(f"Evaluation metrics saved: {metrics_path}")

    # ==========================================
    # 10. GENERATE CONFUSION MATRIX
    # ==========================================

    cm = confusion_matrix(
        y_test,
        test_preds,
        labels=range(len(label_encoder.classes_))
    )

    fig_size = max(12, len(label_encoder.classes_) * 0.45)

    fig, ax = plt.subplots(
        figsize=(fig_size, fig_size)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=90,
        colorbar=True,
        values_format="d"
    )

    plt.title("Medical Disease Prediction - Confusion Matrix")
    plt.tight_layout()

    cm_path = "results/confusion_matrix.png"

    plt.savefig(
        cm_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Confusion matrix saved: {cm_path}")

    # ==========================================
    # 11. SAVE TRAINED MODEL
    # ==========================================

    model_output_path = "models/general_model.pkl"

    with open(model_output_path, "wb") as f:
        pickle.dump(
            (model, list(X.columns), label_encoder),
            f
        )

    print("\n========================================")
    print("       PIPELINE COMPLETED")
    print("========================================")

    print(f"Model saved: {model_output_path}")
    print("Evaluation files saved in results/")

    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    run_pipeline()
