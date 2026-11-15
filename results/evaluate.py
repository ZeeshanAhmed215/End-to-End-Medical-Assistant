import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import joblib

# Define evaluation configuration for each model
models_config = {
    "Cancer Model": {
        "model_path": r"D:\End-to-End-Medical-Assistant\models\cancer_random_forest.pkl",
        "data_file": r"D:\End-to-End-Medical-Assistant\datasets\cancer.csv", # Unified file that was split
        "target_col": "target",
        "use_split": True # Recreate the same 80/20 split to test on the unseen holdout
    },
    "Diabetes Model": {
        "model_path": r"D:\End-to-End-Medical-Assistant\models\diabetes_random_forest.pkl",
        "data_file": r"D:\End-to-End-Medical-Assistant\datasets\diabetes_testing.csv", # Dedicated test file
        "target_col": "target",
        "use_split": False
    },
    "Heart Disease Model": {
        "model_path": r"D:\End-to-End-Medical-Assistant\models\heart_random_forest.pkl",
        "data_file": r"D:\End-to-End-Medical-Assistant\datasets\clean_heart.csv", # Cleaned unified file that was split
        "target_col": "target",
        "use_split": True # Recreate the same 80/20 split
    }
}

results = []

print("="*65)
print("UNIFIED MEDICAL ASSISTANT MODEL PERFORMANCE SUMMARY")
print("="*65)

for name, config in models_config.items():
    try:
        # Load model
        model = joblib.load(config["model_path"])
        
        # Load dataset
        df = pd.read_csv(config["data_file"])
        X = df.drop(columns=[config["target_col"]])
        y = df[config["target_col"]]
        
        # If the model was trained on a split from a single file, isolate the test set portion
        if config["use_split"]:
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_test, y_test = X, y
            
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
        
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4) if isinstance(roc_auc, float) else roc_auc
        })
        print(f"Successfully evaluated: {name}")
        
    except Exception as e:
        print(f"Could not evaluate {name}: {e}")

# Convert to DataFrame for a clean summary table
summary_df = pd.DataFrame(results)
print("\n" + "="*65)
print(summary_df.to_string(index=False))
print("="*65)

# Save summary report
summary_df.to_csv("models_performance_summary.csv", index=False)
print("\nSUCCESS: Master performance report saved to 'models_performance_summary.csv'")