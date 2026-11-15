import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# ==========================================
# 1. LOAD SEPARATE TRAIN & TEST DATA
# ==========================================
TRAIN_FILE = r"D:\End-to-End-Medical-Assistant\datasets\diabetes_training.csv" # Update to your train filename
TEST_FILE = r"D:\End-to-End-Medical-Assistant\datasets\diabetes_testing.csv"     # Update to your test filename
TARGET_COL = "target"                          # Target column name

print(f"Loading training data from {TRAIN_FILE}...")
train_df = pd.read_csv(TRAIN_FILE)

print(f"Loading testing data from {TEST_FILE}...")
test_df = pd.read_csv(TEST_FILE)

# Separate features (X) and target (y) for both sets
X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]

X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# ==========================================
# 2. APPLY SMOTE (TRAINING DATA ONLY)
# ==========================================
print("\nApplying SMOTE to balance the training classes...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE training class distribution: {y_train.value_counts().to_dict()}")
print(f"After SMOTE training class distribution:  {y_train_resampled.value_counts().to_dict()}")

# ==========================================
# 3. MODEL TRAINING
# ==========================================
print("\nTraining Random Forest Classifier on Diabetes Data...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42
)
rf_model.fit(X_train_resampled, y_train_resampled)

# ==========================================
# 4. EVALUATION & METRICS (ON UNSEEN TEST FILE)
# ==========================================
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n" + "="*40)
print("DIABETES MODEL EVALUATION RESULTS (TEST SET)")
print("="*40)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("="*40)

# ==========================================
# 5. SAVE TRAINED MODEL
# ==========================================
model_filename = "diabetes_random_forest.pkl"
joblib.dump(rf_model, model_filename)
print(f"\nSUCCESS: Trained model saved to '{model_filename}'")