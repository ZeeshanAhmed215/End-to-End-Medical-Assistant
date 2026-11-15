import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# ==========================================
# 1. LOAD CLEANED HEART DATA
# ==========================================
DATA_FILE = r"D:\End-to-End-Medical-Assistant\datasets\clean_heart.csv" # Update to your clean heart filename if different
TARGET_COL = "target"                     # Target column name

print(f"Loading cleaned dataset from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

# Separate features (X) and target (y)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ==========================================
# 2. TRAIN-TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# ==========================================
# 3. MODEL TRAINING
# ==========================================
print("\nTraining Random Forest Classifier on Clean Heart Data...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42
)
rf_model.fit(X_train, y_train)

# ==========================================
# 4. EVALUATION & METRICS
# ==========================================
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n" + "="*40)
print("CLEAN HEART MODEL EVALUATION RESULTS")
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
model_filename = "heart_random_forest.pkl"
joblib.dump(rf_model, model_filename)
print(f"\nSUCCESS: Trained model saved to '{model_filename}'")