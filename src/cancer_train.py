import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# ==========================================
# 1. LOAD PREPROCESSED DATA
# ==========================================
DATA_FILE = r"D:\End-to-End-Medical-Assistant\datasets\cancer.csv"# Make sure this points to your cleaned dataset
TARGET_COL = "target"             # Update if your target column has a different name

print(f"Loading preprocessed dataset from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

# Separate features (X) and target (y)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ==========================================
# 2. TRAIN-TEST SPLIT
# ==========================================
# Stratify ensures the train and test sets have the same proportion of class labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# ==========================================
# 3. APPLY SMOTE (TRAINING DATA ONLY)
# ==========================================
print("\nApplying SMOTE to balance the training classes...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE class distribution: {y_train.value_counts().to_dict()}")
print(f"After SMOTE class distribution:  {y_train_resampled.value_counts().to_dict()}")

# ==========================================
# 4. MODEL TRAINING
# ==========================================
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42
)
rf_model.fit(X_train_resampled, y_train_resampled)

# ==========================================
# 5. EVALUATION & METRICS
# ==========================================
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n" + "="*40)
print("MODEL EVALUATION RESULTS")
print("="*40)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("="*40)

# ==========================================
# 6. SAVE TRAINED MODEL
# ==========================================
model_filename = "cancer_random_forest.pkl"
joblib.dump(rf_model, model_filename)
print(f"\nSUCCESS: Trained model saved to '{model_filename}'")