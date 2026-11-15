import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. CONFIGURATION & LOADING
# ==========================================
INPUT_FILE = r"D:\End-to-End-Medical-Assistant\datasets\general_training.csv" # Update to your dataset path if needed
OUTPUT_FILE = "general_training_preprocessed.csv" 
TARGET_COL = "target"  # Your target column name

print(f"Loading dataset from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# Strip any accidental whitespace from column names
df.columns = df.columns.str.strip()

print(f"Initial shape: {df.shape}")
print(f"Columns found: {list(df.columns)}")

# Remove exact duplicate rows
initial_rows = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_rows - len(df)} duplicate rows.\n")

# ==========================================
# 2. TARGET & ID COLUMN HANDLING
# ==========================================
target_series = None
if TARGET_COL in df.columns:
    # Drop rows where the target column itself is NaN (essential for ML training)
    df = df.dropna(subset=[TARGET_COL])
    
    target_series = df[TARGET_COL].copy()
    df = df.drop(columns=[TARGET_COL])
    
    # Encode target if it is text/categorical
    if target_series.dtype == 'object' or target_series.dtype.name == 'category':
        le = LabelEncoder()
        target_series = pd.Series(le.fit_transform(target_series.astype(str)), name=TARGET_COL)
        print(f"Encoded target column '{TARGET_COL}' into numeric values.")
    else:
        target_series = pd.to_numeric(target_series, errors='coerce')
else:
    print(f"[WARNING] Target column '{TARGET_COL}' not found in columns!")
# ==========================================
# 3. MISSING VALUE IMPUTATION
# ==========================================
num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

# Numerical Imputation: Fill NaN with median
for col in num_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Categorical Imputation: Fill NaN with mode or 'Unknown'
for col in cat_cols:
    if df[col].isnull().any():
        if df[col].isnull().mean() > 0.5:
            df[col] = df[col].fillna('Unknown')
        else:
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else 'Unknown'
            df[col] = df[col].fillna(fill_val)

print("Missing value imputation completed.\n")

# ==========================================
# 4. CATEGORICAL ENCODING
# ==========================================
cat_cols = df.select_dtypes(include=['object', 'category']).columns

low_card_cols = [col for col in cat_cols if df[col].nunique() < 10]
high_card_cols = [col for col in cat_cols if df[col].nunique() >= 10]

# One-Hot Encoding for low cardinality
if low_card_cols:
    df = pd.get_dummies(df, columns=low_card_cols, drop_first=True, dtype=int)
    print(f"Applied One-Hot Encoding to: {low_card_cols}")

# Frequency Encoding for high cardinality
for col in high_card_cols:
    freq_map = df[col].value_counts(normalize=True).to_dict()
    df[col] = df[col].map(freq_map)
    print(f"Applied Frequency Encoding to: {col}")

print("\nCategorical encoding completed.\n")

## ==========================================
# 5. VERIFICATION & EXPORT
# ==========================================
# Re-attach target column if it was present
if target_series is not None:
    df[TARGET_COL] = target_series

# --- SAFETY NET: Fill any stubborn remaining NaNs ---
df = df.fillna(0) 

# Assertion checks
assert df.isnull().sum().sum() == 0, "Assertion Error: Dataset still contains missing values!"
non_numeric_cols = df.select_dtypes(exclude=['number']).columns
assert len(non_numeric_cols) == 0, f"Assertion Error: Non-numeric columns remaining: {list(non_numeric_cols)}"

# Export to CSV
df.to_csv(OUTPUT_FILE, index=False)
print("SUCCESS: Preprocessed data saved!")