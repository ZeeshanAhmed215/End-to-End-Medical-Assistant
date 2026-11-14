import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set file path
file_path = r"D:\End-to-End-Medical-Assistant\datasets\heart.csv"

# 1. Load Dataset
print("--- 1. LOADING DATASET ---")
if not os.path.exists(file_path):
    print(f"[ERROR] File not found at: {file_path}. Please check your path.")
else:
    df = pd.read_csv(file_path)
    print(f"Dataset successfully loaded! Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # 2. Display Basic Information & Data Types
    print("--- 2. DATASET INFO & TYPES ---")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head(), "\n")

    # 3. Check for Missing Values
    print("--- 3. MISSING VALUES CHECK ---")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found!")
    print("\n")

    # 4. Statistical Summary
    print("--- 4. STATISTICAL SUMMARY ---")
    print(df.describe().T, "\n")

    # 5. Target Variable Distribution (if applicable)
    # Adjust 'diagnosis' or 'target' based on your actual column name
    target_col = 'diagnosis' if 'diagnosis' in df.columns else 'target'
    if target_col in df.columns:
        print(f"--- 5. TARGET DISTRIBUTION ({target_col}) ---")
        print(df[target_col].value_counts(), "\n")

    # 6. Generate EDA Visualizations
    print("--- 6. GENERATING VISUALIZATIONS ---")
    os.makedirs("outputs", exist_ok=True)

    # Plot A: Target Class Count Plot
    if target_col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df[target_col], palette="Set2")
        plt.title(f"Distribution of {target_col}")
        plt.savefig(r"eda\outputs\heart_target_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Plot B: Correlation Heatmap (for numerical features)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    # Limit to first 10 columns if there are too many features to keep heatmap readable
    corr = numeric_df.iloc[:, :10].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap (First 10 Numerical Features)")
    plt.savefig(r"eda\outputs\heart_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("EDA plots successfully saved inside the 'outputs/' folder!")