
"""
LIVER DISEASE - EXPLORATORY DATA ANALYSIS (EDA)
Dataset: Indian Liver Patient Dataset (ILPD)
Source: UCI Machine Learning Repository
UCI Dataset ID: 225

Run:
    python liver_eda.py

Outputs:
    datasets/liver/ilpd_raw.csv
    datasets/liver/ilpd_combined.csv
    results/liver_eda/
        dataset_info.txt
        missing_values.csv
        duplicate_summary.txt
        numerical_summary.csv
        categorical_summary.csv
        target_distribution.csv
        correlation_matrix.csv
        variables_information.csv
        *.png
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "datasets" / "liver"
RESULTS_DIR = BASE_DIR / "results" / "liver_eda"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

sns.set_theme(style="whitegrid", palette="viridis")

print("\n" + "=" * 65)
print("       INDIAN LIVER PATIENT DATASET - EDA")
print("=" * 65)


# ---------------------------------------------------------
# 2. LOAD DATA DIRECTLY FROM UCI
# ---------------------------------------------------------

def load_dataset():

    print("\n[1/10] Loading dataset from UCI Repository...")

    try:
        dataset = fetch_ucirepo(id=225)

        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()

        # Combine features and target
        df = pd.concat([X, y], axis=1)

        print("Dataset downloaded successfully!")

        print("\nDataset name:")
        print(dataset.metadata.name)

        print("\nDataset ID:")
        print(dataset.metadata.uci_id)

        print("\nOriginal dataset shape:")
        print(df.shape)

        # Save raw and combined datasets
        df.to_csv(DATA_DIR / "ilpd_raw.csv", index=False)
        df.to_csv(DATA_DIR / "ilpd_combined.csv", index=False)

        # Save UCI variable information
        dataset.variables.to_csv(
            RESULTS_DIR / "variables_information.csv",
            index=False
        )

        # Save dataset metadata
        with open(
            RESULTS_DIR / "dataset_info.txt",
            "w",
            encoding="utf-8"
        ) as file:

            file.write("INDIAN LIVER PATIENT DATASET\n")
            file.write("=" * 50 + "\n\n")

            file.write(f"Dataset Name: {dataset.metadata.name}\n")
            file.write(f"UCI ID: {dataset.metadata.uci_id}\n")
            file.write(f"Instances: {dataset.metadata.num_instances}\n")
            file.write(f"Features: {dataset.metadata.num_features}\n")
            file.write(f"Target Column: {dataset.metadata.target_col}\n\n")

            file.write("Dataset Description:\n")
            file.write(str(dataset.metadata.abstract))

        return df

    except Exception as error:
        print("\nERROR: Could not download dataset.")
        print("Details:", error)
        print("\nCheck your internet connection and ucimlrepo installation.")
        raise


# ---------------------------------------------------------
# 3. BASIC DATASET INFORMATION
# ---------------------------------------------------------

def basic_information(df):

    print("\n[2/10] Basic Dataset Information")

    print("\nFirst five rows:")
    print(df.head())

    print("\nLast five rows:")
    print(df.tail())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nDataset information:")
    df.info()

    print("\nStatistical description:")
    print(df.describe(include="all"))

    with open(
        RESULTS_DIR / "basic_information.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("BASIC DATASET INFORMATION\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"Shape: {df.shape}\n\n")
        file.write("Columns:\n")
        file.write(str(df.columns.tolist()))
        file.write("\n\nData Types:\n")
        file.write(str(df.dtypes))
        file.write("\n\nStatistical Description:\n")
        file.write(str(df.describe(include="all")))

    return


# ---------------------------------------------------------
# 4. DATA QUALITY CHECK
# ---------------------------------------------------------

def data_quality(df):

    print("\n[3/10] Data Quality Analysis")

    # Missing values
    missing = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing Percentage": (
            df.isnull().sum() / len(df) * 100
        ).round(2)
    })

    print("\nMissing values:")
    print(missing)

    missing.to_csv(RESULTS_DIR / "missing_values.csv")

    # Duplicate records
    duplicate_count = df.duplicated().sum()

    print("\nDuplicate rows:", duplicate_count)

    with open(
        RESULTS_DIR / "duplicate_summary.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("DUPLICATE ANALYSIS\n")
        file.write("=" * 40 + "\n")
        file.write(f"Total rows: {len(df)}\n")
        file.write(f"Duplicate rows: {duplicate_count}\n")
        file.write(
            f"Duplicate percentage: "
            f"{duplicate_count / len(df) * 100:.2f}%\n"
        )

    # Missing values visualization
    plt.figure(figsize=(12, 5))

    sns.heatmap(
        df.isnull(),
        cbar=False,
        cmap="magma",
        yticklabels=False
    )

    plt.title("Missing Values Overview")
    plt.xlabel("Features")
    plt.ylabel("Records")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "missing_values_heatmap.png",
        dpi=300
    )

    plt.close()

    return missing, duplicate_count


# ---------------------------------------------------------
# 5. TARGET COLUMN ANALYSIS
# ---------------------------------------------------------

def target_analysis(df):

    print("\n[4/10] Target Variable Analysis")

    # UCI ILPD target is Selector
    target_col = next(
        (col for col in df.columns if col.lower() == "selector"),
        None
    )

    if target_col is None:
        raise ValueError(
            "Target column 'Selector' was not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    print("\nTarget column:", target_col)

    print("\nTarget value counts:")
    print(df[target_col].value_counts(dropna=False))

    print("\nTarget percentages:")
    print(
        df[target_col]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
    )

    target_counts = (
        df[target_col]
        .value_counts(dropna=False)
        .rename_axis("Class")
        .reset_index(name="Count")
    )

    target_counts["Percentage"] = (
        target_counts["Count"] / len(df) * 100
    ).round(2)

    target_counts.to_csv(
        RESULTS_DIR / "target_distribution.csv",
        index=False
    )

    # Plot target distribution
    plt.figure(figsize=(8, 5))

    ax = sns.countplot(
        data=df,
        x=target_col,
        hue=target_col,
        legend=False
    )

    for container in ax.containers:
        ax.bar_label(container)

    plt.title("Liver Disease Target Distribution")
    plt.xlabel("Selector Class")
    plt.ylabel("Number of Patients")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "target_distribution.png",
        dpi=300
    )

    plt.close()

    return target_col


# ---------------------------------------------------------
# 6. NUMERICAL FEATURE ANALYSIS
# ---------------------------------------------------------

def numerical_analysis(df, target_col):

    print("\n[5/10] Numerical Feature Analysis")

    numerical_df = df.select_dtypes(include=np.number)

    # Remove target from numerical features
    numerical_features = [
        col for col in numerical_df.columns
        if col != target_col
    ]

    numerical_df = df[numerical_features]

    print("\nNumerical features:")
    print(numerical_features)

    # Statistical summary
    summary = numerical_df.describe().T

    summary["median"] = numerical_df.median()
    summary["skewness"] = numerical_df.skew()
    summary["kurtosis"] = numerical_df.kurtosis()

    print("\nNumerical summary:")
    print(summary)

    summary.to_csv(
        RESULTS_DIR / "numerical_summary.csv"
    )

    # Histograms
    numerical_df.hist(
        figsize=(16, 12),
        bins=25,
        edgecolor="black"
    )

    plt.suptitle(
        "Distribution of Numerical Features",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "numerical_distributions.png",
        dpi=300
    )

    plt.close()

    # Boxplots
    for column in numerical_features:

        plt.figure(figsize=(8, 4))

        sns.boxplot(
            x=df[column],
            color="skyblue"
        )

        plt.title(f"Boxplot of {column}")
        plt.xlabel(column)
        plt.tight_layout()

        filename = (
            column.lower()
            .replace("/", "_")
            .replace(" ", "_")
            + "_boxplot.png"
        )

        plt.savefig(
            RESULTS_DIR / filename,
            dpi=300
        )

        plt.close()

    return numerical_features


# ---------------------------------------------------------
# 7. CATEGORICAL FEATURE ANALYSIS
# ---------------------------------------------------------

def categorical_analysis(df, target_col):

    print("\n[6/10] Categorical Feature Analysis")

    categorical_features = [
        col for col in df.select_dtypes(
            include=["object", "category"]
        ).columns
        if col != target_col
    ]

    print("\nCategorical features:")
    print(categorical_features)

    summary_list = []

    for column in categorical_features:

        print(f"\nValue counts for {column}:")
        print(df[column].value_counts(dropna=False))

        counts = (
            df[column]
            .value_counts(dropna=False)
            .rename_axis("Category")
            .reset_index(name="Count")
        )

        counts.insert(0, "Feature", column)
        summary_list.append(counts)

        plt.figure(figsize=(7, 4))

        sns.countplot(
            data=df,
            x=column,
            hue=column,
            legend=False
        )

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Number of Patients")
        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / f"{column.lower()}_distribution.png",
            dpi=300
        )

        plt.close()

    if summary_list:

        categorical_summary = pd.concat(
            summary_list,
            ignore_index=True
        )

        categorical_summary.to_csv(
            RESULTS_DIR / "categorical_summary.csv",
            index=False
        )

    else:

        print("No categorical features found.")

    return categorical_features


# ---------------------------------------------------------
# 8. CORRELATION ANALYSIS
# ---------------------------------------------------------

def correlation_analysis(df):

    print("\n[7/10] Correlation Analysis")

    numeric_df = df.select_dtypes(include=np.number)

    correlation = numeric_df.corr()

    print("\nCorrelation matrix:")
    print(correlation)

    correlation.to_csv(
        RESULTS_DIR / "correlation_matrix.csv"
    )

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "correlation_heatmap.png",
        dpi=300
    )

    plt.close()

    return correlation


# ---------------------------------------------------------
# 9. FEATURE RELATIONSHIP WITH TARGET
# ---------------------------------------------------------

def feature_target_analysis(df, target_col, numerical_features):

    print("\n[8/10] Feature vs Target Analysis")

    # Compare numerical feature distributions by target class
    for column in numerical_features:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x=target_col,
            y=column,
            hue=target_col,
            legend=False
        )

        plt.title(f"{column} vs Liver Disease Class")
        plt.xlabel("Selector Class")
        plt.ylabel(column)
        plt.tight_layout()

# Convert feature name into a Windows-safe filename
    safe_name = (
        column.lower()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(" ", "")
    )

    output_file = RESULTS_DIR / f"{safe_name}_vs_target.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Grouped mean values
    grouped = df.groupby(target_col)[
        numerical_features
    ].mean()

    print("\nAverage numerical values by target class:")
    print(grouped)

    grouped.to_csv(
        RESULTS_DIR / "feature_means_by_target.csv"
    )

    # Gender vs target relationship, if Gender exists
    gender_col = next(
        (
            col for col in df.columns
            if col.lower() == "gender"
        ),
        None
    )

    if gender_col:

        gender_target = pd.crosstab(
            df[gender_col],
            df[target_col]
        )

        print("\nGender vs Target:")
        print(gender_target)

        gender_target.to_csv(
            RESULTS_DIR / "gender_vs_target.csv"
        )

        plt.figure(figsize=(8, 5))

        sns.countplot(
            data=df,
            x=gender_col,
            hue=target_col
        )

        plt.title("Gender vs Liver Disease Class")
        plt.xlabel("Gender")
        plt.ylabel("Number of Patients")
        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / "gender_vs_target.png",
            dpi=300
        )

        plt.close()


# ---------------------------------------------------------
# 10. OUTLIER SUMMARY
# ---------------------------------------------------------

def outlier_analysis(df, numerical_features):

    print("\n[9/10] Outlier Analysis using IQR")

    outlier_results = []

    for column in numerical_features:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[
            (df[column] < lower_bound)
            | (df[column] > upper_bound)
        ]

        count = len(outliers)

        percentage = count / len(df) * 100

        outlier_results.append({
            "Feature": column,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outlier Count": count,
            "Outlier Percentage": round(percentage, 2)
        })

        print(
            f"{column}: {count} outliers "
            f"({percentage:.2f}%)"
        )

    outlier_df = pd.DataFrame(outlier_results)

    outlier_df.to_csv(
        RESULTS_DIR / "outlier_summary.csv",
        index=False
    )

    return outlier_df


# ---------------------------------------------------------
# 11. FINAL SUMMARY
# ---------------------------------------------------------

def final_summary(df, target_col, missing, duplicate_count):

    print("\n[10/10] Generating Final EDA Summary")

    summary_path = RESULTS_DIR / "eda_summary.txt"

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("LIVER DISEASE DATASET - EDA SUMMARY\n")
        file.write("=" * 55 + "\n\n")

        file.write(f"Total Records: {df.shape[0]}\n")
        file.write(f"Total Columns: {df.shape[1]}\n")
        file.write(f"Target Column: {target_col}\n\n")

        file.write("Missing Values:\n")
        file.write(str(missing))
        file.write("\n\n")

        file.write(f"Duplicate Rows: {duplicate_count}\n\n")

        file.write("Target Distribution:\n")
        file.write(str(df[target_col].value_counts()))
        file.write("\n\n")

        file.write("Data Types:\n")
        file.write(str(df.dtypes))
        file.write("\n\n")

        file.write("Numerical Description:\n")
        file.write(str(df.describe()))
        file.write("\n")

    print("\nEDA completed successfully!")

    print("\nDataset saved at:")
    print(DATA_DIR)

    print("\nEDA results saved at:")
    print(RESULTS_DIR)


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------

def main():

    # Load data
    df = load_dataset()

    # Basic information
    basic_information(df)

    # Data quality
    missing, duplicate_count = data_quality(df)

    # Target analysis
    target_col = target_analysis(df)

    # Numerical analysis
    numerical_features = numerical_analysis(
        df,
        target_col
    )

    # Categorical analysis
    categorical_analysis(
        df,
        target_col
    )

    # Correlation analysis
    correlation_analysis(df)

    # Feature vs target
    feature_target_analysis(
        df,
        target_col,
        numerical_features
    )

    # Outlier analysis
    outlier_analysis(
        df,
        numerical_features
    )

    # Final summary
    final_summary(
        df,
        target_col,
        missing,
        duplicate_count
    )


if __name__ == "__main__":
    main()
