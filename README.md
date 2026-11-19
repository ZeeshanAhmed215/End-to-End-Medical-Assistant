# 🩺 End-to-End-Medical-Assistant (MedSight)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.10%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit-learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A multi-disease machine learning assistant featuring automated data preprocessing, ensemble classification, and explainability across 5 clinical datasets.
---

## 🚀 Overview

**MedSight** is a comprehensive clinical decision-support intelligence platform. The architecture features dedicated, separately trained machine learning pipelines for distinct medical conditions—including breast cancer, cardiovascular disease, diabetes, liver disorders, and multi-symptom general diagnoses. All models are unified within a responsive, dark-themed **Streamlit** web application for real-time risk screening.

**🌐 Live Demo**  
Try the application directly in your browser without any local setup:  
👉 [Launch MedSight Live App](https://zeeshanahmed215-end-to-end-medical-assistant-app-fkeciy.streamlit.app/)

---
## 📈 Model Performance & Evaluation

Each specialized model is trained on distinct clinical datasets and evaluated on stratified held-out test sets. Below is a summary of their performance metrics:

| Model | Target Domain | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | Test Samples |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **General Diagnosis** | Symptom-based multi-disease | **98.33%** | 96.34% | 97.56% | 99.96% | 96.75% | 1.0000 | 60 |
| **Cancer Detection** | Breast cancer morphology | **97.08%** | 92.31% | 100.00% | 95.51% | 96.00% | 0.9822 | 137 |
| **Heart Disease** | Cardiovascular screening | **88.04%** | 88.46% | 90.20% | 85.37% | 89.32% | 0.9366 | 184 |
| **Diabetes** | Metabolic profile | **78.25%** | 63.54% | 65.59% | 83.72% | 64.55% | 0.8528 | 308 |
| **Liver Disease** | Hepatology & ILPD records | **70.09%** | 73.53% | 90.36% | 20.59% | 81.08% | 0.7548 | 117 |

*To regenerate classification reports, evaluation metrics, and multi-panel confusion matrices locally, run:*
```bash
python src/evaluate.py
```

---

## 🛠️ Tech Stack & Dependencies

* **Language**: Python 3.8+
* **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `imbalanced-learn`
* **Model Explainability**: `shap`
* **Data Manipulation**: `pandas`, `numpy`
* **Visualization**: `matplotlib`, `seaborn`
* **Web UI Framework**: `streamlit`
* **Model Serialization**: `joblib`

---

## 📂 Project Structure

```text
End-to-End-Medical-Assistant/
│
├── datasets/                 # Raw and processed clinical CSV datasets
│   ├── cancer.csv            # Breast cancer dataset
│   ├── clean_heart.csv       # Cardiovascular dataset
│   ├── diabetes_testing.csv  # Diabetes evaluation dataset
│   ├── general.csv           # Multi-symptom condition dataset
│   └── ilpd_combined.csv     # Indian Liver Patient Dataset (ILPD)
│
├── models/                   # Serialized machine learning artifacts (.pkl)
│   ├── cancer_random_forest.pkl
│   ├── diabetes_random_forest.pkl
│   ├── heart_random_forest.pkl
│   ├── general_model.pkl
│   └── liver_model.pkl
│
├── src/                      # Source scripts
│   └── evaluate.py           # Evaluation pipeline & metrics generator
│
├── app.py                    # Main Streamlit web application
├── requirements.txt          # Python package dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ZeeshanAhmed215/End-to-End-Medical-Assistant.git
cd End-to-End-Medical-Assistant
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## ⚠️ Clinical Disclaimer

> **Important Note**: MedSight is an experimental clinical decision-support and educational tool. It **does not** provide formal diagnoses, medical treatment, or replace professional healthcare advice from a licensed medical practitioner. Always consult a qualified physician regarding any personal health concerns.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.