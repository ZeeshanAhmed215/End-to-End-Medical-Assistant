import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="AI Medical Triage Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-level professional UI styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer-box {
        padding: 1rem;
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        color: #92400E;
        font-weight: 500;
        border-radius: 4px;
        margin-bottom: 1.5rem;
    }
    .result-card-high {
        padding: 1.5rem;
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        border-radius: 8px;
        color: #991B1B;
        text-align: center;
    }
    .result-card-low {
        padding: 1.5rem;
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 8px;
        color: #065F46;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODELS SAFELY
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


def load_models(model_dir: Path):
    models = {}
    try:
        models["cancer"] = load_model(model_dir / "cancer_random_forest.pkl")
    except Exception:
        models["cancer"] = None
        
    try:
        models["diabetes"] = load_model(model_dir / "diabetes_random_forest.pkl")
    except Exception:
        models["diabetes"] = None
        
    try:
        models["heart"] = load_model(model_dir / "heart_random_forest.pkl")
    except Exception:
        models["heart"] = None
        
    try:
        models["general"] = load_model(model_dir / "general_model.pkl")
    except Exception:
        models["general"] = None

    return models

models = load_models(MODEL_DIR)

# ==========================================
# HEADER & DISCLAIMER
# ==========================================
st.markdown('<div class="main-header">🩺 AI Medical Triage Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Machine Learning Diagnostic Decision Support System</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>CRITICAL MEDICAL DISCLAIMER:</strong> This application is a decision-support tool powered by machine learning models and is intended for informational and educational purposes only. 
    It <strong>does not</strong> provide formal clinical diagnoses. Always consult a qualified specialist (e.g., Cardiologist, Oncologist, Endocrinologist) for professional medical advice, testing, and treatment plans.
</div>
""", unsafe_allow_html=True)

# ==========================================
# MAIN TABS INTERFACE
# ==========================================
tab_cancer, tab_diabetes, tab_heart, tab_liver, tab_general = st.tabs([
    "🎗️ Cancer Risk Triage",
    "🩸 Diabetes Risk Triage",
    "❤️ Heart Disease Risk Triage",
    "🫀 Liver Disease Risk Triage",
    "🩺 General Checkup"
])


# ==========================================
# TAB 1: CANCER MODEL
# ==========================================
with tab_cancer:
    st.subheader("Breast Cancer Risk Assessment")
    st.write("Provide the morphological characteristics to evaluate malignancy risk.")
    
    if models["cancer"] is None:
        st.error("⚠️ Model file `cancer_random_forest.pkl` not found. Please train and save the model first.")
    else:
        with st.form("cancer_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                mean_radius = st.number_input("Mean Radius", value=14.0, format="%.2f")
                mean_texture = st.number_input("Mean Texture", value=20.0, format="%.2f")
                mean_perimeter = st.number_input("Mean Perimeter", value=90.0, format="%.2f")
            with col2:
                mean_area = st.number_input("Mean Area", value=600.0, format="%.2f")
                mean_smoothness = st.number_input("Mean Smoothness", value=0.1, format="%.4f")
                worst_radius = st.number_input("Worst Radius", value=16.0, format="%.2f")
            with col3:
                worst_texture = st.number_input("Worst Texture", value=25.0, format="%.2f")
                worst_perimeter = st.number_input("Worst Perimeter", value=100.0, format="%.2f")
                worst_area = st.number_input("Worst Area", value=700.0, format="%.2f")
            
            submit_cancer = st.form_submit_button("Analyze Cancer Risk")
            
            if submit_cancer:
                # Construct feature array matching training schema (approximate mapping based on standard dataset features)
                # Ensure feature count matches your saved model expectation
                input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, 
                                        mean_smoothness, worst_radius, worst_texture, worst_perimeter, worst_area]])
                
                # Handling feature dimension padding if model expects full set
                expected_features = models["cancer"].n_features_in_
                if input_data.shape[1] < expected_features:
                    input_data = np.pad(input_data, ((0, 0), (0, expected_features - input_data.shape[1])), 'constant')
                
                prob = models["cancer"].predict_proba(input_data)[0][1]
                pred = models["cancer"].predict(input_data)[0]
                
                st.markdown("---")
                st.markdown("### Evaluation Results")
                risk_percentage = prob * 100
                
                if risk_percentage > 50:
                    st.markdown(f"""
                        <div class="result-card-high">
                            <h3>High Risk Detected</h3>
                            <p>Malignancy Probability: <strong>{risk_percentage:.2f}%</strong></p>
                            <p><em>Recommendation: Immediate consultation with an Oncologist for diagnostic biopsy is strongly advised.</em></p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-card-low">
                            <h3>Low Risk Detected</h3>
                            <p>Malignancy Probability: <strong>{risk_percentage:.2f}%</strong></p>
                            <p><em>Recommendation: Routine annual screenings recommended. Consult your primary physician if symptoms change.</em></p>
                        </div>
                    """, unsafe_allow_html=True)

# ==========================================
# TAB 2: DIABETES MODEL
# ==========================================
with tab_diabetes:
    st.subheader("Diabetes Risk Assessment")
    st.write("Enter metabolic and clinical indicators for diabetes screening.")
    
    if models["diabetes"] is None:
        st.error("⚠️ Model file `diabetes_random_forest.pkl` not found. Please train and save the model first.")
    else:
        with st.form("diabetes_form"):
            col1, col2 = st.columns(2)
            with col1:
                pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
                glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
                blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=70)
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
            with col2:
                insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
                bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
                age = st.number_input("Age (years)", min_value=1, max_value=120, value=35)
                
            submit_diabetes = st.form_submit_button("Analyze Diabetes Risk")
            
            if submit_diabetes:
                input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
                
                expected_features = models["diabetes"].n_features_in_
                if input_data.shape[1] < expected_features:
                    input_data = np.pad(input_data, ((0, 0), (0, expected_features - input_data.shape[1])), 'constant')
                    
                prob = models["diabetes"].predict_proba(input_data)[0][1]
                risk_percentage = prob * 100
                
                st.markdown("---")
                st.markdown("### Evaluation Results")
                
                if risk_percentage > 50:
                    st.markdown(f"""
                        <div class="result-card-high">
                            <h3>High Risk Detected</h3>
                            <p>Diabetes Probability: <strong>{risk_percentage:.2f}%</strong></p>
                            <p><em>Recommendation: Consult an Endocrinologist for an HbA1c or Fasting Plasma Glucose test.</em></p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-card-low">
                            <h3>Low Risk Detected</h3>
                            <p>Diabetes Probability: <strong>{risk_percentage:.2f}%</strong></p>
                            <p><em>Recommendation: Maintain a balanced diet, regular exercise, and periodic metabolic checkups.</em></p>
                        </div>
                    """, unsafe_allow_html=True)

# ==========================================
# TAB 3: HEART DISEASE MODEL
# ==========================================
with tab_heart:
    st.subheader("Heart Disease Risk Assessment")
    st.write("Enter cardiovascular parameters to check cardiac risk status.")
    
    if models["heart"] is None:
        st.error("⚠️ Model file `heart_random_forest.pkl` not found. Please train and save the model first.")
    else:
        with st.form("heart_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age_h = st.number_input("Age", min_value=1, max_value=120, value=50, key="h_age")
                sex_h = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                cp_h = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
                trestbps_h = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=250, value=130)
            with col2:
                chol_h = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
                fbs_h = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                restecg_h = st.selectbox("Resting ECG Results", options=[0, 1, 2])
                thalach_h = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
            with col3:
                exang_h = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                oldpeak_h = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
                slope_h = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2])
                ca_h = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
                
            submit_heart = st.form_submit_button("Analyze Heart Risk")
            
            if submit_heart:
                # Basic representation matching feature structure of clean heart dataset
                input_data = np.array([[age_h, sex_h, cp_h, trestbps_h, chol_h, fbs_h, restecg_h, thalach_h, exang_h, oldpeak_h, slope_h, ca_h]])
                
                expected_features = models["heart"].n_features_in_
                if input_data.shape[1] < expected_features:
                    input_data = np.pad(input_data, ((0, 0), (0, expected_features - input_data.shape[1])), 'constant')
                elif input_data.shape[1] > expected_features:
                    input_data = input_data[:, :expected_features]
                    
                prob = models["heart"].predict_proba(input_data)[0][1]
                risk_percentage = prob * 100
                
                st.markdown("---")
                st.markdown("### Evaluation Results")
                
                if risk_percentage > 50:
                    st.markdown(f"""
                        <div class="result-card-high">
                            <h3>High Cardiovascular Risk Detected</h3>
                            <p>Heart Disease Probability: <strong>{risk_percentage:.2f}%</strong></p>
                            <p><em>Recommendation: Urgent consultation with a <strong>Cardiologist</strong> for an ECG, stress test, or cardiac evaluation is strongly recommended.</em></p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-card-low">
                            <h3>Low Cardiovascular Risk Detected</h3>
                            <p>Heart Disease Probability: <strong>{risk_percentage:.2f}%</strong></p>
                            <p><em>Recommendation: Maintain heart-healthy habits, regular cardiovascular workouts, and annual checkups.</em></p>
                        </div>
                    """, unsafe_allow_html=True)


# ==========================================
# TAB 4: LIVER MODEL
# ==========================================
with tab_liver:
    st.subheader("Liver Disease Risk Assessment")
    st.write("Enter the liver function measurements used by the trained model.")

    liver_model_path = MODEL_DIR / "liver_model.pkl"
    if not liver_model_path.is_file():
        st.error(
            f"⚠️ Liver model file not found at: {liver_model_path}"
        )
    else:
        try:
            liver_model = load_model(liver_model_path)
        except Exception as exc:
            st.error(
                f"⚠️ Failed to load liver model at {liver_model_path}: {exc}"
            )
        else:
            with st.form("liver_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    age_l = st.number_input("Age", min_value=1, max_value=120, value=45, key="l_age")
                    gender_l = st.selectbox("Gender", options=["Female", "Male"], key="l_gender")
                    tb_l = st.number_input("Total Bilirubin (TB)", min_value=0.0, value=1.0, format="%.2f")
                    db_l = st.number_input("Direct Bilirubin (DB)", min_value=0.0, value=0.3, format="%.2f")
                with col2:
                    alkphos_l = st.number_input("Alkaline Phosphatase", min_value=0.0, value=200.0, format="%.1f")
                    sgpt_l = st.number_input("Alanine Aminotransferase (SGPT)", min_value=0.0, value=30.0, format="%.1f")
                    sgot_l = st.number_input("Aspartate Aminotransferase (SGOT)", min_value=0.0, value=35.0, format="%.1f")
                with col3:
                    tp_l = st.number_input("Total Protein (TP)", min_value=0.0, value=6.5, format="%.2f")
                    alb_l = st.number_input("Albumin (ALB)", min_value=0.0, value=3.2, format="%.2f")
                    ag_l = st.number_input("Albumin/Globulin Ratio", min_value=0.0, value=0.9, format="%.2f")

                submit_liver = st.form_submit_button("Analyze Liver Risk")

                if submit_liver:
                    liver_input = pd.DataFrame([{
                        "Age": age_l,
                        "Gender": gender_l,
                        "TB": tb_l,
                        "DB": db_l,
                        "Alkphos": alkphos_l,
                        "Sgpt": sgpt_l,
                        "Sgot": sgot_l,
                        "TP": tp_l,
                        "ALB": alb_l,
                        "A/G Ratio": ag_l,
                    }])
                    expected_columns = list(liver_model.feature_names_in_)
                    if list(liver_input.columns) != expected_columns:
                        st.error(
                            "Liver model feature mismatch. "
                            f"Expected {expected_columns}, got {list(liver_input.columns)}."
                        )
                    else:
                        probability = liver_model.predict_proba(liver_input)[0, 1]
                        risk_percentage = probability * 100

                        st.markdown("---")
                        st.markdown("### Evaluation Results")
                        if risk_percentage > 50:
                            st.markdown(f"""
                                <div class="result-card-high">
                                    <h3>Higher Liver Disease Risk Detected</h3>
                                    <p>Liver Disease Probability: <strong>{risk_percentage:.2f}%</strong></p>
                                    <p><em>Please consult a qualified physician for clinical evaluation.</em></p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class="result-card-low">
                                    <h3>Lower Liver Disease Risk Detected</h3>
                                    <p>Liver Disease Probability: <strong>{risk_percentage:.2f}%</strong></p>
                                    <p><em>This screening result is not a diagnosis; consult a qualified physician.</em></p>
                                </div>
                            """, unsafe_allow_html=True)





# ==========================================
# TAB 5: GENERAL MEDICAL CHECKUP
# ==========================================

with tab_general:

    st.subheader("🩺 General Medical Checkup")

    st.write(
        "Select the symptoms you are experiencing "
        "to get AI-based disease predictions."
    )

    if models["general"] is None:

        st.error(
            "General disease model not found. "
            "Please train general_model.pkl first."
        )

    else:

        try:

            # Load trained model artifacts
            general_model, feature_names, general_encoder = models["general"]

            st.markdown("---")

            st.markdown("### Patient Information")

            col1, col2 = st.columns(2)

            with col1:
                patient_age = st.number_input(
                    "Age (Years)",
                    min_value=1,
                    max_value=120,
                    value=25
                )

            with col2:
                patient_gender = st.selectbox(
                    "Gender",
                    ["Male", "Female", "Other"]
                )

            st.markdown("---")

            st.markdown("### Select Your Symptoms")

            st.info(
                "Search and select all symptoms "
                "you are currently experiencing."
            )

            # Convert snake_case into readable names
            symptom_mapping = {
                symptom.replace("_", " ").title(): symptom
                for symptom in feature_names
            }

            # Display readable symptom names
            selected_symptoms = st.multiselect(
                "Search Symptoms",
                options=list(symptom_mapping.keys()),
                placeholder="e.g. Nodal Skin Eruptions",
                help="You can select multiple symptoms."
            )

            st.write(
                f"Selected Symptoms: {len(selected_symptoms)}"
            )

            # Show selected symptoms
            if selected_symptoms:

                st.markdown("#### Your Selected Symptoms")

                for symptom in selected_symptoms:
                    st.write(f"✅ {symptom}")

            st.markdown("---")

            # Analyze button
            analyze_general = st.button(
                "🔍 Analyze General Health",
                type="primary",
                use_container_width=True
            )

            if analyze_general:

                if not selected_symptoms:

                    st.warning(
                        "Please select at least one symptom."
                    )

                else:

                    # Create input using original training columns
                    input_df = pd.DataFrame(
                        0,
                        index=[0],
                        columns=feature_names
                    )

                    # Set selected symptoms to 1
                    for symptom in selected_symptoms:

                        original_name = symptom_mapping[symptom]

                        input_df.loc[0, original_name] = 1

                    # Generate prediction probabilities
                    probabilities = general_model.predict_proba(
                        input_df
                    )[0]

                    # Sort predictions
                    sorted_indices = np.argsort(probabilities)[::-1]

                    # Get top prediction
                    best_index = sorted_indices[0]

                    predicted_class = general_model.classes_[best_index]

                    predicted_disease = general_encoder.inverse_transform(
                        [int(predicted_class)]
                    )[0]

                    prediction_score = probabilities[best_index] * 100

                    st.markdown("---")

                    st.markdown("## 🩺 Prediction Results")

                    st.success(
                        f"Top Predicted Disease: {predicted_disease}"
                    )

                    # Display prediction score
                    st.metric(
                        "Model Prediction Score",
                        f"{prediction_score:.2f}%"
                    )

                    st.progress(
                        float(probabilities[best_index])
                    )

                    st.caption(
                        "This percentage represents the model's "
                        "prediction score, not a clinically validated "
                        "probability of disease."
                    )

                    # Top 5 predictions
                    st.markdown("### Top 5 Possible Predictions")

                    results = []

                    for idx in sorted_indices[:5]:

                        disease_id = general_model.classes_[idx]

                        disease_name = general_encoder.inverse_transform(
                            [int(disease_id)]
                        )[0]

                        score = probabilities[idx] * 100

                        results.append({
                            "Disease": disease_name,
                            "Prediction Score (%)": round(score, 2)
                        })

                    results_df = pd.DataFrame(results)

                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Visualization
                    st.markdown("### Prediction Comparison")

                    chart_df = results_df.set_index("Disease")

                    st.bar_chart(
                        chart_df["Prediction Score (%)"]
                    )

                    st.warning(
                        "⚠️ This is an experimental AI prediction, "
                        "not a medical diagnosis. Consult a qualified "
                        "healthcare professional for proper evaluation."
                    )

        except Exception as e:

            st.error(f"General Checkup Error: {str(e)}")







# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6B7280;'>AI Medical Triage System | Built with Streamlit & Scikit-Learn</p>", unsafe_allow_html=True)