"""Streamlit interface for the End-to-End Medical Assistant."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "heart": "heart_random_forest.pkl",
    "diabetes": "diabetes_random_forest.pkl",
    "cancer": "cancer_random_forest.pkl",
    "general": "general_model.pkl",
    "liver": "liver_model.pkl",
}

st.set_page_config(
    page_title="MedSight | Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #e0e0e0;
        --navy-light: #b8c2c8;
        --teal: #00adb5;
        --teal-light: #17383b;
        --emerald: #22c55e;
        --ink: #e0e0e0;
        --muted: #a7b0b5;
        --line: #333333;
        --canvas: #121212;
        --white: #1e1e1e;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #171717;
        border-right: 1px solid #2d2d2d;
    }

    [data-testid="stSidebar"] * {
        color: #e6f0f7;
    }

    [data-testid="stSidebar"] .stRadio label {
        border-radius: 8px;
        padding: 0.35rem 0.5rem;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255, 255, 255, 0.1);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .brand-mark {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }

    .brand-subtitle {
        color: #9fb6c8;
        font-size: 0.78rem;
        margin-bottom: 1.8rem;
    }

    .eyebrow {
        color: var(--teal);
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    .page-title {
        color: var(--navy);
        font-size: 2.45rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.08;
        margin: 0;
    }

    .page-subtitle {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.6;
        margin: 0.65rem 0 1.5rem;
        max-width: 760px;
    }

    .hero-card,
    .section-card,
    .module-card,
    .result-card {
        background: var(--white);
        border: 1px solid var(--line);
        border-radius: 14px;
        box-shadow: 0 5px 18px rgba(16, 42, 67, 0.05);
    }

    .hero-card {
        background: linear-gradient(135deg, #17383b 0%, #1e1e1e 68%);
        border-color: #245c60;
        padding: 1.5rem 1.7rem;
        margin: 0.5rem 0 1.2rem;
    }

    .hero-card h3,
    .module-card h3 {
        color: var(--navy);
        margin: 0 0 0.45rem;
    }

    .hero-card p,
    .module-card p {
        color: var(--muted);
        line-height: 1.55;
        margin: 0;
    }

    .section-card {
        padding: 1.25rem 1.35rem;
        margin: 1rem 0;
    }

    .section-heading {
        color: var(--navy);
        font-size: 1.1rem;
        font-weight: 750;
        margin: 0 0 0.2rem;
    }

    .section-caption {
        color: var(--muted);
        font-size: 0.87rem;
        margin: 0 0 0.8rem;
    }

    .module-card {
        min-height: 145px;
        padding: 1.15rem;
        margin-bottom: 0.8rem;
    }

    .module-icon {
        font-size: 1.7rem;
        margin-bottom: 0.4rem;
    }

    .module-label {
        color: var(--teal);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    .result-card {
        border-top: 4px solid var(--teal);
        padding: 1.35rem;
        margin-top: 1rem;
    }

    .result-card.high {
        border-top-color: #dc2626;
    }

    .result-card.low {
        border-top-color: var(--emerald);
    }

    .result-card h3 {
        color: var(--navy);
        margin: 0 0 0.25rem;
    }

    .result-card p {
        color: var(--muted);
        margin: 0;
    }

    .recommendation {
        border-radius: 9px;
        font-size: 0.9rem;
        line-height: 1.45;
        margin-top: 0.8rem;
        padding: 0.75rem 0.9rem;
    }

    .recommendation.high {
        background: #3a1d20;
        border: 1px solid #7f3038;
        color: #fda4af;
    }

    .recommendation.low {
        background: #163326;
        border: 1px solid #28734a;
        color: #86efac;
    }

    .status-pill {
        background: var(--teal-light);
        border-radius: 999px;
        color: var(--teal);
        display: inline-block;
        font-size: 0.76rem;
        font-weight: 750;
        padding: 0.35rem 0.75rem;
    }

    .disclaimer {
        background: #332b19;
        border: 1px solid #735d2a;
        border-left: 4px solid #d99a2b;
        border-radius: 10px;
        color: #f3cf83;
        font-size: 0.86rem;
        line-height: 1.5;
        padding: 0.85rem 1rem;
        margin: 1rem 0 1.5rem;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: var(--teal);
        border: 0;
        border-radius: 8px;
        color: #ffffff;
        font-weight: 750;
        min-height: 2.65rem;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #115e59;
        color: #ffffff;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path: Path):
    """Load a model from a resolved path and cache successful loads."""
    return joblib.load(model_path)


def load_models(model_dir: Path):
    models = {}
    errors = {}
    for model_name, filename in MODEL_FILES.items():
        path = model_dir / filename
        try:
            models[model_name] = load_model(path)
        except Exception as exc:
            models[model_name] = None
            errors[model_name] = f"{path}: {exc}"
    return models, errors


models, model_errors = load_models(MODEL_DIR)


def page_header(eyebrow, title, description):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="page-title">{title}</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="page-subtitle">{description}</p>',
        unsafe_allow_html=True,
    )


def section_heading(title, caption):
    st.markdown(
        f"""
        <div class="section-heading">{title}</div>
        <div class="section-caption">{caption}</div>
        """,
        unsafe_allow_html=True,
    )


def model_gate(model_name):
    if models.get(model_name) is not None:
        return models[model_name]
    st.error(
        f"Unable to load the {model_name} model. "
        f"Expected artifact: {MODEL_DIR / MODEL_FILES[model_name]}. "
        f"Details: {model_errors.get(model_name, 'Unknown loading error')}"
    )
    return None


def show_range_warnings(values):
    warnings = []
    for label, value, lower, upper in values:
        if value < lower or value > upper:
            warnings.append(f"{label}: {value:g} is outside the typical range.")
    if warnings:
        st.warning(
            "Please review these values before continuing: "
            + " ".join(warnings)
        )


def make_model_input(model, values):
    expected_columns = list(model.feature_names_in_)
    if len(values) != len(expected_columns):
        raise ValueError(
            f"Expected {len(expected_columns)} model features, received {len(values)}."
        )
    return pd.DataFrame([values], columns=expected_columns)


def render_risk_assessment(probability: float):
    """Render a consistent probability, progress, and triage assessment."""
    probability = float(np.clip(probability, 0.0, 1.0))
    percentage = probability * 100

    if probability < 0.30:
        category = "Low risk"
        guidance = "Routine monitoring is recommended. Continue healthy habits and scheduled care."
        message = "Low-risk signal"
        render_message = st.success
    elif probability < 0.70:
        category = "Moderate risk"
        guidance = "Follow-up with a qualified healthcare professional is recommended."
        message = "Moderate-risk signal"
        render_message = st.warning
    else:
        category = "High risk"
        guidance = "Please seek urgent medical consultation to review this result."
        message = "High-risk signal"
        render_message = st.error

    with st.container(border=True):
        st.subheader("Risk assessment")
        st.metric("Estimated probability", f"{percentage:.2f}%")
        st.progress(probability)
        st.caption(f"{percentage:.2f}% estimated model score")
        render_message(f"{category}: {message}")
        st.caption(guidance)
        st.caption("This is decision support, not a clinical diagnosis.")


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="brand-mark">🩺 MedSight</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="brand-subtitle">Clinical decision support workspace</div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### Workspace")
        route = st.radio(
            "Navigate",
            [
                "Home",
                "Heart Disease",
                "Diabetes",
                "Cancer",
                "Liver Disease",
                "General Checkup",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("#### System status")
        available = sum(model is not None for model in models.values())
        st.markdown(
            f'<span class="status-pill">{available}/5 models online</span>',
            unsafe_allow_html=True,
        )
        st.caption("Artifacts are loaded from the application directory.")
        return route


def render_home():
    page_header(
        "Clinical intelligence platform",
        "A clearer view of patient risk.",
        "MedSight brings five specialized machine-learning assessments into one calm, "
        "structured workspace for informed clinical conversations.",
    )
    st.markdown(
        """
        <div class="hero-card">
            <h3>Decision support, designed for focus</h3>
            <p>Choose an assessment from the sidebar, enter the available clinical
            indicators, and review a transparent model score. Results are educational
            signals and should always be interpreted by a qualified professional.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    stat_one, stat_two, stat_three = st.columns(3)
    stat_one.metric("Assessment modules", "5")
    stat_two.metric("Model artifacts", f"{sum(model is not None for model in models.values())}/5")
    stat_three.metric("Path safety", "Enabled")

    st.markdown("### Choose an assessment")
    modules = [
        ("❤️", "Heart Disease", "Cardiovascular indicators and cardiac risk screening."),
        ("🩸", "Diabetes", "Metabolic measurements and diabetes risk screening."),
        ("🎗️", "Cancer", "Morphological characteristics for malignancy risk screening."),
        ("🫀", "Liver Disease", "Liver function measurements and disease risk screening."),
        ("🩺", "General Checkup", "Symptom-based multi-disease prediction."),
    ]
    for start in range(0, len(modules), 3):
        columns = st.columns(3)
        for column, (icon, name, description) in zip(columns, modules[start:start + 3]):
            with column:
                st.markdown(
                    f"""
                    <div class="module-card">
                        <div class="module-icon">{icon}</div>
                        <div class="module-label">{name}</div>
                        <p>{description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    st.markdown(
        """
        <div class="disclaimer">
            <strong>Important:</strong> MedSight is an experimental decision-support
            application. It does not diagnose, treat, or replace advice from a
            licensed healthcare professional.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_heart():
    page_header(
        "Cardiovascular assessment",
        "Heart disease risk",
        "Review common cardiovascular indicators with a structured, model-assisted screening.",
    )
    model = model_gate("heart")
    if model is None:
        return
    with st.form("heart_form"):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_heading("Patient profile", "Enter the measurements available from the current visit.")
        col_one, col_two, col_three = st.columns(3)
        with col_one:
            age = st.number_input("Age", 1, 120, 50, help="Patient age in years.")
            sex = st.selectbox("Sex", [0, 1], format_func=lambda value: "Female" if value == 0 else "Male", help="Encoded sex used by the trained model.")
            chest_pain = st.selectbox("Chest pain type", [0, 1, 2, 3], help="Encoded chest-pain category from the source dataset.")
            resting_bp = st.number_input("Resting blood pressure", 80, 250, 130, help="Resting systolic blood pressure in mmHg.")
        with col_two:
            cholesterol = st.number_input("Cholesterol", 100, 600, 240, help="Serum cholesterol in mg/dL.")
            fasting_bs = st.selectbox("Fasting blood sugar > 120", [0, 1], format_func=lambda value: "No" if value == 0 else "Yes", help="Whether fasting blood sugar exceeds 120 mg/dL.")
            resting_ecg = st.selectbox("Resting ECG", [0, 1, 2], help="Encoded resting electrocardiogram result.")
            max_hr = st.number_input("Maximum heart rate", 60, 220, 150, help="Maximum heart rate achieved during testing.")
        with col_three:
            exercise_angina = st.selectbox("Exercise-induced angina", [0, 1], format_func=lambda value: "No" if value == 0 else "Yes", help="Whether exercise induces angina.")
            oldpeak = st.number_input("ST depression (Oldpeak)", 0.0, 10.0, 1.0, format="%.1f", help="ST depression induced by exercise relative to rest.")
            slope = st.selectbox("ST segment slope", [0, 1, 2], help="Encoded slope of the peak exercise ST segment.")
            major_vessels = st.selectbox("Major vessels (0–3)", [0, 1, 2, 3], help="Number of major vessels observed by fluoroscopy.")
        st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Analyze heart risk", width="stretch")

    if submitted:
        show_range_warnings([
            ("Resting blood pressure", resting_bp, 90, 180),
            ("Cholesterol", cholesterol, 120, 400),
            ("Maximum heart rate", max_hr, 80, 210),
        ])
        input_data = pd.DataFrame([{
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            "Sex_M": sex,
            "ChestPainType_ATA": int(chest_pain == 1),
            "ChestPainType_NAP": int(chest_pain == 2),
            "ChestPainType_TA": int(chest_pain == 3),
            "RestingECG_Normal": int(resting_ecg == 1),
            "RestingECG_ST": int(resting_ecg == 2),
            "ExerciseAngina_Y": exercise_angina,
            "ST_Slope_Flat": int(slope == 1),
            "ST_Slope_Up": int(slope == 2),
        }])
        probability = model.predict_proba(input_data)[0, 1]
        render_risk_assessment(probability)


def render_diabetes():
    page_header(
        "Metabolic assessment",
        "Diabetes risk",
        "Use routine metabolic measurements to review a diabetes screening signal.",
    )
    model = model_gate("diabetes")
    if model is None:
        return
    with st.form("diabetes_form"):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_heading("Metabolic profile", "Use the most recent validated measurements available.")
        col_one, col_two = st.columns(2)
        with col_one:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1, help="Number of pregnancies.")
            glucose = st.number_input("Glucose", 0, 300, 120, help="Plasma glucose concentration in mg/dL.")
            blood_pressure = st.number_input("Blood pressure", 0, 200, 70, help="Diastolic blood pressure in mmHg.")
            skin_thickness = st.number_input("Skin thickness", 0, 100, 20, help="Triceps skin-fold thickness in mm.")
        with col_two:
            insulin = st.number_input("Insulin", 0, 900, 80, help="Two-hour serum insulin in μU/mL.")
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0, format="%.1f", help="Body mass index in kg/m².")
            pedigree = st.number_input("Diabetes pedigree function", 0.0, 3.0, 0.5, format="%.3f", help="Family-history-based diabetes risk score.")
            age = st.number_input("Age", 1, 120, 35, help="Patient age in years.")
        st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Analyze diabetes risk", width="stretch")

    if submitted:
        show_range_warnings([
            ("Glucose", glucose, 50, 200),
            ("Blood pressure", blood_pressure, 40, 130),
            ("BMI", bmi, 15, 50),
        ])
        input_data = make_model_input(model, [
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, pedigree, age,
        ])
        render_risk_assessment(model.predict_proba(input_data)[0, 1])


def render_cancer():
    page_header(
        "Oncology assessment",
        "Breast cancer risk",
        "Review the morphological measurements used by the breast cancer screening model.",
    )
    model = model_gate("cancer")
    if model is None:
        return
    with st.form("cancer_form"):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_heading("Morphology profile", "Enter the measurements from the pathology or imaging record.")
        col_one, col_two, col_three = st.columns(3)
        with col_one:
            mean_radius = st.number_input("Mean radius", 0.0, 100.0, 14.0, format="%.2f", help="Average radius measurement.")
            mean_texture = st.number_input("Mean texture", 0.0, 100.0, 20.0, format="%.2f", help="Average texture measurement.")
            mean_perimeter = st.number_input("Mean perimeter", 0.0, 500.0, 90.0, format="%.2f", help="Average perimeter measurement.")
        with col_two:
            mean_area = st.number_input("Mean area", 0.0, 5000.0, 600.0, format="%.2f", help="Average area measurement.")
            mean_smoothness = st.number_input("Mean smoothness", 0.0, 1.0, 0.1, format="%.4f", help="Average smoothness measurement.")
            worst_radius = st.number_input("Worst radius", 0.0, 100.0, 16.0, format="%.2f", help="Largest radius measurement.")
        with col_three:
            worst_texture = st.number_input("Worst texture", 0.0, 100.0, 25.0, format="%.2f", help="Largest texture measurement.")
            worst_perimeter = st.number_input("Worst perimeter", 0.0, 500.0, 100.0, format="%.2f", help="Largest perimeter measurement.")
            worst_area = st.number_input("Worst area", 0.0, 5000.0, 700.0, format="%.2f", help="Largest area measurement.")
        st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Analyze cancer risk", width="stretch")

    if submitted:
        input_data = make_model_input(model, [
            mean_radius, mean_texture, mean_perimeter, mean_area,
            mean_smoothness, worst_radius, worst_texture, worst_perimeter,
            worst_area,
        ])
        render_risk_assessment(model.predict_proba(input_data)[0, 1])


def render_liver():
    page_header(
        "Hepatology assessment",
        "Liver disease risk",
        "Review liver function measurements with the pipeline trained on the ILPD dataset.",
    )
    model = model_gate("liver")
    if model is None:
        return
    with st.form("liver_form"):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_heading("Liver function profile", "Enter the latest available laboratory measurements.")
        col_one, col_two, col_three = st.columns(3)
        with col_one:
            age = st.number_input("Age", 1, 120, 45, key="l_age", help="Patient age in years.")
            gender = st.selectbox("Gender", ["Female", "Male"], key="l_gender", help="Gender category used by the trained pipeline.")
            tb = st.number_input("Total bilirubin (TB)", 0.0, 100.0, 1.0, format="%.2f", help="Total bilirubin concentration.")
            db = st.number_input("Direct bilirubin (DB)", 0.0, 100.0, 0.3, format="%.2f", help="Direct bilirubin concentration.")
        with col_two:
            alkphos = st.number_input("Alkaline phosphatase", 0.0, 2000.0, 200.0, format="%.1f", help="Alkaline phosphatase enzyme level.")
            sgpt = st.number_input("SGPT / ALT", 0.0, 2000.0, 30.0, format="%.1f", help="Alanine aminotransferase level.")
            sgot = st.number_input("SGOT / AST", 0.0, 2000.0, 35.0, format="%.1f", help="Aspartate aminotransferase level.")
        with col_three:
            total_protein = st.number_input("Total protein (TP)", 0.0, 20.0, 6.5, format="%.2f", help="Total serum protein.")
            albumin = st.number_input("Albumin (ALB)", 0.0, 15.0, 3.2, format="%.2f", help="Serum albumin concentration.")
            ratio = st.number_input("Albumin / globulin ratio", 0.0, 10.0, 0.9, format="%.2f", help="Albumin-to-globulin ratio.")
        st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Analyze liver risk", width="stretch")

    if submitted:
        show_range_warnings([
            ("Total bilirubin", tb, 0.1, 3.0),
            ("Direct bilirubin", db, 0.0, 1.0),
            ("Alkaline phosphatase", alkphos, 40, 400),
            ("SGPT / ALT", sgpt, 5, 200),
            ("SGOT / AST", sgot, 5, 200),
            ("Albumin", albumin, 2.5, 5.5),
        ])
        liver_input = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "TB": tb,
            "DB": db,
            "Alkphos": alkphos,
            "Sgpt": sgpt,
            "Sgot": sgot,
            "TP": total_protein,
            "ALB": albumin,
            "A/G Ratio": ratio,
        }])
        expected_columns = list(model.feature_names_in_)
        if list(liver_input.columns) != expected_columns:
            st.error(
                "The liver input schema does not match the saved model. "
                f"Expected {expected_columns}; received {list(liver_input.columns)}."
            )
            return
        render_risk_assessment(model.predict_proba(liver_input)[0, 1])


def render_general():
    page_header(
        "Symptom intelligence",
        "General checkup",
        "Select the symptoms currently present to generate a ranked, multi-disease screening signal.",
    )
    model_bundle = model_gate("general")
    if model_bundle is None:
        return
    general_model, feature_names, encoder = model_bundle
    with st.form("general_form"):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_heading("Symptom profile", "Select all symptoms that are currently relevant.")
        patient_col, symptom_col = st.columns([1, 2])
        with patient_col:
            st.number_input("Age", 1, 120, 25, key="g_age", help="Patient age in years.")
            st.selectbox("Gender", ["Male", "Female", "Other"], key="g_gender", help="Patient gender.")
        with symptom_col:
            symptom_mapping = {
                symptom.replace("_", " ").title(): symptom
                for symptom in feature_names
            }
            selected_symptoms = st.multiselect(
                "Symptoms",
                options=list(symptom_mapping.keys()),
                help="Search and select all symptoms currently present.",
            )
        st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Generate symptom assessment", width="stretch")

    if submitted:
        if not selected_symptoms:
            st.warning("Select at least one symptom to generate an assessment.")
            return
        input_df = pd.DataFrame(0, index=[0], columns=feature_names)
        for symptom in selected_symptoms:
            input_df.loc[0, symptom_mapping[symptom]] = 1
        probabilities = general_model.predict_proba(input_df)[0]
        sorted_indices = np.argsort(probabilities)[::-1]
        rows = []
        for index in sorted_indices[:5]:
            disease_name = encoder.inverse_transform([int(general_model.classes_[index])])[0]
            rows.append({
                "Disease": disease_name,
                "Prediction score": f"{probabilities[index] * 100:.2f}%",
            })
        best_index = sorted_indices[0]
        best_name = encoder.inverse_transform([int(general_model.classes_[best_index])])[0]
        st.markdown(f"### Top predicted condition: {best_name}")
        st.caption("Ranked from the selected symptom profile.")
        metric_col, chart_col = st.columns([1, 2])
        with metric_col:
            render_risk_assessment(float(probabilities[best_index]))
        with chart_col:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.info("Consult a qualified healthcare professional for examination and diagnosis.")


def main():
    route = render_sidebar()
    if route == "Home":
        render_home()
    elif route == "Heart Disease":
        render_heart()
    elif route == "Diabetes":
        render_diabetes()
    elif route == "Cancer":
        render_cancer()
    elif route == "Liver Disease":
        render_liver()
    elif route == "General Checkup":
        render_general()

    st.markdown(
        """
        <div class="disclaimer">
            <strong>Clinical disclaimer:</strong> This application is for education
            and decision support only. It is not a diagnosis or substitute for
            professional medical advice.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("MedSight · End-to-End Medical Assistant")


if __name__ == "__main__":
    main()
