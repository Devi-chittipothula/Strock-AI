
import os
import streamlit as st
import pandas as pd

from authentication.auth import (
    register_user,
    login_user,
    create_users_table
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="StrokeAI",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

COMPARISON_FILE = os.path.join(
    MODEL_DIR, "model_comparison.csv"
)

BEST_MODEL_FILE = os.path.join(
    MODEL_DIR, "best_model.txt"
)

create_users_table()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "logged_in": False,
    "user": None,
    "latest_prediction": None,
    "main_navigation": "Dashboard"
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def go_to_page(page):
    st.session_state.main_navigation = page


# --------------------------------------------------
# CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F5F8FC;
}

.block-container {
    padding-top: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #102A43;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}

.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    min-height: 42px;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD PREDICTOR
# --------------------------------------------------

@st.cache_resource
def load_predictor():

    from prediction.predict import StrokePredictor

    return StrokePredictor()


# --------------------------------------------------
# HEALTH RECOMMENDATIONS
# --------------------------------------------------

def generate_recommendations(patient):

    recommendations = []

    if patient["hypertension"] == 1:

        recommendations.append(
            "Monitor your blood pressure regularly and "
            "discuss hypertension management with your doctor."
        )

    if patient["heart_disease"] == 1:

        recommendations.append(
            "Follow your heart-care plan and attend "
            "regular medical check-ups."
        )

    if patient["smoking_status"] in [
        "smokes",
        "formerly smoked"
    ]:

        recommendations.append(
            "Avoid smoking. If you currently smoke, "
            "ask a healthcare professional for support to quit."
        )

    if patient["bmi"] >= 25 and patient["age"] >= 18:

        recommendations.append(
            "Discuss healthy weight management, balanced "
            "nutrition and suitable physical activity "
            "with a healthcare professional."
        )

    if patient["avg_glucose_level"] >= 140:

        recommendations.append(
            "Discuss your glucose reading with a healthcare "
            "professional. Appropriate glucose testing "
            "may be recommended."
        )

    if patient["age"] >= 60:

        recommendations.append(
            "Attend regular health check-ups and discuss "
            "your cardiovascular health with your doctor."
        )

    recommendations.extend([
        "Choose a balanced diet with vegetables, fruits "
        "and appropriate amounts of salt.",

        "Stay physically active according to your health "
        "condition and your doctor's advice.",

        "Maintain a regular sleep schedule and stay hydrated.",

        "Take prescribed medicines as directed by your doctor."
    ])

    return recommendations


# --------------------------------------------------
# DISPLAY ASSESSMENT
# --------------------------------------------------

def display_assessment(saved):

    patient = saved["patient"]
    results = saved["results"]

    prediction = results.get("stroke_prediction")

    if prediction is None:

        ft_result = results.get("FT-Transformer", {})

        prediction = ft_result.get("prediction")

    if prediction is None:

        st.error("Assessment result is unavailable.")
        return

    # Experimental score from selected model

    score = results.get("stroke_probability")

    if score is None:

        score = results.get(
            "FT-Transformer", {}
        ).get("probability")

    st.divider()

    st.subheader("Your Assessment Result")

    # PERCENTAGE DISPLAY

    if score is not None:

        score = float(score)

        if 0 <= score <= 1:

            st.metric(
                label="Experimental AI Model Score",
                value=f"{score * 100:.2f}%"
            )

    # PREDICTION RESULT
        st.write(
            "This does not mean that stroke risk is absent. "
            "Continue routine healthcare and consult a doctor "
            "about any concerns."
        )

    # PRECAUTIONS

    st.divider()

    st.subheader("Recommended Health Precautions")

    st.write(
        "Based on the information entered, here are some "
        "general health suggestions:"
    )

    recommendations = generate_recommendations(patient)

    for i, recommendation in enumerate(
        recommendations,
        start=1
    ):

        st.markdown(f"{i}. {recommendation}")

    # EMERGENCY WARNING

    st.divider()

    st.warning(
        """
        **Important Medical Warning**

        Sudden face drooping, weakness in an arm or leg,
        difficulty speaking, sudden vision problems or loss
        of balance may be signs of a stroke.

        Seek emergency medical care immediately if these
        symptoms occur. Do not wait for an AI prediction.
        """,
        icon="🚨"
    )

    st.caption(
        "StrokeAI is an experimental educational project. "
        "Its model score is not a clinically validated risk "
        "assessment, diagnosis or substitute for medical advice."
    )


# --------------------------------------------------
# LOGIN AND REGISTER
# --------------------------------------------------

def show_login():

    st.write("")

    left, center, right = st.columns([1, 1.2, 1])

    with center:

        st.markdown(
            "<h1 style='text-align:center;color:#2563EB;'>🧠 StrokeAI</h1>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p style='text-align:center;color:#64748B;'>"
            "AI-Powered Brain Stroke Risk Assessment Platform"
            "</p>",
            unsafe_allow_html=True
        )

        login_tab, register_tab = st.tabs(
            ["🔐 Login", "✨ Register"]
        )

        with login_tab:

            st.subheader("Welcome Back")

            email = st.text_input(
                "Email Address",
                key="login_email"
            )

            password = st.text_input(
                "Password",
                type="password",
                key="login_password"
            )

            if st.button(
                "Login",
                use_container_width=True
            ):

                if not email.strip() or not password:

                    st.warning(
                        "Please enter your email and password."
                    )

                else:

                    try:

                        success, user = login_user(
                            email.strip(),
                            password
                        )

                        if success:

                            st.session_state.logged_in = True
                            st.session_state.user = user
                            st.session_state.main_navigation = "Dashboard"

                            st.rerun()

                        else:

                            st.error(
                                "Invalid email or password."
                            )

                    except Exception as e:

                        st.error(f"Login error: {e}")

        with register_tab:

            st.subheader("Create Account")

            name = st.text_input(
                "Full Name",
                key="register_name"
            )

            email = st.text_input(
                "Email Address",
                key="register_email"
            )

            password = st.text_input(
                "Create Password",
                type="password",
                key="register_password"
            )

            confirm = st.text_input(
                "Confirm Password",
                type="password",
                key="confirm_password"
            )

            if st.button(
                "Create Account",
                use_container_width=True
            ):

                if not name.strip() or not email.strip() or not password:

                    st.warning("Please fill all fields.")

                elif password != confirm:

                    st.error("Passwords do not match.")

                elif len(password) < 8:

                    st.error(
                        "Password must contain at least 8 characters."
                    )

                else:

                    try:

                        success, message = register_user(
                            name.strip(),
                            email.strip(),
                            password
                        )

                        if success:

                            st.success(
                                message + " Please login."
                            )

                        else:

                            st.error(message)

                    except Exception as e:

                        st.error(
                            f"Registration error: {e}"
                        )

        st.caption(
            "© 2026 StrokeAI | AI Healthcare Research Project"
        )


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

def show_dashboard():

    user = st.session_state.user

    user_name = (
        user.get("name", "User")
        if isinstance(user, dict)
        else "User"
    )

    st.sidebar.title("🧠 StrokeAI")
    st.sidebar.caption("AI Healthcare Analytics")
    st.sidebar.divider()

    st.sidebar.write(f"Welcome, {user_name}")

    pages = [
        "Dashboard",
        "Stroke Prediction",
        "Model Comparison",
        "Reports"
    ]

    st.sidebar.radio(
        "Navigation",
        pages,
        key="main_navigation"
    )

    page = st.session_state.main_navigation

    st.sidebar.divider()

    if st.sidebar.button(
        "🚪 Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.latest_prediction = None
        st.session_state.main_navigation = "Dashboard"

        st.rerun()

    # --------------------------------------------------
    # DASHBOARD HOME
    # --------------------------------------------------

    if page == "Dashboard":

        st.title(f"Welcome to StrokeAI, {user_name} 👋")

        st.caption(
            "AI-Powered Brain Stroke Risk Assessment Platform"
        )

        st.divider()

        c1, c2, c3 = st.columns(3)

        c1.metric("AI Models", "03")
        c2.metric("Platform Status", "Active")
        c3.metric("Account", "Verified")

        st.write("")

        st.subheader("Your Workspace")

        col1, col2 = st.columns(2)

        with col1:

            with st.container(border=True):

                st.subheader("🧠 Stroke Prediction")

                st.write(
                    "Enter patient health details and receive "
                    "an AI-based assessment with general "
                    "health suggestions."
                )

                st.button(
                    "Open Prediction →",
                    key="open_prediction",
                    use_container_width=True,
                    on_click=go_to_page,
                    args=("Stroke Prediction",)
                )

        with col2:

            with st.container(border=True):

                st.subheader("📊 Model Performance")

                st.write(
                    "Explore the research evaluation of "
                    "DNN, TabNet and FT-Transformer."
                )

                st.button(
                    "View Model Performance →",
                    key="open_comparison",
                    use_container_width=True,
                    on_click=go_to_page,
                    args=("Model Comparison",)
                )

        st.write("")

        st.subheader("About StrokeAI")

        st.info(
            "StrokeAI is an educational machine-learning "
            "research prototype for exploring stroke-related "
            "health information."
        )

        st.warning(
            "This platform is not a medical diagnostic tool. "
            "Never use its predictions to make medical decisions."
        )

    # --------------------------------------------------
    # STROKE PREDICTION
    # --------------------------------------------------

    elif page == "Stroke Prediction":

        st.title("🧠 Stroke Risk Assessment")

        st.caption(
            "Enter patient information to generate "
            "an AI-based assessment."
        )

        st.divider()

        with st.form("prediction_form"):

            st.subheader("Patient Information")

            col1, col2 = st.columns(2)

            with col1:

                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female", "Other"]
                )

                age = st.number_input(
                    "Age",
                    min_value=0.0,
                    max_value=120.0,
                    value=45.0
                )

                hypertension = st.selectbox(
                    "Hypertension",
                    [0, 1],
                    format_func=lambda x: "Yes" if x else "No"
                )

                heart_disease = st.selectbox(
                    "Heart Disease",
                    [0, 1],
                    format_func=lambda x: "Yes" if x else "No"
                )

                ever_married = st.selectbox(
                    "Ever Married",
                    ["Yes", "No"]
                )

            with col2:

                work_type = st.selectbox(
                    "Work Type",
                    [
                        "Private",
                        "Self-employed",
                        "Govt_job",
                        "children",
                        "Never_worked"
                    ]
                )

                residence_type = st.selectbox(
                    "Residence Type",
                    ["Urban", "Rural"]
                )

                avg_glucose_level = st.number_input(
                    "Average Glucose Level",
                    min_value=0.0,
                    max_value=500.0,
                    value=100.0
                )

                bmi = st.number_input(
                    "BMI",
                    min_value=0.0,
                    max_value=100.0,
                    value=25.0
                )

                smoking_status = st.selectbox(
                    "Smoking Status",
                    [
                        "never smoked",
                        "formerly smoked",
                        "smokes",
                        "Unknown"
                    ]
                )

            submitted = st.form_submit_button(
                "🔍 Assess Stroke Risk",
                use_container_width=True
            )

        if submitted:

            if (
                age <= 0
                or bmi <= 0
                or avg_glucose_level <= 0
            ):

                st.error(
                    "Please enter valid patient information."
                )

            else:

                patient_data = {
                    "gender": gender,
                    "age": age,
                    "hypertension": hypertension,
                    "heart_disease": heart_disease,
                    "ever_married": ever_married,
                    "work_type": work_type,
                    "Residence_type": residence_type,
                    "avg_glucose_level": avg_glucose_level,
                    "bmi": bmi,
                    "smoking_status": smoking_status
                }

                try:

                    with st.spinner(
                        "Analyzing patient information..."
                    ):

                        predictor = load_predictor()

                        results = predictor.predict_all(
                            patient_data
                        )

                        st.session_state.latest_prediction = {
                            "patient": patient_data,
                            "results": results
                        }

                    st.success(
                        "Patient assessment completed!"
                    )

                except Exception as e:

                    st.error(
                        f"Prediction error: {e}"
                    )

        saved = st.session_state.latest_prediction

        if saved is not None:

            display_assessment(saved)

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------

    elif page == "Model Comparison":

        st.title("📊 Model Performance Comparison")

        st.caption(
            "Research evaluation of the three trained AI models."
        )

        st.divider()

        if os.path.exists(COMPARISON_FILE):

            try:

                df = pd.read_csv(COMPARISON_FILE)

                st.subheader("Evaluation Metrics")

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                metric_columns = [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "ROC-AUC"
                ]

                available_metrics = [
                    col for col in metric_columns
                    if col in df.columns
                ]

                if "Model" in df.columns and available_metrics:

                    st.subheader("Performance Chart")

                    chart_df = df.set_index(
                        "Model"
                    )[available_metrics]

                    st.bar_chart(chart_df)

                if os.path.exists(BEST_MODEL_FILE):

                    with open(
                        BEST_MODEL_FILE,
                        "r",
                        encoding="utf-8"
                    ) as file:

                        best_model = file.read().strip()

                    st.success(
                        f"Selected model: {best_model}"
                    )

                st.download_button(
                    "⬇️ Download Comparison",
                    data=df.to_csv(index=False),
                    file_name="strokeai_model_comparison.csv",
                    mime="text/csv"
                )

            except Exception as e:

                st.error(
                    f"Comparison error: {e}"
                )

        else:

            st.error(
                "model_comparison.csv not found. "
                "Run python training/compare_models.py"
            )

        st.warning(
            "These metrics are research evaluation results "
            "and do not establish clinical reliability."
        )

    # --------------------------------------------------
    # REPORTS
    # --------------------------------------------------

    elif page == "Reports":

        st.title("📄 Patient Assessment Reports")

        st.caption(
            "View and download your latest patient assessment."
        )

        st.divider()

        saved = st.session_state.latest_prediction

        if saved is None:

            st.info(
                "No assessment report available. "
                "Generate an assessment from the Stroke Prediction page."
            )

            st.button(
                "Go to Stroke Prediction →",
                on_click=go_to_page,
                args=("Stroke Prediction",)
            )

        else:

            patient = saved["patient"]
            results = saved["results"]

            st.subheader("Patient Information")

            patient_df = pd.DataFrame(
                list(patient.items()),
                columns=["Parameter", "Value"]
            )

            st.dataframe(
                patient_df,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Assessment Summary")

            display_assessment(saved)

            prediction = results.get("stroke_prediction")

            if prediction is None:

                prediction = results.get(
                    "FT-Transformer", {}
                ).get("prediction")

            score = results.get("stroke_probability")

            if score is None:

                score = results.get(
                    "FT-Transformer", {}
                ).get("probability")

            if prediction is not None:

                assessment = (
                    "Further Medical Review Suggested"
                    if int(prediction) == 1
                    else "No Elevated Flag From This AI Model"
                )

                report_text = (
                    "STROKEAI - PATIENT ASSESSMENT REPORT\n\n"
                    "PATIENT INFORMATION\n"
                    + patient_df.to_csv(index=False)
                    + "\nASSESSMENT RESULT\n"
                    + assessment
                )

                if score is not None:

                    report_text += (
                        f"\nExperimental AI Model Score: "
                        f"{float(score) * 100:.2f}%"
                    )

                report_text += (
                    "\n\nGENERAL HEALTH SUGGESTIONS\n"
                    + "\n".join(
                        f"- {item}"
                        for item in generate_recommendations(patient)
                    )
                    + "\n\nMEDICAL DISCLAIMER\n"
                    "This experimental model score is not a "
                    "clinically validated stroke probability "
                    "or diagnosis. Consult a qualified "
                    "healthcare professional.\n"
                )

                st.download_button(
                    "⬇️ Download Patient Report",
                    data=report_text,
                    file_name="StrokeAI_Patient_Report.txt",
                    mime="text/plain",
                    use_container_width=True
                )


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

if st.session_state.logged_in:

    show_dashboard()

else:

    show_login()
