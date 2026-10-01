import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & DARK THEME STYLES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Dark AI Dashboard CSS
st.markdown(
    """
    <style>
    /* Global Font & Dark Theme Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        color: #F8FAFC !important;
        background-color: #0B1120 !important;
    }
    
    .stApp {
        background-color: #0B1120 !important;
    }

    header[data-testid="stHeader"] {
        background-color: #0B1120 !important;
    }

    /* Hide Default Clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar Dark Theme Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #263247 !important;
    }

    div[data-testid="stSidebarContent"] {
        background-color: #111827 !important;
        color: #F8FAFC !important;
    }

    .sidebar-header {
        padding: 0.5rem 0;
        text-align: left;
    }

    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #F8FAFC !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .sidebar-subtitle {
        font-size: 0.78rem;
        color: #94A3B8 !important;
        margin-top: 0.15rem;
        font-weight: 500;
    }

    .sidebar-status-box {
        background-color: #172033 !important;
        border: 1px solid #263247 !important;
        border-radius: 8px;
        padding: 0.85rem;
        margin-top: 1.5rem;
    }

    .status-dot {
        height: 9px;
        width: 9px;
        background-color: #22C55E;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }

    /* Typography High Contrast Overrides */
    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }

    p, span, li, label, div {
        color: #CBD5E1;
    }

    .stMarkdown p {
        color: #CBD5E1 !important;
    }

    /* Dark SaaS Card Components */
    .saas-card {
        background-color: #172033 !important;
        border: 1px solid #263247 !important;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        color: #F8FAFC !important;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
    }

    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }

    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #38BDF8 !important;
    }

    .kpi-subtitle {
        font-size: 0.78rem;
        color: #CBD5E1 !important;
        margin-top: 0.2rem;
    }

    /* Hero Banner Component */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #312E81 100%) !important;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        border: 1px solid #3730A3;
        color: #FFFFFF !important;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
    }

    .hero-badge {
        display: inline-block;
        background-color: rgba(56, 189, 248, 0.15);
        color: #38BDF8 !important;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.025em;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #FFFFFF !important;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #CBD5E1 !important;
        max-width: 700px;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }

    /* Prediction Result Cards */
    .result-card-high {
        background: linear-gradient(135deg, #065F46 0%, #047857 100%) !important;
        color: #FFFFFF !important;
        padding: 2rem;
        border-radius: 14px;
        border: 1px solid #10B981;
        box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.4);
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .result-card-medium {
        background: linear-gradient(135deg, #1E40AF 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        padding: 2rem;
        border-radius: 14px;
        border: 1px solid #3B82F6;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.4);
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .result-card-low {
        background: linear-gradient(135deg, #991B1B 0%, #B91C1C 100%) !important;
        color: #FFFFFF !important;
        padding: 2rem;
        border-radius: 14px;
        border: 1px solid #EF4444;
        box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.4);
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .result-title {
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #E2E8F0 !important;
    }

    .result-value {
        font-size: 2.75rem;
        font-weight: 800;
        margin: 0.4rem 0;
        letter-spacing: 0.05em;
        color: #FFFFFF !important;
    }

    .result-model {
        font-size: 0.88rem;
        color: #E2E8F0 !important;
    }

    /* Streamlit Input Controls Visibility Overrides */
    div[data-baseweb="select"] > div, input, textarea {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
    }

    div[data-baseweb="popover"] div {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
    }

    div[role="option"] {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
    }

    div[role="option"]:hover {
        background-color: #334155 !important;
    }

    div[data-testid="stRadio"] label, div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }

    div[data-testid="stSlider"] span {
        color: #38BDF8 !important;
    }

    /* Streamlit Buttons High Contrast */
    div.stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.25rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35) !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #6366F1 0%, #818CF8 100%) !important;
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.5) !important;
    }

    div.stFormSubmitButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.45) !important;
    }

    /* Streamlit Tabs High Contrast */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1.2rem !important;
    }

    button[aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom: 3px solid #38BDF8 !important;
    }

    /* Plot Image Frame Styling */
    .plot-container {
        background-color: #FFFFFF !important;
        padding: 0.5rem;
        border-radius: 10px;
        border: 1px solid #263247;
        margin-bottom: 1rem;
    }

    /* Technology Badge Cards */
    .tech-card {
        background-color: #172033 !important;
        border: 1px solid #263247 !important;
        border-radius: 10px;
        padding: 0.9rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .tech-name {
        font-weight: 700;
        color: #38BDF8 !important;
        font-size: 0.95rem;
    }
    
    .tech-desc {
        font-size: 0.78rem;
        color: #CBD5E1 !important;
        margin-top: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# PATH DEFINITIONS
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'student_performance_model.pkl')
CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'student_performance_cleaned.csv')
EVALUATION_PATH = os.path.join(PROJECT_ROOT, 'reports', 'model_evaluation.csv')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'reports', 'plots')
METADATA_PATH = os.path.join(PROJECT_ROOT, 'models', 'model_metadata.json')

# -----------------------------------------------------------------------------
# CACHED DATA & MODEL LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading trained model binary: {e}")
        return None

@st.cache_data
def load_dataset():
    if not os.path.exists(CLEANED_DATA_PATH):
        return None
    try:
        return pd.read_csv(CLEANED_DATA_PATH)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_data
def load_evaluation():
    if not os.path.exists(EVALUATION_PATH):
        return None
    try:
        return pd.read_csv(EVALUATION_PATH)
    except Exception as e:
        st.error(f"Error loading evaluation metrics: {e}")
        return None

# Load dynamic data
df = load_dataset()
eval_df = load_evaluation()

# Determine best model dynamically
best_model_name = "Logistic Regression"
best_accuracy = 0.7850
best_f1 = 0.7831

if eval_df is not None and not eval_df.empty:
    best_row = eval_df.sort_values(by=['F1_Score', 'Accuracy'], ascending=False).iloc[0]
    best_model_name = str(best_row['Model'])
    best_accuracy = float(best_row['Accuracy'])
    best_f1 = float(best_row['F1_Score'])

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header">
            <div class="sidebar-title">🎓 Student Performance AI</div>
            <div class="sidebar-subtitle">Machine Learning • Education Analytics</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    navigation_page = st.radio(
        "Navigation Menu",
        [
            "🏠 Dashboard",
            "🎯 Predict Performance",
            "📊 Data Analysis",
            "🤖 Model Comparison",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Bottom Status Card
    st.markdown(
        f"""
        <div class="sidebar-status-box">
            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">ML Model Status</div>
            <div style="font-size: 0.95rem; font-weight: 800; color: #F8FAFC; margin: 0.2rem 0;">{best_model_name}</div>
            <div style="font-size: 0.8rem; color: #22C55E; font-weight: 600; display: flex; align-items: center; margin-top: 0.25rem;">
                <span class="status-dot"></span> Model Ready ({best_accuracy*100:.1f}% Acc)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# 1. DASHBOARD PAGE
# -----------------------------------------------------------------------------
def show_dashboard():
    # Hero Banner
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-badge">AI-Powered Education Analytics</div>
            <div class="hero-title">Student Performance AI</div>
            <div class="hero-subtitle">Understand student academic trajectories and predict performance levels using data-driven machine learning classification.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dynamic KPI Cards
    if df is not None:
        total_students = len(df)
        avg_attendance = df['Attendance'].mean()
        avg_study = df['Study_Hours'].mean()
        avg_prev_marks = df['Previous_Marks'].mean()
    else:
        total_students, avg_attendance, avg_study, avg_prev_marks = 1000, 71.1, 4.9, 65.0

    k1, k2, k3, k4, k5 = st.columns(5)
    
    with k1:
        st.markdown(
            f"""
            <div class="saas-card">
                <div class="kpi-title">👨‍🎓 Total Students</div>
                <div class="kpi-value">{total_students:,}</div>
                <div class="kpi-subtitle">Dataset records</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            f"""
            <div class="saas-card">
                <div class="kpi-title">📈 Avg Attendance</div>
                <div class="kpi-value">{avg_attendance:.1f}%</div>
                <div class="kpi-subtitle">Class attendance rate</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k3:
        st.markdown(
            f"""
            <div class="saas-card">
                <div class="kpi-title">📚 Avg Study Hours</div>
                <div class="kpi-value">{avg_study:.1f} <span style="font-size: 1rem; font-weight: normal;">hrs</span></div>
                <div class="kpi-subtitle">Daily self-study time</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k4:
        st.markdown(
            f"""
            <div class="saas-card">
                <div class="kpi-title">🎯 Avg Prev Marks</div>
                <div class="kpi-value">{avg_prev_marks:.1f}%</div>
                <div class="kpi-subtitle">Prior term score</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k5:
        st.markdown(
            f"""
            <div class="saas-card">
                <div class="kpi-title">🤖 Model Accuracy</div>
                <div class="kpi-value">{best_accuracy*100:.1f}%</div>
                <div class="kpi-subtitle">{best_model_name}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Performance Overview & Workflow Section
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.subheader("📊 Performance Target Distribution")
        st.markdown("<p style='color: #CBD5E1; font-size: 0.9rem; margin-bottom: 1rem;'>Distribution of students categorized into Low, Medium, and High performance target classes.</p>", unsafe_allow_html=True)
        
        perf_plot_path = os.path.join(PLOTS_DIR, 'performance_distribution.png')
        if os.path.exists(perf_plot_path):
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.image(perf_plot_path, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.subheader("🔄 How Prediction Works")
        st.markdown(
            """
            <div class="saas-card" style="padding: 1.5rem;">
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <div style="background: #1E293B; color: #38BDF8; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; border: 1px solid #334155;">1</div>
                        <div>
                            <div style="font-weight: 700; color: #F8FAFC; font-size: 0.95rem;">Student Data Input</div>
                            <div style="font-size: 0.8rem; color: #94A3B8;">Academic marks, study hours, attendance & support</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <div style="background: #1E293B; color: #38BDF8; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; border: 1px solid #334155;">2</div>
                        <div>
                            <div style="font-weight: 700; color: #F8FAFC; font-size: 0.95rem;">Feature Preprocessing</div>
                            <div style="font-size: 0.8rem; color: #94A3B8;">StandardScaler for numeric, OneHotEncoder for categorical</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <div style="background: #1E293B; color: #38BDF8; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; border: 1px solid #334155;">3</div>
                        <div>
                            <div style="font-weight: 700; color: #F8FAFC; font-size: 0.95rem;">Trained ML Classifier</div>
                            <div style="font-size: 0.8rem; color: #94A3B8;">Logistic Regression classification model pipeline</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <div style="background: #1E293B; color: #38BDF8; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; border: 1px solid #334155;">4</div>
                        <div>
                            <div style="font-weight: 700; color: #F8FAFC; font-size: 0.95rem;">Performance Prediction</div>
                            <div style="font-size: 0.8rem; color: #94A3B8;">Categorized into LOW, MEDIUM, or HIGH output</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("💡 Dataset Quick Insights")

    # 4 Quick Insights Cards
    i1, i2, i3, i4 = st.columns(4)

    with i1:
        st.markdown(
            f"""
            <div class="saas-card">
                <div style="font-size: 1.25rem; margin-bottom: 0.3rem;">📚</div>
                <div style="font-weight: 700; color: #F8FAFC;">Study Pattern</div>
                <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.2rem;">Students average <b style="color:#38BDF8;">{avg_study:.1f} hours</b> of daily self-study.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i2:
        st.markdown(
            f"""
            <div class="saas-card">
                <div style="font-size: 1.25rem; margin-bottom: 0.3rem;">📅</div>
                <div style="font-weight: 700; color: #F8FAFC;">Attendance Level</div>
                <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.2rem;">Average class attendance rate stands at <b style="color:#38BDF8;">{avg_attendance:.1f}%</b>.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i3:
        st.markdown(
            f"""
            <div class="saas-card">
                <div style="font-size: 1.25rem; margin-bottom: 0.3rem;">💯</div>
                <div style="font-weight: 700; color: #F8FAFC;">Academic History</div>
                <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.2rem;">Prior term marks average <b style="color:#38BDF8;">{avg_prev_marks:.1f}%</b> across cohort.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i4:
        avg_sleep = df['Sleep_Hours'].mean() if df is not None else 6.9
        st.markdown(
            f"""
            <div class="saas-card">
                <div style="font-size: 1.25rem; margin-bottom: 0.3rem;">😴</div>
                <div style="font-weight: 700; color: #F8FAFC;">Lifestyle Balance</div>
                <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.2rem;">Students maintain an average sleep duration of <b style="color:#38BDF8;">{avg_sleep:.1f} hours</b>.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------------------------------------------------------
# 2. PREDICTION PAGE
# -----------------------------------------------------------------------------
def show_prediction():
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-weight: 800; color: #F8FAFC; margin-bottom: 0.2rem;">Predict Student Performance</h1>
            <p style="color: #CBD5E1; font-size: 1rem;">Enter student academic metrics and lifestyle factors to generate a real-time ML estimation.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    model = load_model()
    if model is None:
        st.error("⚠️ Model file `student_performance_model.pkl` is missing. Please train the model first.")
        return

    with st.form("prediction_form_v3"):
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown(
                """
                <div style="font-weight: 700; font-size: 1.05rem; color: #38BDF8; margin-bottom: 1rem; border-bottom: 2px solid #263247; padding-bottom: 0.5rem;">
                    📚 Academic Parameters
                </div>
                """,
                unsafe_allow_html=True
            )
            study_hours = st.slider("Study Hours (per day)", 0.0, 10.0, 6.0, 0.1, help="Average daily study time outside of class")
            attendance = st.slider("Attendance (%)", 40.0, 100.0, 80.0, 0.1, help="Class attendance percentage")
            previous_marks = st.slider("Previous Marks (%)", 30.0, 100.0, 70.0, 0.1, help="Academic percentage obtained in previous term")
            assignment_score = st.slider("Assignment Score (0–20)", 0.0, 20.0, 14.0, 0.1, help="Cumulative score in assignments")
            internal_marks = st.slider("Internal Marks (0–30)", 0.0, 30.0, 20.0, 0.1, help="Marks in internal mid-term examinations")

        with c_right:
            st.markdown(
                """
                <div style="font-weight: 700; font-size: 1.05rem; color: #38BDF8; margin-bottom: 1rem; border-bottom: 2px solid #263247; padding-bottom: 0.5rem;">
                    🌱 Lifestyle & Family Support
                </div>
                """,
                unsafe_allow_html=True
            )
            sleep_hours = st.slider("Sleep Hours (per day)", 4.0, 10.0, 7.0, 0.1, help="Daily average sleep duration")
            internet_hours = st.slider("Internet Hours (non-academic)", 0.0, 10.0, 3.0, 0.1, help="Daily average hours spent on non-academic internet")
            family_support = st.selectbox("Family Support Level", ["Low", "Medium", "High"], index=1, help="Level of educational encouragement from family")
            extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"], index=0, help="Active participation in sports or clubs")

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🎯 Predict Performance", type="primary", use_container_width=True)

    if submit_btn:
        input_data = pd.DataFrame([{
            'Study_Hours': study_hours,
            'Attendance': attendance,
            'Previous_Marks': previous_marks,
            'Assignment_Score': assignment_score,
            'Internal_Marks': internal_marks,
            'Sleep_Hours': sleep_hours,
            'Internet_Hours': internet_hours,
            'Family_Support': family_support,
            'Extracurricular': extracurricular
        }])

        try:
            prediction = model.predict(input_data)[0]
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Prominent Result Card Styling
            card_class = f"result-card-{prediction.lower()}"
            st.markdown(
                f"""
                <div class="{card_class}">
                    <div class="result-title">🎯 PREDICTION RESULT</div>
                    <div class="result-value">{prediction.upper()}</div>
                    <div class="result-model">Model: {best_model_name}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Horizontal Probability Bars & Model Confidence
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_data)[0]
                classes = list(model.classes_)
                
                prob_pairs = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
                top_class, top_prob = prob_pairs[0]
                
                st.markdown(
                    f"""
                    <div class="saas-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <div style="font-weight: 700; color: #F8FAFC; font-size: 1.05rem;">📊 Probability Distribution</div>
                            <div style="background-color: #1E293B; color: #38BDF8; padding: 0.35rem 0.85rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 700; border: 1px solid #334155;">
                                Model Confidence: <b>{top_prob*100:.1f}%</b>
                            </div>
                        </div>
                    """,
                    unsafe_allow_html=True
                )
                
                for cls, pr in prob_pairs:
                    pct = pr * 100
                    color = "#22C55E" if cls == 'High' else ("#3B82F6" if cls == 'Medium' else "#EF4444")
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 0.85rem;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 600; color: #F8FAFC; margin-bottom: 0.25rem;">
                                <span>{cls} Performance</span>
                                <span style="color: {color}; font-weight: 700;">{pct:.1f}%</span>
                            </div>
                            <div style="background-color: #1E293B; height: 12px; border-radius: 9999px; overflow: hidden; border: 1px solid #334155;">
                                <div style="background-color: {color}; width: {pct}%; height: 100%; border-radius: 9999px;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            # Input Summary Card
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div style='font-weight: 700; color: #F8FAFC; font-size: 1.05rem; margin-bottom: 0.75rem;'>📋 Student Input Summary</div>", unsafe_allow_html=True)
            
            s1, s2 = st.columns(2)
            with s1:
                st.markdown(
                    f"""
                    <div class="saas-card" style="font-size: 0.9rem;">
                        <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #263247;">
                            <span style="color: #94A3B8;">Study Hours</span>
                            <span style="font-weight: 700; color: #F8FAFC;">{study_hours} hrs/day</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #263247;">
                            <span style="color: #94A3B8;">Attendance</span>
                            <span style="font-weight: 700; color: #F8FAFC;">{attendance}%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #263247;">
                            <span style="color: #94A3B8;">Previous Marks</span>
                            <span style="font-weight: 700; color: #F8FAFC;">{previous_marks}%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #263247;">
                            <span style="color: #94A3B8;">Assignment Score</span>
                            <span style="font-weight: 700; color: #F8FAFC;">{assignment_score} / 20</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.4rem 0;">
                            <span style="color: #94A3B8;">Internal Marks</span>
                            <span style="font-weight: 700; color: #F8FAFC;">{internal_marks} / 30</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with s2:
                st.markdown(
                    f"""
                    <div class="saas-card" style="font-size: 0.9rem;">
                        <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #263247;">
                            <span style="color: #94A3B8;">Sleep Hours</span>
                            <span style="font-weight: 700; color: #F8FAFC;">{sleep_hours} hrs/day</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #263247;">
                            <span style="color: #94A3B8;">Internet Hours</span>
                            <span style="font-weight: 700; color: #F8FAFC;">{internet_hours} hrs/day</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #263247;">
                            <span style="color: #94A3B8;">Family Support</span>
                            <span style="font-weight: 700; color: #F8FAFC;">{family_support}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.4rem 0;">
                            <span style="color: #94A3B8;">Extracurricular</span>
                            <span style="font-weight: 700; color: #F8FAFC;">{extracurricular}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Explanation Notice
            st.markdown(
                """
                <div style="background-color: #1E293B; border-left: 4px solid #38BDF8; padding: 1rem; border-radius: 6px; margin: 1rem 0;">
                    <div style="font-weight: 700; color: #38BDF8; margin-bottom: 0.2rem;">About this prediction</div>
                    <div style="font-size: 0.85rem; color: #CBD5E1;">This result is generated by a machine learning model trained on the project's dataset. It is an estimated prediction and should not be treated as a definitive judgment of a student's ability or future performance.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Downloadable Prediction Report
            report_text = f"""==================================================
STUDENT PERFORMANCE PREDICTION REPORT
==================================================

[STUDENT INPUT PARAMETERS]
Study Hours:         {study_hours} hrs/day
Attendance:          {attendance}%
Previous Marks:      {previous_marks}%
Assignment Score:    {assignment_score} / 20
Internal Marks:      {internal_marks} / 30
Sleep Hours:         {sleep_hours} hrs/day
Internet Hours:      {internet_hours} hrs/day
Family Support:      {family_support}
Extracurricular:     {extracurricular}

[PREDICTION RESULT]
Predicted Performance Level: {prediction.upper()}
Model Used:                 {best_model_name}
Model Test Accuracy:        {best_accuracy*100:.2f}%
"""
            if hasattr(model, 'predict_proba'):
                report_text += "\n[CLASS PROBABILITIES]\n"
                for cls, pr in prob_pairs:
                    report_text += f"{cls}: {pr * 100:.2f}%\n"

            st.download_button(
                label="⬇ Download Prediction Report",
                data=report_text,
                file_name="student_performance_prediction.txt",
                mime="text/plain",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error during inference execution: {e}")

# -----------------------------------------------------------------------------
# 3. DATA ANALYSIS PAGE
# -----------------------------------------------------------------------------
def show_data_analysis():
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-weight: 800; color: #F8FAFC; margin-bottom: 0.2rem;">Data Analysis</h1>
            <p style="color: #CBD5E1; font-size: 1rem;">Explore underlying distribution patterns and correlations across student cohorts.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    def load_img(filename):
        path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(path):
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.image(path, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning(f"Plot missing: `{filename}`")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Target & Overview",
        "📚 Academic Factors",
        "🌱 Behavioral & Support",
        "🔗 Feature Relationships",
        "🔥 Advanced & Heatmap"
    ])

    with tab1:
        st.subheader("Performance Distribution Overview")
        load_img("performance_distribution.png")

    with tab2:
        st.subheader("Academic Metric Distributions")
        col1, col2 = st.columns(2)
        with col1:
            load_img("study_hours_distribution.png")
            load_img("previous_marks_distribution.png")
        with col2:
            load_img("attendance_distribution.png")
            load_img("assignment_score_distribution.png")
        load_img("internal_marks_distribution.png")

    with tab3:
        st.subheader("Behavioral & Support Factors")
        c1, c2 = st.columns(2)
        with c1:
            load_img("sleep_hours_distribution.png")
            load_img("family_support_vs_performance.png")
        with c2:
            load_img("internet_hours_distribution.png")
            load_img("extracurricular_vs_performance.png")

    with tab4:
        st.subheader("Bivariate Relationships vs Target Performance")
        r1, r2 = st.columns(2)
        with r1:
            load_img("study_hours_vs_performance.png")
            load_img("previous_marks_vs_performance.png")
            load_img("study_hours_vs_previous_marks.png")
        with r2:
            load_img("attendance_vs_performance.png")
            load_img("internal_marks_vs_performance.png")
            load_img("attendance_vs_previous_marks.png")

    with tab5:
        st.subheader("Outlier & Correlation Analysis")
        load_img("numerical_boxplots.png")
        load_img("correlation_heatmap.png")
        load_img("pairplot.png")

# -----------------------------------------------------------------------------
# 4. MODEL COMPARISON PAGE
# -----------------------------------------------------------------------------
def show_model_comparison():
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-weight: 800; color: #F8FAFC; margin-bottom: 0.2rem;">Model Performance</h1>
            <p style="color: #CBD5E1; font-size: 1rem;">Compare machine learning classifiers evaluated on unseen test data.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if eval_df is None:
        st.error("⚠️ Evaluation metrics file missing.")
        return

    # Selected Model Highlight Banner
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #312E81 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #3730A3;">
            <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: #38BDF8;">🏆 Selected Final Model</div>
            <div style="font-size: 1.8rem; font-weight: 800; margin: 0.2rem 0; color: #FFFFFF;">{best_model_name}</div>
            <div style="display: flex; gap: 2rem; margin-top: 0.75rem; font-size: 0.95rem; color: #CBD5E1;">
                <div>Accuracy: <b style="color:#38BDF8;">{best_accuracy*100:.2f}%</b></div>
                <div>F1-Score: <b style="color:#38BDF8;">{best_f1*100:.2f}%</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📊 Evaluation Metrics Table")
    formatted_eval = eval_df.copy()
    formatted_eval['Accuracy'] = formatted_eval['Accuracy'].apply(lambda x: f"{x*100:.2f}%")
    formatted_eval['Precision'] = formatted_eval['Precision'].apply(lambda x: f"{x*100:.2f}%")
    formatted_eval['Recall'] = formatted_eval['Recall'].apply(lambda x: f"{x*100:.2f}%")
    formatted_eval['F1_Score'] = formatted_eval['F1_Score'].apply(lambda x: f"{x*100:.2f}%")
    
    st.dataframe(formatted_eval, use_container_width=True)

    st.subheader("📈 Model Comparison Chart")
    chart_path = os.path.join(PLOTS_DIR, 'model_comparison.png')
    if os.path.exists(chart_path):
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.image(chart_path, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🧩 Confusion Matrix Analysis")
    
    cm_list = [
        ("Logistic Regression", "confusion_matrix_logistic_regression.png"),
        ("K-Nearest Neighbors", "confusion_matrix_knn.png"),
        ("Support Vector Machine", "confusion_matrix_svm.png"),
        ("Decision Tree", "confusion_matrix_decision_tree.png"),
        ("Random Forest", "confusion_matrix_random_forest.png")
    ]

    c1, c2 = st.columns(2)
    for idx, (title, fname) in enumerate(cm_list):
        fpath = os.path.join(PLOTS_DIR, fname)
        col = c1 if idx % 2 == 0 else c2
        with col:
            if os.path.exists(fpath):
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.image(fpath, caption=f"Confusion Matrix — {title}", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. ABOUT PAGE
# -----------------------------------------------------------------------------
def show_about():
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-weight: 800; color: #F8FAFC; margin-bottom: 0.2rem;">About Student Performance AI</h1>
            <p style="color: #CBD5E1; font-size: 1rem;">Architecture, methodologies, and technical stack details.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown(
            """
            ### 🎯 Project Objective
            To build a machine learning web application capable of predicting student performance levels (**Low**, **Medium**, **High**) based on academic history, study habits, and support systems.

            ### 🧠 Machine Learning Approach
            - **Data Cleaning:** Median imputation for continuous variables, mode imputation for categorical variables.
            - **Feature Preprocessing:** `StandardScaler` for continuous metrics, `OneHotEncoder(drop='first')` for categorical inputs.
            - **Validation:** 80/20 stratified train/test split (`random_state=42`).
            - **Evaluation Metrics:** Weighted Accuracy, Precision, Recall, and F1-Score.

            ### 🔬 Models Evaluated
            1. Logistic Regression (Selected Best Model)
            2. Support Vector Machine (SVM)
            3. Random Forest Classifier
            4. K-Nearest Neighbors (KNN)
            5. Decision Tree Classifier
            """
        )

        st.markdown("### ⚙️ Technology Stack UI")
        
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.markdown('<div class="tech-card"><div class="tech-name">Python</div><div class="tech-desc">Core Language</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="tech-card" style="margin-top:0.5rem;"><div class="tech-name">Scikit-learn</div><div class="tech-desc">ML Pipelines</div></div>', unsafe_allow_html=True)
        with t2:
            st.markdown('<div class="tech-card"><div class="tech-name">Pandas</div><div class="tech-desc">Dataframes</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="tech-card" style="margin-top:0.5rem;"><div class="tech-name">Matplotlib</div><div class="tech-desc">Visualizations</div></div>', unsafe_allow_html=True)
        with t3:
            st.markdown('<div class="tech-card"><div class="tech-name">NumPy</div><div class="tech-desc">Numerical Math</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="tech-card" style="margin-top:0.5rem;"><div class="tech-name">Seaborn</div><div class="tech-desc">Statistical Charts</div></div>', unsafe_allow_html=True)
        with t4:
            st.markdown('<div class="tech-card"><div class="tech-name">Streamlit</div><div class="tech-desc">Web App UI</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="tech-card" style="margin-top:0.5rem;"><div class="tech-name">Joblib</div><div class="tech-desc">Model Serialization</div></div>', unsafe_allow_html=True)

    with c2:
        st.markdown(
            f"""
            <div class="saas-card" style="padding: 1.5rem;">
                <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 0.75rem; border-bottom: 1px solid #263247; padding-bottom: 0.5rem;">
                    📌 Model Status Card
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8;">STATUS</div>
                <div style="font-weight: 700; color: #22C55E; font-size: 1.1rem; margin-bottom: 0.75rem;">● Ready for Inference</div>
                
                <div style="font-size: 0.85rem; color: #94A3B8;">SELECTED MODEL</div>
                <div style="font-weight: 700; color: #F8FAFC; font-size: 1rem; margin-bottom: 0.75rem;">{best_model_name}</div>
                
                <div style="font-size: 0.85rem; color: #94A3B8;">TEST ACCURACY</div>
                <div style="font-weight: 700; color: #38BDF8; font-size: 1rem; margin-bottom: 0.75rem;">{best_accuracy*100:.2f}%</div>
                
                <div style="font-size: 0.85rem; color: #94A3B8;">TEST F1-SCORE</div>
                <div style="font-weight: 700; color: #38BDF8; font-size: 1rem;">{best_f1*100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Student Performance AI • Machine Learning Project | Built with Python, Scikit-learn & Streamlit")

# -----------------------------------------------------------------------------
# MAIN DISPATCHER
# -----------------------------------------------------------------------------
if navigation_page == "🏠 Dashboard":
    show_dashboard()
elif navigation_page == "🎯 Predict Performance":
    show_prediction()
elif navigation_page == "📊 Data Analysis":
    show_data_analysis()
elif navigation_page == "🤖 Model Comparison":
    show_model_comparison()
elif navigation_page == "ℹ️ About Project":
    show_about()
