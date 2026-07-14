# ========================================
# 🏥 Heart Disease Prediction Web App
# ========================================
# รันด้วยคำสั่ง: streamlit run app.py
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ========================================
# 🎨 Page Configuration
# ========================================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# 💅 Custom CSS
# ========================================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Title Styling */
    .main-title {
        text-align: center;
        color: white;
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    
    .sub-title {
        text-align: center;
        color: white;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card Styling */
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    .result-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .safe-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Metric Card */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# 📊 Load Model
# ========================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('heart_disease_model.pkl')
        return model
    except:
        st.error("❌ ไม่พบไฟล์โมเดล 'heart_disease_model.pkl'")
        st.info("💡 กรุณาอัปโหลดไฟล์โมเดลที่ฝึกจาก Google Colab")
        return None

model = load_model()

# ========================================
# 🏠 Header Section
# ========================================
st.markdown('<h1 class="main-title">❤️ Heart Disease Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">ระบบทำนายความเสี่ยงโรคหัวใจด้วย Machine Learning</p>', unsafe_allow_html=True)

# ========================================
# 📋 Sidebar - Input Form
# ========================================
st.sidebar.header("📝 กรอกข้อมูลสุขภาพ")
st.sidebar.markdown("---")

# Input Fields
col1, col2 = st.sidebar.columns(2)

with col1:
    age = st.number_input("🎂 อายุ (ปี)", min_value=20, max_value=100, value=50, step=1)
    sex = st.selectbox("⚧️ เพศ", options=["หญิง", "ชาย"])
    chest_pain = st.selectbox("💔 อาการเจ็บหน้าอก", 
                              options=["1: Typical Angina", "2: Atypical Angina", 
                                      "3: Non-Anginal Pain", "4: Asymptomatic"])
    resting_bp = st.number_input("💉 ความดันโลหิตขณะพัก (mm Hg)", 
                                  min_value=80, max_value=200, value=120, step=1)
    cholesterol = st.number_input("🩸 คอเลสเตอรอล (mg/dl)", 
                                   min_value=0, max_value=600, value=200, step=1)
    fasting_bs = st.selectbox("🍬 น้ำตาลในเลือดหลังอดอาหาร > 120 mg/dl", 
                               options=["ไม่", "ใช่"])

with col2:
    resting_ecg = st.selectbox("📈 ผล ECG ขณะพัก", 
                                options=["0: Normal", "1: ST-T Wave Abnormality", 
                                        "2: Left Ventricular Hypertrophy"])
    max_hr = st.number_input("💓 อัตราการเต้นหัวใจสูงสุด", 
                              min_value=60, max_value=220, value=150, step=1)
    exercise_angina = st.selectbox("🏃 อาการเจ็บหน้าอกขณะออกกำลังกาย", 
                                    options=["ไม่", "ใช่"])
    oldpeak = st.number_input("📊 ST Depression (Oldpeak)", 
                               min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    st_slope = st.selectbox("📉 Slope ของ ST Segment", 
                             options=["1: Upsloping", "2: Flat", "3: Downsloping"])

# Convert inputs to model format
sex_val = 1 if sex == "ชาย" else 0
chest_pain_val = int(chest_pain.split(':')[0])
fasting_bs_val = 1 if fasting_bs == "ใช่" else 0
resting_ecg_val = int(resting_ecg.split(':')[0])
exercise_angina_val = 1 if exercise_angina == "ใช่" else 0
st_slope_val = int(st_slope.split(':')[0])

# ========================================
# 🔮 Prediction Section
# ========================================
st.markdown("---")

if st.button("🔮 ทำนายผล", use_container_width=True):
    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_val],
        'ChestPainType': [chest_pain_val],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs_val],
        'RestingECG': [resting_ecg_val],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina_val],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope_val]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Display Results
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if prediction == 1:
            st.markdown(f'''
            <div class="result-card">
                <h1 style="font-size: 4rem; margin: 0;">⚠️</h1>
                <h2 style="color: white; margin: 1rem 0;">มีความเสี่ยงเป็นโรคหัวใจ</h2>
                <p style="font-size: 1.5rem; color: white;">
                    ความน่าจะเป็น: <strong>{probability[1]:.1%}</strong>
                </p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="safe-card">
                <h1 style="font-size: 4rem; margin: 0;">✅</h1>
                <h2 style="color: white; margin: 1rem 0;">ไม่มีความเสี่ยงเป็นโรคหัวใจ</h2>
                <p style="font-size: 1.5rem; color: white;">
                    ความน่าจะเป็น: <strong>{probability[0]:.1%}</strong>
                </p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Probability Gauge
    st.markdown("### 📊 ความน่าจะเป็นในการทำนาย")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability[1] * 100,
        title={'text': "ความเสี่ยงโรคหัวใจ (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': '#4facfe'},
                {'range': [30, 70], 'color': '#ffd700'},
                {'range': [70, 100], 'color': '#f5576c'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability[1] * 100
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Input Summary
    st.markdown("### 📋 สรุปข้อมูลที่คุณกรอก")
    
    summary_data = {
        'อายุ': f'{age} ปี',
        'เพศ': sex,
        'อาการเจ็บหน้าอก': chest_pain,
        'ความดันโลหิต': f'{resting_bp} mm Hg',
        'คอเลสเตอรอล': f'{cholesterol} mg/dl',
        'น้ำตาลในเลือดสูง': fasting_bs,
        'ผล ECG': resting_ecg,
        'อัตราการเต้นหัวใจสูงสุด': f'{max_hr} bpm',
        'เจ็บหน้าอกขณะออกกำลังกาย': exercise_angina,
        'ST Depression': f'{oldpeak}',
        'ST Slope': st_slope
    }
    
    summary_df = pd.DataFrame(list(summary_data.items()), columns=['รายการ', 'ค่า'])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ========================================
# ℹ️ About Section
# ========================================
st.markdown("---")
st.markdown("### ℹ️ เกี่ยวกับแอปพลิเคชันนี้")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>🤖 Model</h3>
        <p>Decision Tree Classifier</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>📊 Features</h3>
        <p>11 Features</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>🎯 Target</h3>
        <p>Heart Disease (0/1)</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="background: white; padding: 1.5rem; border-radius: 10px; margin-top: 1rem;">
    <h4>📌 คำอธิบาย Features:</h4>
    <ul>
        <li><strong>Age:</strong> อายุของผู้ป่วย</li>
        <li><strong>Sex:</strong> เพศ (0 = หญิง, 1 = ชาย)</li>
        <li><strong>ChestPainType:</strong> ประเภทอาการเจ็บหน้าอก</li>
        <li><strong>RestingBP:</strong> ความดันโลหิตขณะพัก</li>
        <li><strong>Cholesterol:</strong> ระดับคอเลสเตอรอล</li>
        <li><strong>FastingBS:</strong> น้ำตาลในเลือดหลังอดอาหาร (> 120 mg/dl)</li>
        <li><strong>RestingECG:</strong> ผลการตรวจ ECG ขณะพัก</li>
        <li><strong>MaxHR:</strong> อัตราการเต้นหัวใจสูงสุดที่ đạtได้</li>
        <li><strong>ExerciseAngina:</strong> อาการเจ็บหน้าอกขณะออกกำลังกาย</li>
        <li><strong>Oldpeak:</strong> ST depression ที่เกิดจากการออกกำลังกาย</li>
        <li><strong>ST_Slope:</strong> ความชันของ ST segment</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ========================================
# 📝 Footer
# ========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 1rem;">
    <p>⚠️ <strong>คำเตือน:</strong> แอปพลิเคชันนี้ใช้เพื่อการศึกษาเท่านั้น ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้</p>
    <p>© 2026 Heart Disease Prediction App | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)