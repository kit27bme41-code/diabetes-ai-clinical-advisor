import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from fpdf import FPDF
import datetime

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Diabetes AI Clinical Advisor", layout="wide")

# --- DATA & MODEL ENGINE ---
@st.cache_resource
def load_clinical_engine():
    df = pd.read_csv('cleaned_diabetes_data.csv')
    # Features including HOMA_IR
    features = ['Gender', 'Age', 'Glucose', 'BMI', 'Waist', 'Sleep', 'Sedentary', 'HOMA_IR']
    X = df[features]
    y = df['Target']
    
    imputer = SimpleImputer(strategy='median').fit(X)
    X_imputed = imputer.transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_imputed, y)
    return model, imputer, df, features

model, imputer, raw_df, features = load_clinical_engine()

# --- SIDEBAR INPUTS ---
st.sidebar.header("📋 Patient Vitals")
age = st.sidebar.slider("Age", 1, 100, 52)
gender = st.sidebar.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x==1 else "Female")
glucose = st.sidebar.number_input("Fasting Glucose (mg/dL)", 50, 300, 113)
insulin = st.sidebar.number_input("Insulin (uIU/mL)", 1.0, 100.0, 15.11)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 31.57)
waist = st.sidebar.slider("Waist Circumference (cm)", 30.0, 180.0, 106.21)
sleep = st.sidebar.slider("Sleep Hours", 2, 14, 8)
sedentary = st.sidebar.number_input("Sedentary Minutes/Day", 0, 1440, 300)

# Automatic HOMA-IR Calculation
calculated_homa = (glucose * insulin) / 405

# --- MAIN DASHBOARD ---
st.title("🏥 Diabetes Risk AI Advisor")
st.write(f"**Calculated HOMA-IR (Insulin Resistance Index):** {calculated_homa:.2f}")

if st.button("🚀 Run Complete Clinical Analysis"):
    # Prepare Data
    user_data = pd.DataFrame([[gender, age, glucose, bmi, waist, sleep, sedentary, calculated_homa]], columns=features)
    user_imputed = imputer.transform(user_data)
    
    # 1. AI Prediction
    risk_prob = model.predict_proba(user_imputed)[0][1]
    risk_score = risk_prob * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Risk", f"{risk_score:.1f}%")
        
        # 2. CLINICAL SUGGESTIONS (Logic with Unicode Safety)
        st.subheader("Clinical Guidance")
        
        # We define a plain text message for the PDF and a decorated one for the Web UI
        if glucose > 126 or risk_prob > 0.7:
            clean_msg = "HIGH RISK: Patient meets criteria for Diabetic screening. Recommend HbA1c test."
            st.error(f"🔴 {clean_msg}")
        elif calculated_homa > 2.5 or 100 <= glucose <= 125:
            clean_msg = "MODERATE RISK: Insulin resistance or prediabetic glucose detected. Lifestyle change advised."
            st.warning(f"🟡 {clean_msg}")
        else:
            clean_msg = "LOW RISK: Maintain healthy habits and annual checkups."
            st.success(f"🟢 {clean_msg}")

    with col2:
        st.subheader("Population Standings")
        waist_percentile = (raw_df['Waist'] < waist).mean() * 100
        st.write(f"Waist Circumference is higher than **{waist_percentile:.1f}%** of population.")
        st.progress(waist_percentile / 100)

    # 3. SHAP EXPLAINER
    st.divider()
    st.subheader("🔬 Why did the AI give this score? (SHAP Analysis)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_imputed)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.bar_plot(shap_values[0][:, 1], feature_names=features, show=False)
    st.pyplot(plt.gcf())

    # 4. PDF REPORT GENERATOR (Uses clean_msg to avoid Emoji errors)
    st.divider()
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(200, 10, "NHANES AI Clinical Report", ln=True, align='C')
        pdf.set_font("Helvetica", size=12)
        pdf.ln(10)
        pdf.cell(200, 10, f"Date: {datetime.date.today()}", ln=True)
        pdf.cell(200, 10, f"AI Risk Prediction: {risk_score:.1f}%", ln=True)
        pdf.ln(5)
        pdf.cell(200, 10, f"Glucose: {glucose} mg/dL | HOMA-IR: {calculated_homa:.2f}", ln=True)
        pdf.cell(200, 10, f"BMI: {bmi} | Age: {age}", ln=True)
        pdf.ln(10)
        pdf.set_font("Helvetica", 'I', 12)
        # Using clean_msg which contains no emojis
        pdf.multi_cell(0, 10, f"Recommendation: {clean_msg}")
        return pdf.output()

    try:
        pdf_report = generate_pdf()
        st.download_button(
            label="📥 Download Patient PDF Report", 
            data=bytes(pdf_report), 
            file_name=f"Report_{datetime.date.today()}.pdf", 
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"PDF Error: {e}")