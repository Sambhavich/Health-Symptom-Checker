# ============================================================
# HOSPITAL-GRADE AI DOCTOR SYSTEM (FINAL UPGRADED VERSION)
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import spacy
import plotly.express as px
import os
from datetime import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Hospital Assistant",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# HOSPITAL UI DESIGN
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0b1220, #020617);
    color: white;
}

/* HEADER */
.title {
    font-size: 54px;
    text-align: center;
    font-weight: 900;
    background: linear-gradient(90deg,#00e5ff,#7c3aed,#22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* GLASS CARD */
.card {
    background: rgba(255,255,255,0.05);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    margin-bottom: 15px;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg,#00e5ff,#7c3aed);
    color: white;
    border-radius: 10px;
    font-weight: 700;
}

/* METRIC */
.metric {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "final_validated_model.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "models", "label_encoder.pkl"))
mlb = joblib.load(os.path.join(BASE_DIR, "models", "symptom_encoder.pkl"))

nlp = spacy.load("en_core_web_sm")
symptom_list = list(mlb.classes_)

# -----------------------------
# SESSION
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "chat" not in st.session_state:
    st.session_state.chat = []

# -----------------------------
# SYMPTOM EXTRACTION (FIXED MULTI-SYMPTOM BUG)
# -----------------------------
def extract_symptoms(text):
    doc = nlp(text.lower())

    # FIX: proper phrase detection (not single token only)
    text = " ".join([t.text for t in doc])

    detected = []
    for s in symptom_list:
        if s in text:
            detected.append(s)

    return list(set(detected))

# -----------------------------
# PREDICTION + RISK SCORE SYSTEM
# -----------------------------
def predict(symptoms):
    if not symptoms:
        return None, 0, 0

    X = pd.DataFrame(mlb.transform([symptoms]), columns=mlb.classes_)

    pred = model.predict(X)
    disease = label_encoder.inverse_transform(pred)[0]

    confidence = 0
    if hasattr(model, "predict_proba"):
        confidence = round(model.predict_proba(X).max() * 100, 2)

    # -----------------------------
    # RISK SCORE ENGINE (NEW)
    # -----------------------------
    risk_score = min(100, len(symptoms) * 20 + confidence * 0.5)

    return disease, confidence, round(risk_score, 2)

# -----------------------------
# CHATBOT (FIXED: NOW USES FULL SYMPTOM LIST)
# -----------------------------
def doctor_bot(msg):
    symptoms = extract_symptoms(msg)

    if len(symptoms) == 0:
        return "❌ Please describe multiple symptoms clearly (fever, cough, headache, fatigue etc.)"

    disease, conf, risk = predict(symptoms)

    # ADVICE ENGINE
    if risk < 30:
        advice = "🟢 Low risk — Rest and hydration recommended"
    elif risk < 60:
        advice = "🟡 Moderate risk — Monitor symptoms closely"
    else:
        advice = "🔴 High risk — Consult a doctor immediately"

    return f"""
🏥 AI MEDICAL ANALYSIS REPORT

🔎 Symptoms Detected:
{', '.join(symptoms)}

🧠 Disease Prediction: {disease}

📊 Confidence: {conf}%

⚠️ Risk Score: {risk}/100

💡 Medical Advice:
{advice}

⚠️ This is NOT a medical diagnosis.
"""

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
mode = st.sidebar.radio(
    "🏥 HOSPITAL SYSTEM",
    ["🏠 Dashboard", "📝 Diagnosis", "🩹 Symptom Picker", "📊 Analytics", "🤖 AI Doctor"]
)

st.markdown('<div class="title">🏥 AI HOSPITAL DIAGNOSTIC SYSTEM</div>', unsafe_allow_html=True)

# ============================================================
# DASHBOARD (HOSPITAL GRADE HOME)
# ============================================================
if mode == "🏠 Dashboard":

    col1, col2, col3 = st.columns(3)

    col1.metric("🧠 AI Engine", "ML + NLP")
    col2.metric("📊 Patients Analyzed", len(st.session_state.history))
    col3.metric("🤖 Doctor Bot", "ACTIVE")

    st.markdown("<div class='card'>📌 System Overview</div>", unsafe_allow_html=True)

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(df["Prediction"].value_counts(),
                         title="Disease Cases Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.line(df, y="Risk", markers=True,
                           title="Patient Risk Trend")
            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("No patient data yet — start diagnosis")

# ============================================================
# NLP DIAGNOSIS
# ============================================================
elif mode == "📝 Diagnosis":

    text = st.text_area("Enter patient symptoms")

    if st.button("Analyze"):
        symptoms = extract_symptoms(text)
        disease, conf, risk = predict(symptoms)

        st.success(f"Disease: {disease}")
        st.info(f"Confidence: {conf}%")
        st.warning(f"Risk Score: {risk}/100")

        st.session_state.history.append({
            "Time": datetime.now(),
            "Input": text,
            "Prediction": disease,
            "Confidence": conf,
            "Risk": risk
        })

# ============================================================
# SYMPTOM PICKER
# ============================================================
elif mode == "🩹 Symptom Picker":

    selected = st.multiselect("Select symptoms", symptom_list)

    if st.button("Diagnose"):
        disease, conf, risk = predict(selected)

        st.success(disease)
        st.info(f"Confidence: {conf}%")
        st.warning(f"Risk Score: {risk}/100")

# ============================================================
# ANALYTICS
# ============================================================
elif mode == "📊 Analytics":

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        st.markdown("### 📊 Hospital Analytics")

        st.plotly_chart(px.bar(df["Prediction"].value_counts(),
                               title="Disease Frequency"))

        st.plotly_chart(px.line(df, y="Risk",
                                title="Risk Progression"))

        st.dataframe(df)

    else:
        st.warning("No data available")

# ============================================================
# AI DOCTOR CHATBOT (FIXED MULTI-SYMPTOM READING)
# ============================================================
elif mode == "🤖 AI Doctor":

    st.markdown("<div class='card'>💬 Talk to AI Doctor</div>", unsafe_allow_html=True)

    user_msg = st.chat_input(" Analyzing the symptoms with challenges")

    if user_msg:
        reply = doctor_bot(user_msg)

        st.session_state.chat.append(("user", user_msg))
        st.session_state.chat.append(("bot", reply))

    for role, msg in st.session_state.chat:
        if role == "user":
            with st.chat_message("user"):
                st.write(msg)
        else:
            with st.chat_message("assistant"):
                st.write(msg)
                