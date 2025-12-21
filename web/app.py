import streamlit as st
from keras.models import load_model
import numpy as np

st.set_page_config(page_title="Parkinsons AI",layout="centered")
st.markdown("""
<div style="
    background-color:#0e4c92;
    padding:20px;
    border-radius:10px;
    text-align:center;
    color:white;
">
    <h1>🧠 Parkinson Detection System</h1>
    <h4>This app uses Machine Learning to detect Parkinson's disease from voice measurements</h4>
</div>
""", unsafe_allow_html=True)

#Sidebar layout
st.sidebar.header("⚙️ Configuration")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Random Forest", "SVM", "Neural Network"]
)

predict_btn = st.sidebar.button("🔍 Predict")


uploaded_file = st.file_uploader("Upload patient data (CSV)")

st.subheader("🎤 Voice Features Input")

fo = st.number_input("MDVP:Fo (Hz)")
fhi = st.number_input("MDVP:Fhi (Hz)")
flo = st.number_input("MDVP:Flo (Hz)")
jitter = st.number_input("Jitter (%)")
shimmer = st.number_input("Shimmer")
hnr = st.number_input("HNR")
ppe = st.number_input("PPE")

model = load_model("model.h5")
if predict_btn:
    X = np.array([[fo, fhi, flo, jitter, shimmer, hnr, ppe]])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][prediction]

    if prediction == 1:
        st.error(f"🟥 Parkinson Detected (Confidence: {proba:.2f})")
    else:
        st.success(f"🟩 Healthy Voice (Confidence: {proba:.2f})")

st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f8f9fa;
    color: #333;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    border-top: 1px solid #ddd;
}
.footer a {
    margin: 0 10px;
    color: #0e4c92;
    text-decoration: none;
    font-weight: 600;
}
.footer a:hover {
    text-decoration: underline;
}
</style>

<div class="footer">
    Developed by <b>Houda Ahmitti</b> | Parkinson’s AI Project – 2025  
    <br/>
    <a href="https://www.linkedin.com/in/houda-ahmitti-568497243/" target="_blank">🔗 LinkedIn</a>
    |
    <a href="https://github.com/houdaahmitti" target="_blank">💻 GitHub</a>
</div>
""", unsafe_allow_html=True)

