import streamlit as st
import pickle
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

# -------------------------------
# Custom CSS (🔥 UI Upgrade)
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.stTextArea textarea {
    font-size: 16px;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #00C9A7;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #A0A0A0;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
.real {
    background-color: #1E5631;
    color: #C7F9CC;
}
.fake {
    background-color: #5A1E1E;
    color: #FFC9C9;
}
.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model (FIXED PATH)
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "fake_news_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# -------------------------------
# Title Section
# -------------------------------
st.markdown('<div class="title">📰 Fake News Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered system to detect Real vs Fake news</div>', unsafe_allow_html=True)

st.write("")

# -------------------------------
# Sample Inputs
# -------------------------------
real_example = "India’s space agency successfully launched a new satellite to improve communication and weather forecasting systems."

fake_example = "Drinking hot water every 10 minutes can completely cure all types of cancer without any medical treatment."

col1, col2 = st.columns(2)

with col1:
    if st.button("🟢 Try Real News"):
        st.session_state["news_input"] = real_example

with col2:
    if st.button("🔴 Try Fake News"):
        st.session_state["news_input"] = fake_example

# -------------------------------
# Input Box
# -------------------------------
user_input = st.text_area(
    "✍️ Enter News Content:",
    value=st.session_state.get("news_input", ""),
    height=180
)

# -------------------------------
# Predict Button
# -------------------------------
if st.button("🚀 Analyze News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]

        try:
            probability = model.predict_proba(transformed_input)[0]
            confidence = max(probability)
        except:
            confidence = None

        # -------------------------------
        # Styled Output
        # -------------------------------
        if prediction == 1:
            st.markdown(
                f'<div class="result-box real">✅ REAL NEWS<br>Confidence: {confidence:.2f}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box fake">❌ FAKE NEWS<br>Confidence: {confidence:.2f}</div>',
                unsafe_allow_html=True
            )

# -------------------------------
# Footer
# -------------------------------
st.markdown('<div class="footer">👩‍💻 Built by Beesam Gayathri</div>', unsafe_allow_html=True)
