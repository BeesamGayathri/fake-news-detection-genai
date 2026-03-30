import streamlit as st
import pickle
import os
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
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
# Load Model
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "fake_news_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("❌ Model files not found. Please ensure fake_news_model.pkl and vectorizer.pkl are in the repository.")
    st.stop()

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# -------------------------------
# Title
# -------------------------------
st.markdown('<div class="title">📰 Fake News Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered system with ML insights</div>', unsafe_allow_html=True)

st.write("")

# -------------------------------
# Examples
# -------------------------------
examples = {
    "🟢 Real News": "India’s space agency successfully launched a new satellite to improve communication.",
    "🔴 Fake News": "Drinking hot water every 10 minutes cures all cancer instantly."
}

selected = st.selectbox("🎯 Try Sample:", ["Select"] + list(examples.keys()))

if selected != "Select":
    st.session_state["news_input"] = examples[selected]

# -------------------------------
# Input
# -------------------------------
user_input = st.text_area(
    "✍️ Enter News Content:",
    value=st.session_state.get("news_input", ""),
    height=180
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🚀 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Enter text")
    else:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]

        try:
            prob = model.predict_proba(transformed)[0]
        except:
            prob = None

        # Result
        if prediction == 1:
            st.markdown(
                f'<div class="result-box real">✅ REAL NEWS</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box fake">❌ FAKE NEWS</div>',
                unsafe_allow_html=True
            )

        # -------------------------------
        # 📊 Probability Chart (NEW 🔥)
        # -------------------------------
        if prob is not None:
            st.subheader("📊 Prediction Confidence")

            labels = ["Fake", "Real"]

            fig, ax = plt.subplots()
            ax.bar(labels, prob)
            ax.set_ylabel("Probability")
            ax.set_title("Model Confidence")

            st.pyplot(fig)

# -------------------------------
# 🧠 Model Info Section (NEW)
# -------------------------------
with st.expander("🧠 Model Details"):
    st.write("""
    - Algorithm Used: Logistic Regression / Naive Bayes  
    - Feature Extraction: TF-IDF Vectorization  
    - Type: Supervised Machine Learning  
    """)

# -------------------------------
# Footer
# -------------------------------
st.markdown('<div class="footer">👩‍💻 Built by Beesam Gayathri</div>', unsafe_allow_html=True)
