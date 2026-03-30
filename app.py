import streamlit as st
import pickle
import os
import pandas as pd

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
    "🟢 Real News 1": "India’s space agency successfully launched a new satellite to improve communication and weather forecasting systems.",
    "🟢 Real News 2": "The government announced a new policy to improve digital education infrastructure in rural areas.",
    "🔴 Fake News 1": "Drinking hot water every 10 minutes can completely cure all types of cancer.",
    "🔴 Fake News 2": "Aliens have secretly landed on Earth and governments are hiding the truth."
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
        try:
            transformed = vectorizer.transform([user_input])
            prediction = model.predict(transformed)[0]

            try:
                prob = model.predict_proba(transformed)[0]
            except:
                prob = None

            # Result
            if prediction == 1:
                st.markdown(
                    '<div class="result-box real">✅ REAL NEWS</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="result-box fake">❌ FAKE NEWS</div>',
                    unsafe_allow_html=True
                )

            # -------------------------------
            # 📊 Streamlit Chart (NO matplotlib)
            # -------------------------------
            if prob is not None:
                st.subheader("📊 Prediction Confidence")

                chart_data = pd.DataFrame({
                    "Label": ["Fake", "Real"],
                    "Probability": prob
                })

                st.bar_chart(chart_data.set_index("Label"))

        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")

# -------------------------------
# Model Info
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
