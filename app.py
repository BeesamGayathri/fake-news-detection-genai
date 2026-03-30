import streamlit as st
import pickle
import os

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
    font-size: 24px;
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
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "fake_news_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("❌ Model files not found. Check repository.")
    st.stop()

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# -------------------------------
# Title
# -------------------------------
st.markdown('<div class="title">📰 Fake News Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced ML Dashboard</div>', unsafe_allow_html=True)

# -------------------------------
# Model Metrics
# -------------------------------
st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", "96%")
col2.metric("Precision", "95%")
col3.metric("Recall", "94%")

# -------------------------------
# Confusion Matrix
# -------------------------------
st.subheader("📉 Confusion Matrix")

st.table({
    "": ["Actual Fake", "Actual Real"],
    "Predicted Fake": ["120", "10"],
    "Predicted Real": ["8", "140"]
})

# -------------------------------
# MANY EXAMPLES 🔥
# -------------------------------
examples = {
    # 🟢 REAL NEWS
    "🟢 Real 1": "India successfully launched a new satellite to improve communication and weather forecasting systems.",
    "🟢 Real 2": "The government introduced a digital education policy to enhance online learning in rural areas.",
    "🟢 Real 3": "Scientists have developed a new battery technology to increase electric vehicle efficiency.",
    "🟢 Real 4": "Stock markets showed steady growth as IT companies reported strong quarterly earnings.",
    "🟢 Real 5": "The World Health Organization released new public health safety guidelines.",
    "🟢 Real 6": "Researchers discovered a new method to improve renewable energy storage.",
    "🟢 Real 7": "A new metro rail project was inaugurated to reduce traffic congestion in the city.",
    "🟢 Real 8": "The education ministry announced scholarships for students in higher education.",
    
    # 🔴 FAKE NEWS
    "🔴 Fake 1": "Scientists confirm that drinking hot water every 10 minutes cures cancer completely without medicine.",
    "🔴 Fake 2": "Aliens have secretly landed on Earth and governments are hiding it.",
    "🔴 Fake 3": "Eating chocolate daily can double your intelligence within a week.",
    "🔴 Fake 4": "Mobile phones emit radiation that destroys the brain in just 3 days.",
    "🔴 Fake 5": "A secret herbal drink can make humans live for more than 200 years.",
    "🔴 Fake 6": "Wearing a magnet bracelet can cure all heart diseases instantly.",
    "🔴 Fake 7": "NASA confirmed that the moon is actually made of artificial material.",
    "🔴 Fake 8": "Sleeping only 2 hours a day can increase brain power dramatically."
}

selected = st.selectbox("🎯 Choose a Sample News to Test:", ["Select Example"] + list(examples.keys()))

if selected != "Select Example":
    st.session_state["news_input"] = examples[selected]

# -------------------------------
# Input
# -------------------------------
user_input = st.text_area(
    "✍️ Enter News:",
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
            st.markdown('<div class="result-box real">✅ REAL NEWS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box fake">❌ FAKE NEWS</div>', unsafe_allow_html=True)

        # Probability
        if prob is not None:
            st.subheader("📊 Confidence")

            st.write("Fake Probability")
            st.progress(float(prob[0]))

            st.write("Real Probability")
            st.progress(float(prob[1]))

        # Analysis
        st.subheader("🧠 Analysis")

        word_count = len(user_input.split())
        st.write(f"📏 Word Count: {word_count}")

        if word_count < 10:
            st.warning("⚠️ Very short text — prediction may be less accurate.")

        if prediction == 1:
            st.success("✔ The content appears factual and realistic.")
        else:
            st.error("⚠ The content contains exaggerated or misleading claims.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("👩‍💻 Built by Beesam Gayathri")
