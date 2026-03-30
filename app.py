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
st.markdown('<div class="subtitle">Hybrid AI with Confidence-Based Logic</div>', unsafe_allow_html=True)

# -------------------------------
# Examples
# -------------------------------
examples = {

    # 🟢 REAL NEWS
    "🟢 Real 1": "The government announced a new policy to improve digital education in rural areas.",
    "🟢 Real 2": "India successfully launched a communication satellite from Sriharikota.",
    "🟢 Real 3": "Scientists published research on renewable energy efficiency in a leading journal.",
    "🟢 Real 4": "Stock markets increased as major IT companies reported higher profits.",
    "🟢 Real 5": "The health ministry issued new safety guidelines to prevent disease spread.",

    # 🔴 FAKE NEWS
    "🔴 Fake 1": "Scientists confirm a miracle cure that can eliminate all diseases instantly without medicine.",
    "🔴 Fake 2": "Aliens have officially taken control of Earth and governments are hiding it.",
    "🔴 Fake 3": "Eating one chocolate daily can increase intelligence by 200 percent instantly.",
    "🔴 Fake 4": "A secret formula allows humans to live up to 300 years without aging.",
    "🔴 Fake 5": "Drinking a special juice for 3 days can cure cancer completely without any treatment."
}

selected = st.selectbox("🎯 Choose Sample News:", ["Select Example"] + list(examples.keys()))

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
        # ML Prediction
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]

        try:
            prob = model.predict_proba(transformed)[0]
            confidence = max(prob)
        except:
            prob = None
            confidence = 1

        # Rule-based keywords (STRICT)
        fake_keywords = [
            "miracle cure",
            "cure all diseases",
            "cure cancer instantly",
            "without any treatment",
            "live 300 years",
            "aliens control earth",
            "100% guarantee",
            "double intelligence instantly"
        ]

        text_lower = user_input.lower()
        rule_flag = any(keyword in text_lower for keyword in fake_keywords)

        # SMART HYBRID LOGIC
        if rule_flag and confidence < 0.70:
            prediction = 0  # Override only when model is unsure

        # Result
        if prediction == 1:
            st.markdown('<div class="result-box real">✅ REAL NEWS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box fake">❌ FAKE NEWS</div>', unsafe_allow_html=True)

        # Show rule trigger
        if rule_flag and confidence < 0.70:
            st.warning("⚠ Suspicious pattern detected (low confidence override applied)")

        # Confidence
        if prob is not None:
            st.subheader("📊 Confidence")

            st.write(f"Overall Confidence: {confidence:.2f}")

            st.write("Fake Probability")
            st.progress(float(prob[0]))

            st.write("Real Probability")
            st.progress(float(prob[1]))

        # Analysis
        st.subheader("🧠 Analysis")

        word_count = len(user_input.split())
        st.write(f"📏 Word Count: {word_count}")

        if word_count < 10:
            st.warning("⚠ Short text may reduce accuracy")

        if prediction == 1:
            st.success("✔ Content appears factual and realistic.")
        else:
            st.error("⚠ Content contains exaggerated or misleading claims.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("👩‍💻 Built by Beesam Gayathri")
