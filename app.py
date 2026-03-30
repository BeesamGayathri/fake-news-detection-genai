import streamlit as st
import pickle
import os

# -------------------------------
# Load Model & Vectorizer (Safe Path)
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

# Check if files exist
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.error("❌ Model files not found. Please ensure model.pkl and vectorizer.pkl are in the repository.")
    st.stop()

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Fake News Detection", page_icon="📰")

st.title("📰 Fake News Detection App")
st.markdown("Detect whether a news article is **Real or Fake** using Machine Learning.")

# -------------------------------
# Sample Inputs
# -------------------------------
real_example = "India’s space agency successfully launched a new satellite to improve communication and weather forecasting systems."

fake_example = "Drinking hot water every 10 minutes can completely cure all types of cancer without any medical treatment."

# -------------------------------
# Buttons Layout
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🟢 Try Real Example"):
        st.session_state["news_input"] = real_example

with col2:
    if st.button("🔴 Try Fake Example"):
        st.session_state["news_input"] = fake_example

# -------------------------------
# Text Input
# -------------------------------
user_input = st.text_area(
    "✍️ Enter News Text:",
    value=st.session_state.get("news_input", ""),
    height=180
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🚀 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text")
    else:
        try:
            # Transform input
            transformed_input = vectorizer.transform([user_input])

            # Prediction
            prediction = model.predict(transformed_input)[0]

            # Confidence Score
            try:
                probability = model.predict_proba(transformed_input)[0]
                confidence = max(probability)
            except:
                confidence = None

            # Output
            if prediction == 1:
                st.success("✅ This is Real News")
            else:
                st.error("❌ This is Fake News")

            if confidence is not None:
                st.info(f"🔍 Confidence: {confidence:.2f}")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("👩‍💻 Built by **Beesam Gayathri**")
