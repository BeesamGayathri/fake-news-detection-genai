import streamlit as st
import pickle

# -------------------------------
# Load Model & Vectorizer
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

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
        # Transform input
        transformed_input = vectorizer.transform([user_input])

        # Prediction
        prediction = model.predict(transformed_input)[0]

        # Probability (Confidence)
        try:
            probability = model.predict_proba(transformed_input)[0]
            confidence = max(probability)
        except:
            confidence = None

        # -------------------------------
        # Output Result
        # -------------------------------
        if prediction == 1:
            st.success("✅ This is Real News")
        else:
            st.error("❌ This is Fake News")

        # Confidence Score
        if confidence:
            st.info(f"🔍 Confidence: {confidence:.2f}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("👩‍💻 Built by **Beesam Gayathri**")
