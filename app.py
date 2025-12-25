# app.py
import streamlit as st
import pickle

# Load the model and vectorizer
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit app
st.set_page_config(page_title="Fake News Classification", page_icon="📰")
st.title("📰 Fake News Classification App")
st.write(
    "Enter the news text below and the model will predict whether it is Real or Fake."
)

# Text input
user_input = st.text_area("Enter News Text:", height=150)

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text!")
    else:
        # Transform the input using the vectorizer
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 1:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")
