# 📰 Fake News Detection using Machine Learning

## 📌 Overview
This project aims to classify news articles as **Real or Fake** using Machine Learning and Natural Language Processing (NLP) techniques.

In today’s digital world, misinformation spreads rapidly. This system helps users verify the authenticity of news content quickly and efficiently.

---

## 🚀 Features
- 🧹 Text preprocessing (cleaning, stopword removal, tokenization)
- 🔍 Feature extraction using **TF-IDF Vectorization**
- 🤖 Machine Learning-based classification model
- 🌐 Interactive **Streamlit web application**
- ⚡ Real-time prediction from user input

---

## 🛠️ Tech Stack
- **Programming:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **NLP:** TF-IDF Vectorizer  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment:** Streamlit  
- **Version Control:** Git, GitHub  

---

## 📊 Project Workflow

1. **Data Collection**
   - Dataset containing labeled real and fake news articles  

2. **Data Preprocessing**
   - Removed punctuation and special characters  
   - Converted text to lowercase  
   - Removed stopwords  

3. **Feature Engineering**
   - Converted text into numerical format using TF-IDF  

4. **Model Training**
   - Applied supervised machine learning algorithms  
   - Evaluated model performance using accuracy  

5. **Deployment**
   - Built a user-friendly interface using Streamlit  
   - Enabled real-time predictions  

---

## 💡 How It Works
- User enters a news article 📝  
- The system preprocesses the text  
- The trained model predicts:  
  - ✅ **Real News**  
  - ❌ **Fake News**

---

## 🌐 Live Demo
👉 https://fake-news-detection-genai-affemcburdaahsrthtucvy.streamlit.app/

---

## 📂 Project Structure

fake-news-detection-genai/  
│── app.py                 # Streamlit application  
│── model.pkl              # Trained ML model  
│── vectorizer.pkl         # TF-IDF vectorizer  
│── dataset.csv            # Dataset  
│── requirements.txt       # Dependencies  
│── README.md              # Documentation  

---

## ▶️ Installation & Usage

### 1. Clone the Repository
git clone https://github.com/BeesamGayathri/fake-news-detection-genai.git  
cd fake-news-detection-genai  

### 2. Install Dependencies
pip install -r requirements.txt  

### 3. Run the Application
streamlit run app.py  

---

## 📈 Future Improvements
- 🔄 Implement Deep Learning models (LSTM, BERT)  
- 🌍 Support for multiple languages  
- 📊 Display prediction confidence score  
- 🔗 Integrate real-time news APIs  

---

## 📫 Contact
- **GitHub:** https://github.com/BeesamGayathri  
- **LinkedIn:** https://linkedin.com/in/beesam-gayathri  

---

## ⭐ Support
If you found this project useful, please consider giving it a ⭐ on GitHub!
