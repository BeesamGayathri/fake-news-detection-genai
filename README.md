# Fake News Detection with GenAI

This project demonstrates a **Fake News Detection system** using Python, scikit-learn, and a simple machine learning model. The model predicts whether a news text is **Real News 📰** or **Fake News ❌**.

---

## 📁 Project Structure


---

## 🛠️ How to Run

1. Clone the repository or download it to your local machine.
2. Install dependencies:

```bash
pip install -r requirements.txt
predict_news("Your news text here")
## 📊 Dataset

- CSV file: `data/fake_news.csv`  
- Columns:
  - `text` → news content  
  - `label` → 1 = Real, 0 = Fake  

Example rows:

| text                                  | label |
|---------------------------------------|-------|
| Government launches new AI initiative | 1     |
| Celebrity endorses miracle diet       | 0     |
