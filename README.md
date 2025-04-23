# 📧 Email Spam Detection

A simple machine learning project that classifies emails as **Spam** or **Ham** using **Logistic Regression** and **TF-IDF vectorization**. Built with Python and scikit-learn.

---

## 🚀 Features

- Preprocesses and cleans email text data
- Converts text to numerical features using TF-IDF
- Trains a Logistic Regression model
- Predicts if a given message is spam or not
- Simple and beginner-friendly code

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- TfidfVectorizer
- LogisticRegression

---

## 📂 Dataset

The dataset `mail_data.csv` contains:
- `Category`: Label (`spam` or `ham`)
- `Message`: Email/message text

---

## 📌 Example Usage

```python
input = ["🎉 You've won $500! Click now to claim your prize."]
input_data_features = feature_extraction.transform(input)
prediction_on_input = model.predict(input_data_features)

if prediction_on_input[0] == 0:
    print('Spam')
else:
    print('Ham')
```
## 📈 Output
Spam

---

Let me know if you want to include installation instructions or a screenshot!
