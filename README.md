Here’s a **simpler, more natural and human-written** version of your README.md — perfect for a **BCA student portfolio or final-year project**. It keeps it professional but easy to read 👇

---

# ❤️ Heart Disease Risk Prediction

A **machine learning web app** that predicts the **risk of heart disease** based on your health details.
You enter some basic health info — and the app instantly shows your risk level!

---

## 🧠 Project Overview

This project combines **Machine Learning + Flask + HTML** to make health predictions.
It uses a **Logistic Regression** model trained on health data to calculate your heart disease risk.

---

## ⚙️ System Architecture

```
Frontend (index.html)
   ↓  (sends data via POST request)
Backend (Flask - server.py)
   ↓  (processes and scales input features)
ML Model (logistic_model.pkl)
   ↓  (returns prediction + probability)
User sees result on screen
```

---

## ✨ Features

* ✅ Predicts heart disease risk using **8 health parameters**
  (Age, Smoking, Cholesterol, BP, BMI, Heart Rate, Glucose)
* 🧮 Built-in **BMI Calculator**
* ⚡ **Real-time prediction** through Flask API
* 🧠 Supports multiple models (Logistic Regression, SVM, KNN, Decision Tree)
* 💻 Simple and clean web interface

---

## 🧩 Tech Stack

| Layer            | Technology            |
| ---------------- | --------------------- |
| Frontend         | HTML, CSS, JavaScript |
| Backend          | Flask (Python)        |
| Machine Learning | scikit-learn          |
| Deployment       | Render (Gunicorn)     |

---

## 🚀 How to Run Locally

```bash
# Step 1: Go inside the project folder
cd app

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the Flask server
python server.py
```

Then open your browser and go to 👉 `http://localhost:5000`

---

## ☁️ Deployment (Render)

1. Push your project to GitHub
2. Connect your GitHub repo to [Render.com](https://render.com)
3. Set up build and start commands:

```
Build Command:  cd app && pip install -r requirements.txt  
Start Command:  cd app && gunicorn -w 4 -b 0.0.0.0:$PORT server:app
```

4. Click **Deploy** — your web app will go live! 🚀

---

## 📦 Model Files

| File Name            | Description                           |
| -------------------- | ------------------------------------- |
| `logistic_model.pkl` | Main trained ML model                 |
| `scaler.pkl`         | Used for input data scaling           |
| (Optional)           | Other models: KNN, SVM, Decision Tree |

---

## 🔌 API Endpoint

### **POST** `/api/predict`

Send user data → get prediction (JSON)

**Example Request:**

```json
{
  "age": 50,
  "cigsPerDay": 0,
  "totChol": 200,
  "sysBP": 130,
  "diaBP": 85,
  "BMI": 25.5,
  "heartRate": 70,
  "glucose": 100
}
```

**Example Response:**

```json
{
  "prediction": 0,
  "probability": 0.35
}
```

👉 `prediction = 0` means **Low Risk**, `1` means **High Risk**.

---

## 🧑‍🎓 About This Project

This project is made as part of my **BCA final year** coursework.
It shows how machine learning can be used in real-world healthcare systems.

---

## ⚠️ Disclaimer

This is a **learning project**.
It’s **not a medical tool** — always consult a doctor for real medical advice.


