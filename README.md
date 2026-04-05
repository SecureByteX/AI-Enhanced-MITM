# 🔐 MITM Attack Detection System using Machine Learning

## 📌 Overview

This project demonstrates how **Man-in-the-Middle (MITM) attacks** work, how they can be **prevented using HTTPS + HSTS**, and how **Machine Learning (XGBoost)** can be used to **detect suspicious network traffic**.

The system captures real network traffic, extracts meaningful features, trains a model, and provides predictions through an interactive dashboard.

---

## 🎯 Objectives

* Simulate MITM attacks in a controlled environment
* Demonstrate security using TLS and HSTS
* Build a Machine Learning model to detect attack traffic
* Provide explainable predictions using SHAP
* Visualize results through a user-friendly interface

---

## 🧩 Project Modules

### 1️⃣ Attack Simulation

* MITM attack using ARP spoofing
* Tools: Kali Linux, Ettercap, Wireshark

### 2️⃣ Security Implementation

* HTTPS (TLS encryption)
* HSTS (prevents downgrade attacks)
* Flask-based web server

### 3️⃣ Data Collection & Processing

* Packet capture using Wireshark
* Feature extraction using TShark
* Dataset creation and labeling

### 4️⃣ Machine Learning Model

* Algorithm: XGBoost
* Features:

  * Packet size
  * Source/Destination ports
  * Protocol
  * TCP length
  * TCP window size
  * Time delta

### 5️⃣ Explainable AI

* SHAP used to explain predictions
* Shows feature contribution for each decision

### 6️⃣ User Interface

* Built using Streamlit
* Displays:

  * Prediction results
  * Feature importance
  * SHAP explanations
  * Traffic summary

---

## ⚙️ Tech Stack

* Python
* XGBoost
* Scikit-learn
* Pandas, NumPy
* Streamlit
* Wireshark & TShark
* Flask

---

## 📊 Model Performance

* Accuracy: ~99%
* ROC-AUC: ~0.99
* Cross-validation: ~97%

> Note: Performance depends on dataset size and quality.

---

## 📂 Project Structure

```
MITM/
│
├── app.py                # Streamlit dashboard
├── train_model.py        # Model training script
├── model.pkl             # Trained ML model
├── scaler.pkl            # Feature scaler
├── final_dataset.csv     # Dataset
├── README.md
```

---

## 🚀 How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/mitm-detection.git
cd mitm-detection
```

---

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Train Model

```bash
python train_model.py
```

---

### 5️⃣ Run Dashboard

```bash
streamlit run app.py
```

---

## 📈 Dataset Details

* Source: Captured network traffic (PCAP files)
* Format: CSV (extracted via TShark)
* Labels:

  * 0 → Normal traffic
  * 1 → MITM attack

---

## 🔍 Key Features

✔ Real-world traffic capture
✔ Feature engineering using network data
✔ High-performance ML model
✔ Explainable AI (SHAP)
✔ Interactive dashboard

---

## ⚠️ Limitations

* Dataset is collected in a controlled environment
* Limited attack types (MITM only)
* Packet-level analysis (not full flow-based detection)

---

## 🚀 Future Improvements

* Flow-based traffic analysis
* Real-time detection system
* Integration with FastAPI
* Deployment using Docker
* Support for multiple attack types

---

## 📚 Learning Outcomes

* Network traffic analysis
* Cybersecurity fundamentals
* Machine learning in security
* Explainable AI
* Data preprocessing and feature engineering

---

## 👨‍💻 Author

**Hajay**

---

## ⭐ Acknowledgment

This project is built for learning and demonstration purposes in cybersecurity and machine learning.

