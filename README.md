# ml-project-premium-prediction
Health insurance premium prediction app built during a data science bootcamp.


---

````markdown
# 🏥 Premium Health Insurance Prediction App

This project is a Machine Learning-powered web app built using **Streamlit** to predict premium costs for health insurance based on user inputs such as age, dependents, employment status, and more. The app also uses dual ML models based on age segmentation for better accuracy.

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/mahima-ds/premium-prediction-app.git
cd premium-prediction-app
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app

```bash
streamlit run app/main.py
```

---

## 📁 Project Structure

```text
.
├── app/
│   ├── main.py                  # Streamlit UI
│   ├── prediction_helper.py     # Model prediction logic
│
├── artifacts/
│   ├── model_rest.joblib        # Model for age > 25
│   ├── model_young.joblib       # Model for age <= 25
│   ├── scaler_rest.joblib       # Scaler for older group
│   ├── scaler_young.joblib      # Scaler for younger group
│
├── screenshots/
│   ├── homepage.png             # App homepage screenshot
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
```

---

## 🧠 ML Pipeline Overview

* **Data Cleaning**
* **Feature Engineering**
* **Age-based Model Split**
* **Training Separate Models**

  * `model_young` (<= 25 years)
  * `model_rest` (> 25 years)
* **MinMax Scaling**
* **Streamlit Deployment**

---

## 📊 Dataset Info

The model is trained on synthetic health insurance data with the following fields:

* Age
* Gender
* Marital Status
* BMI Category
* Smoking Status
* Employment Status
* Number of Dependents
* Annual Income
* Medical History Risk Score

---


## 📜 License

This project is licensed under the Apache License 2.0. Feel free to use and modify it according to your needs.

---

## 🙌 Feedback & Support

If you encounter issues, have questions, or want to contribute, please open an issue or pull request on GitHub.

You can also reach out via LinkedIn:
👉 [Mahima Reddy Kota on LinkedIn](https://www.linkedin.com/in/mahima-reddy-kota-21a26436a/)

---














