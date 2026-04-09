# 📊 Naive Bayes Classification Dashboard

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![License](https://img.shields.io/badge/License-MIT-green)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-brightgreen?logo=rocket)](https://bayesclassifierdash.streamlit.app/)

</p>

---

## 🚀 Overview

An interactive **Machine Learning Dashboard** that demonstrates the **Naive Bayes Classification algorithm** using the Social Network Ads dataset.

This app allows users to explore how features like **Age** and **Estimated Salary** influence purchasing decisions, with real-time predictions and visualized decision boundaries.

---

## ✨ Features

* 🧠 Train a **Gaussian Naive Bayes model**
* 🎛️ Adjustable model parameters (test size, random state)
* 📊 Accuracy score & confusion matrix
* 🗺️ Decision boundary visualization (training & test sets)
* 🎯 Real-time prediction tool
* 📂 Upload your own dataset or use default dataset
* 👀 Raw dataset preview

---

## 🛠️ Tech Stack

* Python 3.x
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## 📂 Project Structure

```bash
.
├── app.py
├── Social_Network_Ads.csv
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/naive-bayes-dashboard.git
cd naive-bayes-dashboard
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

---

## 📊 Model Details

* Algorithm: **Gaussian Naive Bayes**
* Features:

  * Age
  * Estimated Salary
* Target:

  * Purchase Decision (0 / 1)

---

## 📈 Visualizations

* 🗺️ Decision boundary (Training & Test sets)
* 📊 Confusion matrix
* 📈 Accuracy metric
* 👀 Dataset preview

---

## 🧠 How It Works

1. Dataset is loaded (default or uploaded)
2. Data is split into training and test sets
3. Features are scaled using StandardScaler
4. Gaussian Naive Bayes model is trained
5. Predictions and visualizations are generated

---

## 🎯 Live Prediction

Users can input:

* Age
* Salary

The model predicts whether the user is likely to **purchase or not**.

---

## 📁 Dataset

The default dataset (`Social_Network_Ads.csv`) contains:

* User ID
* Gender
* Age
* Estimated Salary
* Purchased (Target)

---

## 🚀 Deployment

Deploy easily using **Streamlit Cloud**:

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Create a new app
4. Select your repository
5. Deploy 🎉

---

## 🔮 Future Improvements

* 📊 Add ROC Curve & AUC
* 🧠 Compare with other models (SVM, Logistic Regression)
* 📉 Hyperparameter tuning
* 📊 Interactive Plotly visualizations
* 📁 Upload custom datasets with flexible columns

---

## 👨‍💻 Author

**Chinmay V Chatradamath.**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
