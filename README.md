# 🌸 Iris Flower Classification (Machine Learning)

This project implements a multi-class classification system on the classic Iris dataset using Scikit-learn.  
Multiple machine learning models were trained and evaluated, and hyperparameter tuning was applied to Logistic Regression to obtain the best-performing model.

The final tuned Logistic Regression model achieved 100% accuracy on the test set using proper preprocessing and stratified splitting.

---

## 📌 Project Overview

The goal of this project is to classify iris flowers into three species:

- Iris-setosa  
- Iris-versicolor  
- Iris-virginica  

Based on four features:

- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

---

## 🚀 Features

- Data cleaning (duplicate removal, missing value checks)
- Outlier visualization using boxplots
- Feature scaling using `StandardScaler`
- Label encoding of target variable
- Model comparison:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
- Hyperparameter tuning using `GridSearchCV`
- Model evaluation using:
  - Accuracy  
  - Classification Report  
  - Confusion Matrix  
- Model saving using `pickle`

---

## 📊 Results (Best Model)

**Model:** Tuned Logistic Regression  

Accuracy: 1.00 (100%)

Confusion Matrix:
[[10 0 0]
[ 0 10 0]
[ 0 0 10]]

---

## 🧠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## ⚙️ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/iris-flower-classification.git
cd iris-flower-classification

### 2️⃣ Install dependencies
pip install -r requirements.txt

### 3️⃣ Run the training script
python train.py

