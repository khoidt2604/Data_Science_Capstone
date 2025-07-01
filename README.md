# 🎯 Model Comparison for Student Dropout Prediction

This repository contains Python code and analysis for comparing multiple machine learning models to predict student dropout. The goal is to identify the most effective model using validation and test accuracy as key metrics.

---

## 📌 Introduction

This project evaluates different classification models for predicting student dropout using a dataset that includes demographic and academic attributes. By training and validating multiple models, the project identifies the most reliable classifier for accurate and generalisable predictions.

The primary aim is to assist educators and institutions in making data-driven decisions to support students at risk of dropping out.

---

## 💡 Motivation

With increasing concerns around student attrition in educational institutions, developing predictive systems can be an effective early intervention strategy. This project explores and compares the performance of several supervised learning models to determine the most suitable algorithm for identifying at-risk students.

---

## 📊 Key Visualisations

### 1. Accuracy Comparison of Models

This bar chart displays both validation and test accuracy for four classification models:

![Model Accuracy Comparison](Comparison_model_accuracy.png)

> The models included in this comparison are:
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors (KNN)

> As seen above, the **Decision Tree** model achieved the highest accuracy on both validation and test sets, suggesting better generalisability and predictive power.

### 2. Accuracy Scores Table

| Model                  | Validation Accuracy | Test Accuracy |
|------------------------|---------------------|---------------|
| Logistic Regression    | 0.8464              | 0.8333        |
| Support Vector Machine | 0.8482              | 0.8333        |
| Decision Tree          | **0.8893**          | **0.8889**    |
| K–Nearest Neighbor     | 0.8482              | 0.8333        |

The **Decision Tree** classifier performed best, with both the highest validation and test accuracy, making it the most robust model in this analysis.

---

### 🔹 3. Success Rate by Orbit Type

This chart shows the success rate of missions by orbit type.  
Orbits such as GEO, HEO, and SSO achieved a **100% success rate**, while others like GTO and ISS have lower reliability.

![Success Rate by Orbit Type](Success_rate_orbit.png)

---

### 🔹 4. Flight Number vs Orbit Type (Class Outcome)

The scatter plot visualises flight numbers across different orbit types, color-coded by class (success or failure).  
This helps identify trends or irregularities in outcomes based on flight order and orbit type.

![Flight Number vs Orbit](Flight_number_orbit.png)

## 🤖 Models Used

- Logistic Regression  
- Support Vector Machine (SVM)  
- Decision Tree  
- K-Nearest Neighbors (KNN)  

Each model was trained using cross-validation and evaluated using accuracy metrics on both validation and test datasets.

---

## 📂 Files

- `model_comparison.ipynb` — Jupyter Notebook with all code for preprocessing, model training, validation, and visualisation.  
- `bar_chart.png` — Visualisation comparing validation and test accuracy across models.  
- `README.md` — Project summary and usage instructions.

---

## ▶️ How to Run the Code

1. Open the `model_comparison.ipynb` file using Jupyter Notebook or JupyterLab.
2. Run all cells in the notebook to:
   - Preprocess the dataset
   - Train and validate each model
   - Compare accuracy results
   - Display summary tables and charts
3. Make sure you have the required packages installed. You can install them using:

   ```bash
   pip install pandas numpy matplotlib scikit-learn
