# 🧠 Diabetes Prediction using Machine Learning (SVM)

This MATLAB project implements a machine learning model that predicts whether a person has diabetes based on health and lifestyle features. The model uses a **Support Vector Machine (SVM)** for classification.

---

##  Files

- `main_script.m` – Main script for data loading, preprocessing, model training, evaluation, and visualization.
- `evaluation.m` – Function that computes Accuracy, R² (coefficient of determination), and RMSE.
- `plot_classif.m` – Function to plot the confusion matrix.
- `diabetes_dataset1.xlsx` – Dataset used for training and testing the model.

---

## Dataset

The dataset contains personal health information such as:

- Gender
- Age
- BMI
- Smoking history
- HbA1c level
- Blood glucose level
- Diabetes diagnosis (target variable)

Categorical variables like gender and smoking history are encoded into numerical format for compatibility with the model.

---

##  Framework

1. **Data Loading:** The dataset is loaded from an Excel file.
2. **Preprocessing:**
   - String features are encoded into numerical format.
   - Data is normalized to the range [-1, 1].
3. **Data Splitting:**
   - 70% training and 30% testing split.
4. **Model Training:** A Support Vector Machine is trained using the training set.
5. **Prediction:** The model predicts diabetes status on the test set.
6. **Evaluation:** The model is evaluated using:
   - Accuracy
   - R² Score
   - RMSE
   - Confusion Matrix

---

## How to Run

1. Make sure you have **MATLAB** installed.
2. Clone this repository or download the files:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction-ml.git
   cd diabetes-prediction-ml
3.Open MATLAB and navigate to the project folder.

4.Run the main_script.m file.

5.Ensure that diabetes_dataset1.xlsx is in the same folder.

6.You will see printed evaluation metrics and a confusion matrix plot.
