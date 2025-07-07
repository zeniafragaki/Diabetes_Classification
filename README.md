#  Diabetes Prediction using Machine Learning (SVM)

This MATLAB project implements a machine learning model that predicts whether a person has diabetes based on health and lifestyle features. The model uses a **Support Vector Machine (SVM)** for classification.



##  Dataset

The dataset is sourced from Kaggle:

**Diabetes Prediction Dataset** by iammustafatz  
🔗 [https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

It contains medical and demographic data—such as age, gender, BMI, hypertension, heart disease, smoking history, HbA1c level, and blood glucose level—along with a diabetes diagnosis label (positive or negative).



## Files Included

- `main_script.m` – Main MATLAB script for data loading, preprocessing, model training, evaluation, and visualization  
- `evaluation.m` – Function to compute Accuracy, R² score, and RMSE  
- `plot_classif.m` – Function to generate the confusion matrix plot  
- `diabetes_dataset1.xlsx` – Excel version of the dataset (you must download and place this manually)

---

##  Framework

1. **Data Loading**  
2. **Preprocessing**  
   - Encode categorical variables (e.g., gender, smoking history) into numerical values  
   - Normalize all features to the range [-1, 1]  
3. **Data Splitting**  
   - 70% training set  
   - 30% testing set  
4. **Model Training**  
   - Train an SVM classifier on the training set  
5. **Prediction**  
   - Predict diabetes status on the test set  
6. **Evaluation**  
   - Accuracy  
   - R² Score  
   - RMSE  
   - Confusion Matrix plot

---

## Instructions 

1. **Clone the repository**

    ```bash
    git clone https://github.com/your-username/diabetes-prediction-ml.git
    cd diabetes-prediction-ml
    ```

2. **Download the dataset**

    - Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)  
    - Download `diabetes_prediction_dataset.csv` or `.xlsx`  
    - Rename it to `diabetes_dataset1.xlsx`  
    - Place it in the project folder

3. **Run in MATLAB**

    - Open MATLAB  
    - Navigate to the project folder  
    - Run the main script:

      ```matlab
      main_script
      ```

4. **Requirements**

    - MATLAB 
 

