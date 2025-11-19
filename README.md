# Diabetes Prediction: Multi-Model ML Benchmark

This MATLAB project implements a comparative study of machine learning models to predict whether a person has diabetes. It focuses on handling **Class Imbalance** using automated hyperparameter tuning and cost-sensitive learning.

The project benchmarks four distinct classifiers: **SVM (RBF)**, **Random Forest**, **MLP Neural Network**, and **RBF (K-Means)**.

*All algorithms have been designed in MATLAB R2024b @ AMD Ryzen 5 7600X, 16GB RAM with Windows 11.*

##  Dataset

The dataset is sourced from Kaggle:

**Diabetes Prediction Dataset** by iammustafatz  
Source: [https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

It contains medical and demographic data—such as age, gender, BMI, hypertension, heart disease, smoking history, HbA1c level, and blood glucose level—along with a diagnosis label.

##  Files Included

- `diabetes_benchmark_optimized.m` – The main all-in-one script. Handles data loading, auto-tuning, training 4 models, and plotting results.
- `diabetes_dataset.xlsx` – Excel version of the dataset (you must download and place this manually).

##  Framework

1. **Data Loading & Processing**
   - Encodes categorical variables (Gender, Smoking History) to numerical scales.
   - Normalizes all features to the range `[-1, 1]`.

2. **Data Splitting (70-15-15)**
   - **70% Training:** Used to train the models.
   - **15% Validation:** Used for **Automated Grid Search** to find the optimal Class Weight (Balance Factor).
   - **15% Testing:** Used for the final unbiased performance evaluation.

3. **Automated Hyperparameter Tuning**
   - The script automatically hunts for the best "Balance Factor" (weight applied to the Diabetic class) to maximize the F1-Score, ensuring the model doesn't ignore the minority class.

4. **Multi-Model Training**
   - **SVM:** RBF Kernel with optimized weights.
   - **Random Forest:** 50 Trees with a cost matrix to penalize false negatives.
   - **MLP:** Feedforward Neural Network (2 hidden layers: [20, 10]).
   - **RBF Network:** Hybrid model using K-Means clustering (`K=50`) and linear SVM.

5. **Evaluation**
   - Accuracy, Recall (Sensitivity), Precision, and F1-Score.
   - Confusion Matrix visualization for all models.

## Benchmark Results

*Typical performance on the held-out Test Set:*

| Model | Accuracy | Recall (Sensitivity) | F1-Score | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **MLP (Neural Net)** | 93.62% | **84.12%** | 0.6972 | 🏆 **Best Overall:** High safety & precision. |
| **SVM (RBF Kernel)** | 93.01% | 84.05% | 0.6775 | **High Safety:** Very similar to MLP. |
| **Random Forest** | **96.33%** | 74.20% | **0.7795** | **High Precision:** Best accuracy, but misses cases. |
| **RBF (K-Means)** | 91.50% | 81.30% | 0.6256 | **Baseline:** Good sensitivity. |

##  Instructions

1. **Clone the repository**

    ```bash
    git clone [https://github.com/your-username/diabetes-prediction-ml.git](https://github.com/your-username/diabetes-prediction-ml.git)
    cd diabetes-prediction-ml
    ```

2. **Download the dataset**

    - Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
    - Download the file and convert/save it as `diabetes_dataset.xlsx`.
    - Place it in the project folder.

3. **Run in MATLAB**

    - Open MATLAB.
    - Navigate to the project folder.
    - Run the benchmark script:

      ```matlab
      diabetes_benchmark_optimized
      ```

4. **Requirements**
    - MATLAB R2021a or later (for `fitcnet` and `confusionchart`).
    - Statistics and Machine Learning Toolbox.
