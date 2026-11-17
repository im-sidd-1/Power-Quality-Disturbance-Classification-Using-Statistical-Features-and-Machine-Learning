# ⚡ Power Quality Disturbance Classification Using Statistical Features and Machine Learning

## 📌 Project Overview

Power quality disturbances can affect the performance, reliability, and efficiency of electrical power systems.

This project focuses on classifying different power quality disturbances using statistical feature analysis and machine learning algorithms.

The system uses extracted statistical features from power quality signals to train multiple machine learning models and compare their classification performance.

The objective is to identify the most suitable model for accurate and reliable power quality disturbance classification.

## 🎯 Objectives

- Analyze power quality disturbance data.
- Perform data cleaning and preprocessing.
- Use statistical features for disturbance classification.
- Train and evaluate multiple machine learning algorithms.
- Compare model performance using evaluation metrics.
- Develop an interactive dashboard using Streamlit.

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning |
| XGBoost | Gradient boosting classification |
| Matplotlib / Seaborn | Data visualization |
| Streamlit | Interactive dashboard |
| Joblib / Pickle | Model storage |

## 📊 Dataset Description

The project uses a power quality disturbance dataset containing extracted statistical features.

The dataset includes:

- 750 samples after data cleaning.
- 21 input features.
- 13 class labels (0–12).
- No missing values in the cleaned dataset.

The statistical features describe different characteristics of the power quality signals and are used as inputs for machine learning models.

## 🔬 Methodology

The project follows these major steps:

### 1. Data Preprocessing
- Loaded the original dataset.
- Identified and removed invalid records.
- Removed unnecessary columns.
- Handled data inconsistencies.
- Generated a cleaned dataset for model training.

### 2. Exploratory Data Analysis
- Analyzed the distribution of disturbance classes.
- Examined feature characteristics.
- Visualized class distribution.
- Studied relationships between features.

### 3. Model Training

Multiple machine learning algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)
6. Naive Bayes
7. XGBoost

### 4. Model Evaluation

The models were evaluated and compared using classification performance metrics.

The evaluation process includes:
- Accuracy
- Confusion Matrix
- Classification Report
- Cross-validation

### 5. Hyperparameter Tuning

Random Forest hyperparameters were optimized using Grid Search Cross-Validation to improve model performance.

The best parameters obtained were:

- `n_estimators = 200`
- `max_depth = None`
- `min_samples_split = 2`
- `min_samples_leaf = 1`

The best cross-validation accuracy obtained was approximately **89.17%**.

## 📈 Model Performance Comparison

The following results were obtained during model evaluation:

| Machine Learning Model | Accuracy |
|---|---:|
| Random Forest | 88.00% |
| XGBoost | 88.00% |
| Support Vector Machine | 86.00% |
| K-Nearest Neighbors | 85.33% |
| Logistic Regression | 84.67% |
| Decision Tree | 84.67% |
| Naive Bayes | 84.67% |

**Observation:** Random Forest and XGBoost achieved the highest test accuracy of 88% among the evaluated models.

## 🖥️ Interactive Dashboard

An interactive dashboard was developed using Streamlit.

The dashboard provides functionality for:
- Exploring power quality data.
- Viewing model performance comparisons.
- Analyzing classification results.
- Understanding model predictions.

## 📁 Project Structure

```text
Power-Quality-Disturbance-Classification/
│
├── app.py
├── clean_data.py
├── eda.py
├── inspect_data.py
├── feature_analysis.py
├── compare_models.py
├── train_models.py
├── train_xgboost.py
├── tune_random_forest.py
├── permutation_analysis.py
│
├── data/
│   ├── Compressed PQ Data.xlsx
│   ├── cleaned_pq_data.csv
│   ├── model_comparison.csv
│   ├── class_distribution.png
│   ├── feature_importance.png
│   ├── permutation_importance.png
│   ├── random_forest_confusion_matrix.png
│   └── xgboost_confusion_matrix.png
│
├── models/
│   ├── tuned_random_forest.pkl
│   └── xgboost_model.pkl
│
├── requirements.txt
└── README.md
```

## 🚀 Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/im-sidd-1/Power-Quality-Disturbance-Classification-Using-Statistical-Features-and-Machine-Learning.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Power-Quality-Disturbance-Classification-Using-Statistical-Features-and-Machine-Learning
```

### Step 3: Create a Virtual Environment (Optional)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your default browser.

## 🔮 Future Scope

- Improve classification accuracy through advanced feature engineering.
- Explore deep learning techniques for disturbance classification.
- Evaluate additional machine learning algorithms.
- Integrate real-time power quality monitoring.
- Deploy the dashboard as a web application.
- Extend the system for practical power system applications.

## 👨‍💻 Author

**Siddhant Gupta**

B.Tech – Electrical and Computer Engineering (3rd Year)
KIET Group of Institutions

GitHub: [im-sidd-1](https://github.com/im-sidd-1)

---

⭐ If you find this project useful, consider giving the repository a star!
