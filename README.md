
# Customer Churn Prediction

This project focuses on predicting customer churn using a telecom dataset. It leverages several machine learning algorithms to evaluate performance and identify the most effective model for predicting whether a customer will leave the service.

This project is a re-implementation of the research done by Kahti _et al.,_ 2024.

The report can be found [Here](https://www.researchgate.net/publication/382334373_An_analysis_on_classification_models_for_customer_churn_prediction)

---

## Project Overview

Customer churn is a critical problem in the telecom industry. Accurately predicting churn enables businesses to take preemptive action to retain customers. In this project, multiple classification models are trained and compared using cross-validation. The best-performing model is further evaluated with confusion matrix and ROC analysis.

---

## 2. Data Source

- **File**: [WA_Fn-UseC_-Telco-Customer-Churn.csv](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Observations**: 7,043 customer records
- **Features**:
 0   customerID        7043 non-null   object 
 1   gender            7043 non-null   object 
 2   SeniorCitizen     7043 non-null   int64  
 3   Partner           7043 non-null   object 
 4   Dependents        7043 non-null   object 
 5   tenure            7043 non-null   int64  
 6   PhoneService      7043 non-null   object 
 7   MultipleLines     7043 non-null   object 
 8   InternetService   7043 non-null   object 
 9   OnlineSecurity    7043 non-null   object 
 10  OnlineBackup      7043 non-null   object 
 11  DeviceProtection  7043 non-null   object 
 12  TechSupport       7043 non-null   object 
 13  StreamingTV       7043 non-null   object 
 14  StreamingMovies   7043 non-null   object 
 15  Contract          7043 non-null   object 
 16  PaperlessBilling  7043 non-null   object 
 17  PaymentMethod     7043 non-null   object 
 18  MonthlyCharges    7043 non-null   float64
 19  TotalCharges      7043 non-null   object 
- **Target**: `Churn` (Yes = churned, No = retained)

---

## 3. 🧰 Tools and Libraries

- `pandas`, `numpy` – data processing
- `seaborn`, `matplotlib` – visualization
- `scikit-learn` – machine learning and evaluation
- `xgboost` – boosted gradient tree classifier
- `imblearn` – oversampling using SMOTE

---

## 4. 📊 Exploratory Data Analysis (EDA)

The EDA includes:
- Viewing dataset structure using `df.info()` and `.head()`
- Identifying missing or inconsistent values
- Visualizing churn by contract type, tenure, and payment method
- Confirming that `TotalCharges` can be derived from `MonthlyCharges` and `tenure`

---

## 5. 🧹 Data Processing

Steps taken:
- Converted `TotalCharges` to numeric
- Removed rows with missing values
- Dropped `customerID` as it is not useful for prediction
- Categorical features were encoded appropriately
- Class imbalance was handled using **SMOTE** to oversample the minority class

---

## 6. 🛠️ Feature Engineering

- Removed `TotalCharges` due to redundancy (correlated with `tenure` × `MonthlyCharges`)
- Created dummy variables for categorical features
- Scaled features using `StandardScaler` for certain models

---

## 7. 🧠 Model Approach

### Models Trained:
- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- **Random Forest**
- AdaBoost
- Gradient Boosting
- XGBoost

### Cross-Validation:
- Used `StratifiedKFold` with 5 splits
- Trained models on balanced data from SMOTE
- Collected and compared evaluation metrics: accuracy, precision, recall, F1, F2

---

## 8. 📈 Model Evaluation

### 🔥 Best Performing Model: **Random Forest**

**Confusion Matrix:**
```
                 Predicted
               |   0   |   1   |
           ---------------------
Actual |  0  | 4404 |  770 |
       |  1  |  760 | 4414 |
```

### 🔢 Metrics:
| Metric     | Value    |
|------------|----------|
| Accuracy   | 0.8560   |
| Precision  | 0.8588   |
| Recall     | 0.8527   |
| F1 Score   | 0.8555   |
| F2 Score   | 0.8538   |

- The ROC curve was plotted using out-of-fold predictions.
- Random Forest showed the best overall balance of precision and recall.

---

## ✅ Conclusion

- Random Forest was selected as the best model due to its high accuracy and F1 score.
- The project successfully demonstrates how to prepare data, balance classes, train multiple classifiers, and evaluate them robustly using cross-validation.

---

## 📌 Future Improvements

- Perform hyperparameter tuning using GridSearchCV
- Use SHAP values for feature importance and explainability
- Deploy the model using Flask or Streamlit
