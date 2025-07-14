
# Customer Churn Prediction

This project focuses on predicting customer churn using a telecom dataset. It leverages several machine learning algorithms to evaluate performance and identify the most effective model for predicting whether a customer will leave the service.

This project is a re-implementation of the research done by Kahti _et al.,_ 2024.

The report can be found [Here](https://www.researchgate.net/publication/382334373_An_analysis_on_classification_models_for_customer_churn_prediction)

---

## Project Overview

Customer churn is a critical problem in the telecom industry. Accurately predicting churn enables businesses to take preemptive action to retain customers. In this project, multiple classification models are trained and compared using cross-validation. The best-performing model is further evaluated with confusion matrix and ROC analysis.

---

## Data Source

- **File**: [WA_Fn-UseC_-Telco-Customer-Churn.csv](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Observations**: 7,043 customer records
- **Features**:
   1. customerID
   2. gender       
   3. SeniorCitizen  
   4. Partner 
   5. Dependents
   6. tenure
   7. PhoneService
   8. MultipleLines
   9. InternetService
   10. OnlineSecurity
   11. OnlineBackup
   12. DeviceProtection
   13. TechSupport
   14. StreamingTV 
   15. StreamingMovies 
   16. Contract
   17. PaperlessBilling
   18. PaymentMethod
   19. MonthlyCharges
   20. TotalCharges
- **Target**: `Churn` (Yes = churned, No = retained)

---

## Tools and Libraries

- `pandas`, `numpy` – data processing
- `seaborn`, `matplotlib` – visualization
- `scikit-learn` – machine learning and evaluation
- `statsmodels` – multi-collinearity test
- `imblearn` – oversampling using SMOTE

---

## Exploratory Data Analysis (EDA)

The EDA includes:
- Viewing dataset structure using `df.info()` and `.head()`
- Identifying missing or inconsistent values
- Visualizing churn by contract type, tenure, and payment method

---

## Data Processing

Steps taken:
- Converted `TotalCharges` to numeric
- Dropped `customerID` as it is not useful for prediction
- Categorical features were encoded appropriately
- Class imbalance was handled using **SMOTE** to oversample the minority class
- Feature scaling using **StandardScaler**

---

## Feature Engineering

- Removed `TotalCharges` due to redundancy (correlated with `tenure` × `MonthlyCharges`)
- Created dummy variables for categorical features
- Scaled features using `StandardScaler` for certain models

---

## Model Approach

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
- Used `StratifiedKFold` with 10 splits
- Trained models on balanced data from SMOTE
- Collected and compared evaluation metrics: accuracy, precision, recall, F1, F2

---

## Model Evaluation

### Best Performing Model: **Random Forest**

**Confusion Matrix:**
```
                 Predicted
               |   0   |   1   |
           ---------------------
Actual |  0  | 4404 |  770 |
       |  1  |  760 | 4414 |
```

### Metrics:
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

## Conclusion

- Random Forest was selected as the best model due to its high accuracy and F1 score.
- The project successfully demonstrates how to prepare data, balance classes, train multiple classifiers, and evaluate them robustly using cross-validation.

---

## Future Improvements

- Perform hyperparameter tuning using GridSearchCV
- Deploy the model using Flask or Streamlit
