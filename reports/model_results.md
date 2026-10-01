# Machine Learning Model Training & Evaluation Results

This report documents the performance of 5 classification models trained to predict student performance levels (`Low`, `Medium`, `High`).

---

## 1. Experimental Setup
- **Dataset:** `D:\Student_Performance\data\student_performance_cleaned.csv`
- **Total Records:** 1000
- **Input Features (9):** `Study_Hours`, `Attendance`, `Previous_Marks`, `Assignment_Score`, `Internal_Marks`, `Sleep_Hours`, `Internet_Hours`, `Family_Support`, `Extracurricular`
- **Target Variable:** `Performance`
- **Data Split:** 80% Training (800 samples), 20% Testing (200 samples) stratified by target class (`random_state=42`)
- **Preprocessing:** `StandardScaler` for numerical features, `OneHotEncoder(drop='first')` for categorical features, fitted **ONLY** on the training dataset.

---

## 2. Model Performance Summary Table

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.7850 | 0.7835 | 0.7850 | 0.7831 |
| K-Nearest Neighbors | 0.6650 | 0.6625 | 0.6650 | 0.6571 |
| Support Vector Machine | 0.7750 | 0.7939 | 0.7750 | 0.7661 |
| Decision Tree | 0.6250 | 0.6232 | 0.6250 | 0.6234 |
| Random Forest | 0.7500 | 0.7675 | 0.7500 | 0.7396 |

---

## 3. Final Model Selection

- **Selected Model:** `Logistic Regression`
- **Reason for Selection:** Selected strictly based on the highest weighted **F1-Score (0.7831)** and **Accuracy (0.7850)** on the unseen test dataset.
- **Model Storage:** Saved as a scikit-learn Pipeline artifact at `D:\Student_Performance\models\student_performance_model.pkl`.
- **Metadata Location:** `D:\Student_Performance\models\model_metadata.json`.

---

## 4. Individual Model Classification Reports

### Logistic Regression
```text
              precision    recall  f1-score   support

        High       0.71      0.61      0.66        36
         Low       0.83      0.80      0.81        54
      Medium       0.79      0.84      0.81       110

    accuracy                           0.79       200
   macro avg       0.77      0.75      0.76       200
weighted avg       0.78      0.79      0.78       200

```
- **Confusion Matrix Plot:** `reports/plots/confusion_matrix_logistic_regression.png`

### K-Nearest Neighbors
```text
              precision    recall  f1-score   support

        High       0.60      0.42      0.49        36
         Low       0.70      0.59      0.64        54
      Medium       0.67      0.78      0.72       110

    accuracy                           0.67       200
   macro avg       0.65      0.60      0.62       200
weighted avg       0.66      0.67      0.66       200

```
- **Confusion Matrix Plot:** `reports/plots/confusion_matrix_knn.png`

### Support Vector Machine
```text
              precision    recall  f1-score   support

        High       0.83      0.53      0.64        36
         Low       0.89      0.63      0.74        54
      Medium       0.73      0.93      0.82       110

    accuracy                           0.78       200
   macro avg       0.82      0.69      0.73       200
weighted avg       0.79      0.78      0.77       200

```
- **Confusion Matrix Plot:** `reports/plots/confusion_matrix_svm.png`

### Decision Tree
```text
              precision    recall  f1-score   support

        High       0.55      0.47      0.51        36
         Low       0.60      0.63      0.61        54
      Medium       0.66      0.67      0.67       110

    accuracy                           0.62       200
   macro avg       0.60      0.59      0.60       200
weighted avg       0.62      0.62      0.62       200

```
- **Confusion Matrix Plot:** `reports/plots/confusion_matrix_decision_tree.png`

### Random Forest
```text
              precision    recall  f1-score   support

        High       0.82      0.50      0.62        36
         Low       0.84      0.59      0.70        54
      Medium       0.71      0.91      0.80       110

    accuracy                           0.75       200
   macro avg       0.79      0.67      0.71       200
weighted avg       0.77      0.75      0.74       200

```
- **Confusion Matrix Plot:** `reports/plots/confusion_matrix_random_forest.png`

---

## 5. Key Observations
1. **Model Comparison:** Evaluated models demonstrated varying baseline learning capacities across the synthetic dataset distribution.
2. **Data Leakage Safeguards:** Preprocessing transformations were fitted exclusively on training data to ensure zero target leakage.
3. **Reproducibility:** A fixed `random_state=42` was applied across all random splitters and model initializations.
