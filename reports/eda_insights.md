# Exploratory Data Analysis & Preprocessing Insights

This document presents the factual findings and statistical observations derived from the exploratory data analysis (EDA) and data preprocessing phase of the **Student Performance Prediction** project.

---

## 1. Summary of Data Preprocessing
- **Original Dataset Shape:** `1,010` rows × `11` columns.
- **Duplicate Records Removed:** `10` duplicate student rows were identified and removed, bringing the unique student record count to `1,000`.
- **Missing Value Imputation:**
  - `25` missing values across numerical and categorical input features were imputed without using target variable information (`Performance`).
  - **Numerical features** (`Study_Hours`, `Attendance`, `Previous_Marks`, `Assignment_Score`, `Internal_Marks`, `Sleep_Hours`, `Internet_Hours`) were imputed using feature-wise **medians**.
  - **Categorical features** (`Family_Support`, `Extracurricular`) were imputed using feature-wise **modes**.
  - **Post-Imputation Missing Count:** `0` missing values across all columns.
- **Cleaned Dataset Shape:** `1,000` rows × `11` columns saved to `D:\Student_Performance\data\student_performance_cleaned.csv`.

---

## 2. Key Data Distribution & Statistical Observations

### A. Target Class Distribution (`Performance`)
- **Medium Performance:** `554` students (~55.4%)
- **Low Performance:** `268` students (~26.8%)
- **High Performance:** `178` students (~17.8%)
- **Observation:** The target variable exhibits a realistic class distribution where the majority of students fall into the `Medium` performance category, while `Low` and `High` represent lower proportions.

### B. Feature Distributions & Ranges
1. **Study Hours:**
   - **Range:** `0.0` to `10.0` hours/day (Median: `5.0` hours/day).
   - **Distribution:** Uniformly distributed across the 0–10 hour range. Students with higher study hours consistently display higher rates of `High` performance classification.
2. **Attendance:**
   - **Range:** `40.2%` to `100.0%` (Median: `71.1%`).
   - **Distribution:** Spans across the minimum threshold of 40% up to full attendance.
3. **Previous Marks:**
   - **Range:** `30.0%` to `99.8%` (Median: `65.0%`).
4. **Assignment Score:**
   - **Range:** `0.0` to `20.0` points (Median: `9.7` points).
5. **Internal Marks:**
   - **Range:** `0.0` to `29.9` marks (Median: `14.9` marks).
6. **Sleep & Internet Hours:**
   - **Sleep Hours Range:** `4.0` to `10.0` hours/day (Median: `6.9` hours/day).
   - **Internet Hours Range:** `0.0` to `10.0` hours/day (Median: `4.8` hours/day).

---

## 3. Correlation Matrix Analysis (Association vs Causation)

*Note: Pearson correlation measures linear statistical association between numerical variables and does **not** establish causal relationships.*

### A. Numerical Feature Inter-Correlations
- **Study Hours & Previous Marks:** Correlation coefficient = `+0.014` (Very weak linear association).
- **Study Hours & Attendance:** Correlation coefficient = `+0.034` (Weak positive linear association).
- **Internal Marks & Assignment Score:** Correlation coefficient = `-0.013` (Negligible association).
- **Sleep Hours & Study Hours:** Correlation coefficient = `-0.060` (Slight inverse linear association).

### B. Analytical Interpretation
- Individual numerical features independently contribute to student performance through a non-linear composite relationship, meaning single-feature linear correlations with each other are relatively weak.
- Multivariable decision tree and ensemble algorithms (e.g. Random Forest, Gradient Boosting) will be well-suited for capturing these multi-feature combinations.

---

## 4. Outlier Evaluation
- **Method:** Interquartile Range (IQR) analysis with standard 1.5 × IQR threshold bounds.
- **Findings:** No extreme mathematical outliers were flagged outside the theoretical bounds, as all variables remain within realistic academic bounds (e.g., Attendance 40-100%, Marks 30-100%).
- **Decision:** All records are retained in full without truncating or removing valid academic edge cases.

---

## 5. Summary of Visual Plots Generated

All 18 generated plots and the correlation heatmap have been saved to `D:\Student_Performance\reports\plots\`:
1. `performance_distribution.png`
2. `study_hours_distribution.png`
3. `attendance_distribution.png`
4. `previous_marks_distribution.png`
5. `assignment_score_distribution.png`
6. `internal_marks_distribution.png`
7. `sleep_hours_distribution.png`
8. `internet_hours_distribution.png`
9. `study_hours_vs_performance.png`
10. `attendance_vs_performance.png`
11. `previous_marks_vs_performance.png`
12. `internal_marks_vs_performance.png`
13. `family_support_vs_performance.png`
14. `extracurricular_vs_performance.png`
15. `study_hours_vs_previous_marks.png`
16. `attendance_vs_previous_marks.png`
17. `numerical_boxplots.png`
18. `pairplot.png`
19. `correlation_heatmap.png`
