# Data Dictionary: Student Performance Prediction Dataset

This document details the schema, data types, descriptions, and valid ranges/categories for the `student_performance.csv` dataset.

## Dataset Overview
- **File Name:** `student_performance.csv`
- **Location:** `D:\Student_Performance\data\`
- **Target Variable:** `Performance` (`Low`, `Medium`, `High`)
- **Primary Key:** `Student_ID`

---

## Column Specifications

| Column Name | Data Type | Range / Categories | Description |
| :--- | :--- | :--- | :--- |
| `Student_ID` | String / Object | `STU_1001` to `STU_2000` | Unique identifier assigned to each student record. |
| `Study_Hours` | Float | `0.0` – `10.0` | Daily average hours spent on self-study outside of class. |
| `Attendance` | Float | `40.0` – `100.0` | Percentage of classes attended by the student. |
| `Previous_Marks` | Float | `30.0` – `100.0` | Academic percentage score obtained in the previous academic term/grade. |
| `Assignment_Score` | Float | `0.0` – `20.0` | Cumulative score achieved across semester assignments. |
| `Internal_Marks` | Float | `0.0` – `30.0` | Marks obtained in internal mid-term examinations. |
| `Sleep_Hours` | Float | `4.0` – `10.0` | Daily average hours of sleep. |
| `Internet_Hours` | Float | `0.0` – `10.0` | Daily average hours spent on non-academic internet usage. |
| `Family_Support` | Categorical (String) | `Low`, `Medium`, `High` | Level of educational and financial encouragement received from family. |
| `Extracurricular` | Categorical (String) | `Yes`, `No` | Active participation in sports, clubs, or non-academic activities. |
| `Performance` | Categorical (String) | `Low`, `Medium`, `High` | **Target Variable:** Overall academic outcome of the student. |

---

## Data Quality Characteristics
1. **Missing Values:** Small percentage of missing values (`NaN`) injected across input feature columns to simulate real-world data collection issues.
2. **Duplicate Rows:** A small set of duplicate student records included to test data cleaning pipelines.
3. **Target Variable Generation:** `Performance` is derived non-deterministically from academic metrics (`Study_Hours`, `Previous_Marks`, `Internal_Marks`, `Assignment_Score`, `Attendance`) and behavioral features with added Gaussian noise to prevent target leakage.
