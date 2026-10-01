import os
import numpy as np
import pandas as pd

def main():
    # Fixed random seed for reproducibility
    np.random.seed(42)

    num_samples = 1000

    # Generate Student IDs
    student_ids = [f"STU_{1001 + i}" for i in range(num_samples)]

    # Generate feature variables within specified ranges
    study_hours = np.round(np.random.uniform(0.0, 10.0, num_samples), 1)
    attendance = np.round(np.random.uniform(40.0, 100.0, num_samples), 1)
    previous_marks = np.round(np.random.uniform(30.0, 100.0, num_samples), 1)
    assignment_score = np.round(np.random.uniform(0.0, 20.0, num_samples), 1)
    internal_marks = np.round(np.random.uniform(0.0, 30.0, num_samples), 1)
    sleep_hours = np.round(np.random.uniform(4.0, 10.0, num_samples), 1)
    internet_hours = np.round(np.random.uniform(0.0, 10.0, num_samples), 1)

    family_support = np.random.choice(['Low', 'Medium', 'High'], size=num_samples, p=[0.25, 0.50, 0.25])
    extracurricular = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.45, 0.55])

    # Latent performance calculation (academic and behavioral variables)
    norm_study = study_hours / 10.0
    norm_attend = (attendance - 40.0) / 60.0
    norm_prev = (previous_marks - 30.0) / 70.0
    norm_assign = assignment_score / 20.0
    norm_internal = internal_marks / 30.0

    fam_map = {'Low': 0.0, 'Medium': 0.5, 'High': 1.0}
    fam_val = np.array([fam_map[val] for val in family_support])

    extra_map = {'Yes': 1.0, 'No': 0.0}
    extra_val = np.array([extra_map[val] for val in extracurricular])

    # Weighted score + non-deterministic random noise
    composite_score = (
        0.28 * norm_study +
        0.26 * norm_prev +
        0.22 * norm_internal +
        0.14 * norm_assign +
        0.06 * norm_attend +
        0.03 * fam_val +
        0.01 * extra_val
    )
    
    noise = np.random.normal(0, 0.07, num_samples)
    latent_score = composite_score + noise

    # Map to Performance classes: Low, Medium, High
    performance = []
    for score in latent_score:
        if score < 0.40:
            performance.append('Low')
        elif score < 0.64:
            performance.append('Medium')
        else:
            performance.append('High')

    # Construct DataFrame
    df = pd.DataFrame({
        'Student_ID': student_ids,
        'Study_Hours': study_hours,
        'Attendance': attendance,
        'Previous_Marks': previous_marks,
        'Assignment_Score': assignment_score,
        'Internal_Marks': internal_marks,
        'Sleep_Hours': sleep_hours,
        'Internet_Hours': internet_hours,
        'Family_Support': family_support,
        'Extracurricular': extracurricular,
        'Performance': performance
    })

    # Ensure clipping to avoid impossible boundary values
    df['Study_Hours'] = df['Study_Hours'].clip(0.0, 10.0)
    df['Attendance'] = df['Attendance'].clip(40.0, 100.0)
    df['Previous_Marks'] = df['Previous_Marks'].clip(30.0, 100.0)
    df['Assignment_Score'] = df['Assignment_Score'].clip(0.0, 20.0)
    df['Internal_Marks'] = df['Internal_Marks'].clip(0.0, 30.0)
    df['Sleep_Hours'] = df['Sleep_Hours'].clip(4.0, 10.0)
    df['Internet_Hours'] = df['Internet_Hours'].clip(0.0, 10.0)

    # Inject small number of missing values in feature columns
    input_cols = [
        'Study_Hours', 'Attendance', 'Previous_Marks', 'Assignment_Score', 
        'Internal_Marks', 'Sleep_Hours', 'Internet_Hours', 'Family_Support', 'Extracurricular'
    ]
    
    nan_count = 25
    for _ in range(nan_count):
        r_idx = np.random.randint(0, num_samples)
        c_idx = np.random.choice(input_cols)
        df.loc[r_idx, c_idx] = np.nan

    # Add small number of duplicate rows
    duplicate_indices = np.random.choice(df.index, size=10, replace=False)
    duplicate_rows = df.loc[duplicate_indices].copy()
    
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # Save to CSV file
    output_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'student_performance.csv')
    df.to_csv(csv_path, index=False)

    print(f"Dataset successfully created and saved to: {csv_path}")
    print("\n--- DATASET SUMMARY ---")
    print(f"Dataset Shape: {df.shape}")
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nColumn Names:")
    print(list(df.columns))
    print("\nMissing-Value Count per Column:")
    print(df.isnull().sum())
    print(f"\nTotal Missing Values: {df.isnull().sum().sum()}")
    print(f"\nDuplicate Count: {df.duplicated().sum()}")
    print("\nPerformance Distribution:")
    print(df['Performance'].value_counts())

if __name__ == '__main__':
    main()
