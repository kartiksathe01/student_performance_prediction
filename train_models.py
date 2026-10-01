import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Set global seaborn theme
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.family': 'sans-serif'})

def main():
    print("==================================================")
    print(" 1. LOAD CLEANED DATASET & TRAIN/TEST SPLIT")
    print("==================================================")
    data_path = os.path.join('data', 'student_performance_cleaned.csv')
    df = pd.read_csv(data_path)
    
    print(f"Loaded cleaned dataset shape: {df.shape}")
    
    target_col = 'Performance'
    feature_cols = [
        'Study_Hours', 'Attendance', 'Previous_Marks', 'Assignment_Score',
        'Internal_Marks', 'Sleep_Hours', 'Internet_Hours', 'Family_Support', 'Extracurricular'
    ]
    
    X = df[feature_cols]
    y = df[target_col]
    
    class_names = sorted(list(y.unique()))
    print(f"Target classes ({len(class_names)}): {class_names}")
    
    # Split data: 80% train, 20% test, stratify=y, random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Train set size: {X_train.shape[0]} rows")
    print(f"Test set size:  {X_test.shape[0]} rows")
    
    print("\n==================================================")
    print(" 2. CONSTRUCT PREPROCESSING PIPELINE")
    print("==================================================")
    num_features = [
        'Study_Hours', 'Attendance', 'Previous_Marks', 'Assignment_Score',
        'Internal_Marks', 'Sleep_Hours', 'Internet_Hours'
    ]
    cat_features = ['Family_Support', 'Extracurricular']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
        ]
    )
    
    print("Preprocessor initialized: StandardScaler for numerical, OneHotEncoder for categorical features.")
    
    print("\n==================================================")
    print(" 3. TRAIN AND EVALUATE MODELS")
    print("==================================================")
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine': SVC(kernel='rbf', random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    file_slug_map = {
        'Logistic Regression': 'logistic_regression',
        'K-Nearest Neighbors': 'knn',
        'Support Vector Machine': 'svm',
        'Decision Tree': 'decision_tree',
        'Random Forest': 'random_forest'
    }
    
    plots_dir = os.path.join('reports', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    evaluation_results = []
    trained_pipelines = {}
    classification_reports = {}
    
    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        # Build full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Fit on training data ONLY
        pipeline.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        clf_rep = classification_report(y_test, y_pred)
        
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        evaluation_results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1_Score': round(f1, 4)
        })
        
        trained_pipelines[name] = pipeline
        classification_reports[name] = clf_rep
        
        # Generate and save Confusion Matrix plot
        cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High'])
        
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        plt.title(f'Confusion Matrix — {name}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        slug = file_slug_map[name]
        cm_filename = f"confusion_matrix_{slug}.png"
        cm_filepath = os.path.join(plots_dir, cm_filename)
        plt.savefig(cm_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix: {cm_filepath}")

    # Convert results to DataFrame
    eval_df = pd.DataFrame(evaluation_results)
    
    print("\n==================================================")
    print(" 4. SAVE MODEL EVALUATION METRICS CSV")
    print("==================================================")
    eval_csv_path = os.path.join('reports', 'model_evaluation.csv')
    eval_df.to_csv(eval_csv_path, index=False)
    print(f"Model evaluation metrics saved to: {eval_csv_path}")
    print(eval_df.to_string(index=False))

    print("\n==================================================")
    print(" 5. GENERATE MODEL COMPARISON CHART")
    print("==================================================")
    eval_melted = eval_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=eval_melted, x='Model', y='Score', hue='Metric', palette='muted')
    plt.title('Performance Comparison Across All 5 ML Classification Models', fontsize=14, fontweight='bold')
    plt.xlabel('Machine Learning Model', fontsize=12)
    plt.ylabel('Evaluation Metric Score', fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.legend(title='Metric', loc='lower right')
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=90, xytext=(0, 3), textcoords='offset points')
            
    plt.tight_layout()
    comparison_chart_path = os.path.join(plots_dir, 'model_comparison.png')
    plt.savefig(comparison_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved model comparison chart to: {comparison_chart_path}")

    print("\n==================================================")
    print(" 6. SELECT & SAVE FINAL MODEL & METADATA")
    print("==================================================")
    # Select best model based on highest weighted F1-Score (secondary: Accuracy)
    best_row = eval_df.sort_values(by=['F1_Score', 'Accuracy'], ascending=False).iloc[0]
    best_model_name = best_row['Model']
    best_pipeline = trained_pipelines[best_model_name]
    
    print(f"Selected Best Model: {best_model_name} (F1-Score: {best_row['F1_Score']}, Accuracy: {best_row['Accuracy']})")
    
    # Save model binary using joblib
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    model_pkl_path = os.path.join(models_dir, 'student_performance_model.pkl')
    joblib.dump(best_pipeline, model_pkl_path)
    print(f"Saved best model pipeline binary to: {model_pkl_path}")
    
    # Save metadata JSON
    metadata = {
        "model_name": best_model_name,
        "feature_names": feature_cols,
        "target_name": target_col,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_state": 42,
        "accuracy": float(best_row['Accuracy']),
        "precision": float(best_row['Precision']),
        "recall": float(best_row['Recall']),
        "f1_score": float(best_row['F1_Score']),
        "class_names": ['Low', 'Medium', 'High']
    }
    
    metadata_json_path = os.path.join(models_dir, 'model_metadata.json')
    with open(metadata_json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata JSON to: {metadata_json_path}")

    print("\n==================================================")
    print(" 7. CREATE MODEL RESULTS REPORT")
    print("==================================================")
    results_md_path = os.path.join('reports', 'model_results.md')
    
    md_content = f"""# Machine Learning Model Training & Evaluation Results

This report documents the performance of 5 classification models trained to predict student performance levels (`Low`, `Medium`, `High`).

---

## 1. Experimental Setup
- **Dataset:** `D:\\Student_Performance\\data\\student_performance_cleaned.csv`
- **Total Records:** {df.shape[0]}
- **Input Features (9):** `Study_Hours`, `Attendance`, `Previous_Marks`, `Assignment_Score`, `Internal_Marks`, `Sleep_Hours`, `Internet_Hours`, `Family_Support`, `Extracurricular`
- **Target Variable:** `Performance`
- **Data Split:** 80% Training ({X_train.shape[0]} samples), 20% Testing ({X_test.shape[0]} samples) stratified by target class (`random_state=42`)
- **Preprocessing:** `StandardScaler` for numerical features, `OneHotEncoder(drop='first')` for categorical features, fitted **ONLY** on the training dataset.

---

## 2. Model Performance Summary Table

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
"""
    for _, row in eval_df.iterrows():
        md_content += f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1_Score']:.4f} |\n"

    md_content += f"""
---

## 3. Final Model Selection

- **Selected Model:** `{best_model_name}`
- **Reason for Selection:** Selected strictly based on the highest weighted **F1-Score ({best_row['F1_Score']:.4f})** and **Accuracy ({best_row['Accuracy']:.4f})** on the unseen test dataset.
- **Model Storage:** Saved as a scikit-learn Pipeline artifact at `D:\\Student_Performance\\models\\student_performance_model.pkl`.
- **Metadata Location:** `D:\\Student_Performance\\models\\model_metadata.json`.

---

## 4. Individual Model Classification Reports

"""
    for name, rep in classification_reports.items():
        slug = file_slug_map[name]
        md_content += f"### {name}\n```text\n{rep}\n```\n- **Confusion Matrix Plot:** `reports/plots/confusion_matrix_{slug}.png`\n\n"

    md_content += """---

## 5. Key Observations
1. **Model Comparison:** Evaluated models demonstrated varying baseline learning capacities across the synthetic dataset distribution.
2. **Data Leakage Safeguards:** Preprocessing transformations were fitted exclusively on training data to ensure zero target leakage.
3. **Reproducibility:** A fixed `random_state=42` was applied across all random splitters and model initializations.
"""

    with open(results_md_path, 'w') as f:
        f.write(md_content)
    print(f"Created detailed markdown results report at: {results_md_path}")

if __name__ == '__main__':
    main()
