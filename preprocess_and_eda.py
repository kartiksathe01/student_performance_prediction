import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set global plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.family': 'sans-serif'})

def main():
    print("==================================================")
    print(" 1. LOAD DATA")
    print("==================================================")
    raw_path = os.path.join('data', 'student_performance.csv')
    df = pd.read_csv(raw_path)
    
    original_shape = df.shape
    print(f"Loaded dataset from: {raw_path}")
    print(f"Dataset Shape: {original_shape}")
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Last 5 Rows ---")
    print(df.tail())
    
    print("\n--- Column Names ---")
    print(list(df.columns))
    
    print("\n--- Data Types ---")
    print(df.dtypes)
    
    print("\n--- Statistical Summary ---")
    print(df.describe(include='all'))
    
    print("\n==================================================")
    print(" 2. DATA QUALITY CHECK")
    print("==================================================")
    missing_before = df.isnull().sum()
    total_missing_before = missing_before.sum()
    duplicates_before = df.duplicated().sum()
    
    print(f"Missing Values per Column:\n{missing_before}")
    print(f"\nTotal Missing Values Before Cleaning: {total_missing_before}")
    print(f"Duplicate Rows Count Before Cleaning: {duplicates_before}")
    
    print("\nCategorical Feature Values:")
    print("Family_Support:", df['Family_Support'].value_counts(dropna=False).to_dict())
    print("Extracurricular:", df['Extracurricular'].value_counts(dropna=False).to_dict())
    print("Performance:", df['Performance'].value_counts(dropna=False).to_dict())
    
    print("\nNumerical Ranges:")
    num_cols = ['Study_Hours', 'Attendance', 'Previous_Marks', 'Assignment_Score', 
                'Internal_Marks', 'Sleep_Hours', 'Internet_Hours']
    for col in num_cols:
        print(f"  {col}: Min = {df[col].min()}, Max = {df[col].max()}")
        
    print("\n==================================================")
    print(" 3. HANDLE DUPLICATES")
    print("==================================================")
    print(f"Number of duplicate rows before removal: {duplicates_before}")
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    shape_after_duplicates = df.shape
    print(f"Number of rows after removing duplicates: {shape_after_duplicates[0]}")
    
    print("\n==================================================")
    print(" 4. HANDLE MISSING VALUES")
    print("==================================================")
    print("Missing values before imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Impute numerical columns with median
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed numerical column '{col}' missing values with median: {median_val:.2f}")
            
    # Impute categorical columns with mode
    cat_cols = ['Family_Support', 'Extracurricular']
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"Imputed categorical column '{col}' missing values with mode: '{mode_val}'")
            
    missing_after = df.isnull().sum().sum()
    print(f"\nTotal Missing Values After Imputation: {missing_after}")
    
    print("\n==================================================")
    print(" 5. OUTLIER ANALYSIS")
    print("==================================================")
    outlier_summary = {}
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary[col] = len(outliers)
        print(f"  {col}: IQR={IQR:.2f}, Bounds=[{lower_bound:.2f}, {upper_bound:.2f}], Potential Outliers Count={len(outliers)}")
    print("\nOutlier Handling Decision: Potential outliers represent realistic extreme academic/behavioral variations and are retained to avoid removing valid data points.")

    print("\n==================================================")
    print(" 6. EXPLORATORY DATA ANALYSIS (PLOTS)")
    print("==================================================")
    plots_dir = os.path.join('reports', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Order for performance categorization
    perf_order = ['Low', 'Medium', 'High']
    
    # 1. Performance distribution
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x='Performance', hue='Performance', order=perf_order, palette='Set2', legend=False)
    plt.title('Distribution of Student Performance Classes', fontsize=14, fontweight='bold')
    plt.xlabel('Performance Level', fontsize=12)
    plt.ylabel('Number of Students', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
    plt.savefig(os.path.join(plots_dir, 'performance_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Helper function for distribution plots
    def plot_dist(col_name, title, xlabel, filename):
        plt.figure(figsize=(7, 5))
        sns.histplot(df[col_name], kde=True, color='skyblue', bins=20)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Study Hours distribution
    plot_dist('Study_Hours', 'Distribution of Daily Study Hours', 'Study Hours (per day)', 'study_hours_distribution.png')
    
    # 3. Attendance distribution
    plot_dist('Attendance', 'Distribution of Class Attendance Percentage', 'Attendance (%)', 'attendance_distribution.png')
    
    # 4. Previous Marks distribution
    plot_dist('Previous_Marks', 'Distribution of Previous Academic Marks', 'Previous Marks (%)', 'previous_marks_distribution.png')
    
    # 5. Assignment Score distribution
    plot_dist('Assignment_Score', 'Distribution of Assignment Scores', 'Assignment Score (out of 20)', 'assignment_score_distribution.png')
    
    # 6. Internal Marks distribution
    plot_dist('Internal_Marks', 'Distribution of Internal Examination Marks', 'Internal Marks (out of 30)', 'internal_marks_distribution.png')
    
    # 7. Sleep Hours distribution
    plot_dist('Sleep_Hours', 'Distribution of Daily Sleep Hours', 'Sleep Hours (per day)', 'sleep_hours_distribution.png')
    
    # 8. Internet Hours distribution
    plot_dist('Internet_Hours', 'Distribution of Daily Non-Academic Internet Hours', 'Internet Hours (per day)', 'internet_hours_distribution.png')

    # Helper function for bivariate boxplots vs Performance
    def plot_vs_perf(col_name, title, ylabel, filename):
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x='Performance', y=col_name, hue='Performance', order=perf_order, palette='Blues', legend=False)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Performance Level', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    # 9. Study Hours vs Performance
    plot_vs_perf('Study_Hours', 'Study Hours across Performance Classes', 'Study Hours', 'study_hours_vs_performance.png')

    # 10. Attendance vs Performance
    plot_vs_perf('Attendance', 'Attendance Percentage across Performance Classes', 'Attendance (%)', 'attendance_vs_performance.png')

    # 11. Previous Marks vs Performance
    plot_vs_perf('Previous_Marks', 'Previous Marks across Performance Classes', 'Previous Marks (%)', 'previous_marks_vs_performance.png')

    # 12. Internal Marks vs Performance
    plot_vs_perf('Internal_Marks', 'Internal Marks across Performance Classes', 'Internal Marks (out of 30)', 'internal_marks_vs_performance.png')

    # 13. Family Support vs Performance
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Family_Support', hue='Performance', hue_order=perf_order, order=['Low', 'Medium', 'High'], palette='Purples')
    plt.title('Family Support Level by Student Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Family Support Level', fontsize=12)
    plt.ylabel('Student Count', fontsize=12)
    plt.legend(title='Performance', loc='upper right')
    plt.savefig(os.path.join(plots_dir, 'family_support_vs_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 14. Extracurricular vs Performance
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Extracurricular', hue='Performance', hue_order=perf_order, order=['No', 'Yes'], palette='Greens')
    plt.title('Extracurricular Participation by Student Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Extracurricular Participation', fontsize=12)
    plt.ylabel('Student Count', fontsize=12)
    plt.legend(title='Performance', loc='upper right')
    plt.savefig(os.path.join(plots_dir, 'extracurricular_vs_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 15. Study Hours vs Previous Marks
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Study_Hours', y='Previous_Marks', hue='Performance', hue_order=perf_order, alpha=0.7, palette='viridis')
    plt.title('Study Hours vs Previous Marks', fontsize=14, fontweight='bold')
    plt.xlabel('Study Hours', fontsize=12)
    plt.ylabel('Previous Marks (%)', fontsize=12)
    plt.legend(title='Performance', loc='upper left')
    plt.savefig(os.path.join(plots_dir, 'study_hours_vs_previous_marks.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 16. Attendance vs Previous Marks
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Attendance', y='Previous_Marks', hue='Performance', hue_order=perf_order, alpha=0.7, palette='coolwarm')
    plt.title('Attendance Percentage vs Previous Marks', fontsize=14, fontweight='bold')
    plt.xlabel('Attendance (%)', fontsize=12)
    plt.ylabel('Previous Marks (%)', fontsize=12)
    plt.legend(title='Performance', loc='upper left')
    plt.savefig(os.path.join(plots_dir, 'attendance_vs_previous_marks.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 17. Numerical Boxplots (Grid view)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for idx, col in enumerate(num_cols):
        sns.boxplot(data=df, y=col, ax=axes[idx], color='lightcoral')
        axes[idx].set_title(f'Boxplot of {col}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(col, fontsize=10)
    # Hide empty subplots
    for idx in range(len(num_cols), len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'numerical_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 18. Pairplot
    pair_grid = sns.pairplot(df[num_cols + ['Performance']], hue='Performance', hue_order=perf_order, palette='tab10', corner=True)
    pair_grid.fig.suptitle('Pair Plot of Numerical Academic Features by Performance', y=1.02, fontsize=16, fontweight='bold')
    pair_grid.savefig(os.path.join(plots_dir, 'pairplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Generated and saved 18 plots in '{plots_dir}'")

    print("\n==================================================")
    print(" 7. CORRELATION MATRIX")
    print("==================================================")
    corr_matrix = df[num_cols].corr()
    print("Correlation Matrix:")
    print(corr_matrix.round(3))

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, cbar_kws={'label': 'Pearson Correlation Coefficient'})
    plt.title('Correlation Matrix Heatmap of Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    heatmap_path = os.path.join(plots_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nCorrelation heatmap saved as: {heatmap_path}")
    print("\nNote: Correlation indicates statistical association and does NOT establish causal relationship.")

    print("\n==================================================")
    print(" 8. SAVE CLEAN DATA")
    print("==================================================")
    clean_path = os.path.join('data', 'student_performance_cleaned.csv')
    df.to_csv(clean_path, index=False)
    print(f"Cleaned dataset saved successfully to: {clean_path}")
    print(f"Final Cleaned Dataset Shape: {df.shape}")

if __name__ == '__main__':
    main()
