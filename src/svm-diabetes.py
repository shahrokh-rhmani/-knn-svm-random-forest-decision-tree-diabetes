import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Create directory for images if it doesn't exist
os.makedirs('../img/svm', exist_ok=True)

# Set display options
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 1. Load the dataset
df = pd.read_csv('../diabetes.csv')

print("First rows of the dataset:")
print(df.head())
print("\nDataset information:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe())

# 2. Check for null and unreasonable values
print("\nChecking zero values in important columns:")
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} zero values ({zero_count/len(df)*100:.2f}%)")

# 3. Data preprocessing
df_clean = df.copy()
for col in zero_columns:
    mean_val = df_clean[df_clean[col] != 0][col].mean()
    df_clean[col] = df_clean[col].replace(0, mean_val)

print("\nAfter replacing zero values:")
for col in zero_columns:
    zero_count = (df_clean[col] == 0).sum()
    print(f"{col}: {zero_count} zero values")

# 4. Separate features and labels
X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

print(f"\nClass distribution:\n{y.value_counts()}")
print(f"Class proportions:\n{y.value_counts(normalize=True)}")

# 5. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, 
                                                    random_state=42, stratify=y)

print(f"\nTraining data dimensions: {X_train.shape}")
print(f"Test data dimensions: {X_test.shape}")

# 7. Evaluate SVM with different kernels and parameters
print("\nEvaluating SVM with different kernels...")

# Define parameter grids for different kernels
param_grids = {
    'linear': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear']
    },
    'rbf': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    },
    'poly': {
        'C': [0.1, 1, 10],
        'degree': [2, 3, 4],
        'kernel': ['poly']
    }
}

best_models = {}
results = {}

for kernel_name, param_grid in param_grids.items():
    print(f"\nTraining SVM with {kernel_name} kernel...")
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_models[kernel_name] = grid_search.best_estimator_
    results[kernel_name] = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'test_accuracy': grid_search.best_estimator_.score(X_test, y_test)
    }
    
    print(f"Best parameters for {kernel_name}: {grid_search.best_params_}")
    print(f"Best CV accuracy for {kernel_name}: {grid_search.best_score_:.4f}")

# 8. Compare kernel performances
kernel_names = list(results.keys())
cv_scores = [results[kernel]['best_score'] for kernel in kernel_names]
test_scores = [results[kernel]['test_accuracy'] for kernel in kernel_names]

plt.figure(figsize=(10, 6))
x_pos = np.arange(len(kernel_names))
width = 0.35

plt.bar(x_pos - width/2, cv_scores, width, label='CV Accuracy', alpha=0.8)
plt.bar(x_pos + width/2, test_scores, width, label='Test Accuracy', alpha=0.8)

plt.xlabel('SVM Kernel')
plt.ylabel('Accuracy')
plt.title('SVM Performance Comparison with Different Kernels')
plt.xticks(x_pos, kernel_names)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('../img/svm/svm_kernels_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. Find the best kernel
best_kernel = max(results.keys(), key=lambda x: results[x]['best_score'])
best_svm = best_models[best_kernel]
best_params = results[best_kernel]['best_params']

print(f"\nBest kernel: {best_kernel}")
print(f"Best parameters: {best_params}")
print(f"Best CV accuracy: {results[best_kernel]['best_score']:.4f}")

# 10. Train final model with best parameters
final_svm = SVC(**best_params, random_state=42)
final_svm.fit(X_train, y_train)
y_pred_final = final_svm.predict(X_test)

# 11. Calculate final evaluation metrics
final_accuracy = accuracy_score(y_test, y_pred_final)
final_precision = precision_score(y_test, y_pred_final, zero_division=0)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0

print(f"\nFinal results with {best_kernel} kernel:")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"Precision: {final_precision:.4f}")
print(f"Recall: {final_recall:.4f}")
print(f"F1-Score: {final_f1:.4f}")

# 12. Confusion matrix
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'], 
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix for SVM ({best_kernel} kernel)')
plt.savefig('../img/svm/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 13. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, 
                          target_names=['No Diabetes', 'Diabetes']))

# 14. Feature importance analysis (for linear kernel)
if best_kernel == 'linear':
    print("\nFeature Importance Analysis (Linear SVM):")
    feature_names = X.columns
    coefficients = final_svm.coef_[0]
    
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Absolute_Coefficient': np.abs(coefficients)
    }).sort_values('Absolute_Coefficient', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_imp_df['Feature'], feature_imp_df['Absolute_Coefficient'])
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Feature Importance (Linear SVM)')
    plt.tight_layout()
    plt.savefig('../img/svm/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nFeature coefficients:")
    for _, row in feature_imp_df.iterrows():
        print(f"{row['Feature']}: {row['Coefficient']:.4f}")

# 15. Hyperparameter tuning visualization for RBF kernel
if best_kernel == 'rbf':
    print("\nRBF Kernel Hyperparameter Analysis...")
    
    # Create a more detailed grid for visualization
    C_values = [0.1, 1, 10, 100]
    gamma_values = [0.001, 0.01, 0.1, 'scale', 'auto']
    
    # Sample for visualization (using a subset for clarity)
    svm_rbf = SVC(kernel='rbf', random_state=42)
    param_grid_vis = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 'scale']}
    grid_search_vis = GridSearchCV(svm_rbf, param_grid_vis, cv=3, scoring='accuracy')
    grid_search_vis.fit(X_train, y_train)
    
    # Create heatmap of CV results
    results_df = pd.DataFrame(grid_search_vis.cv_results_)
    scores = results_df.pivot_table(index='param_gamma', columns='param_C', values='mean_test_score')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(scores, annot=True, fmt='.3f', cmap='viridis')
    plt.title('SVM RBF Kernel - Hyperparameter Tuning Results')
    plt.xlabel('C (Regularization)')
    plt.ylabel('Gamma')
    plt.tight_layout()
    plt.savefig('../img/svm/rbf_hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
    plt.show()

# 16. Learning curve analysis (simplified)
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    final_svm, X_scaled, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy', n_jobs=-1, random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Cross-validation score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='green')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title(f'Learning Curve for SVM ({best_kernel} kernel)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../img/svm/learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 17. Results analysis
print("\n" + "="*50)
print("RESULTS ANALYSIS:")
print("="*50)

print(f"1. Best kernel: {best_kernel}")
print(f"2. Best parameters: {best_params}")
print(f"3. Final model accuracy: {final_accuracy*100:.2f}%")
print(f"4. Precision: {final_precision*100:.2f}% - Accuracy in identifying true patients")
print(f"5. Recall: {final_recall*100:.2f}% - Ability to detect all actual patients")
print(f"6. F1-Score: {final_f1*100:.2f}% - Harmonic mean of precision and recall")

tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix Analysis:")
print(f"- True Negative (TN): {tn} - Correctly identified healthy individuals")
print(f"- False Positive (FP): {fp} - Healthy individuals incorrectly identified as patients")
print(f"- False Negative (FN): {fn} - Patients incorrectly identified as healthy")
print(f"- True Positive (TP): {tp} - Correctly identified patients")

# Calculate additional metrics
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"\nAdditional Metrics:")
print(f"- Specificity: {specificity:.4f} - Ability to identify healthy individuals")
print(f"- False Positive Rate: {false_positive_rate:.4f}")

print(f"\nKernel Performance Summary:")
for kernel in kernel_names:
    print(f"- {kernel.upper()} kernel: CV={results[kernel]['best_score']:.4f}, Test={results[kernel]['test_accuracy']:.4f}")

print(f"\nKey Insights:")
print(f"- SVM with {best_kernel} kernel showed the best performance")
print(f"- Data standardization is crucial for SVM")
print(f"- Different kernels capture different patterns in the data")
print(f"- Class imbalance in diabetes dataset: {y.value_counts().to_dict()}")
print(f"- The model balance between precision and recall: {'Good' if abs(final_precision - final_recall) < 0.2 else 'Needs improvement'}")

# 18. Cross-validation for more reliable results
print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS:")
print("="*50)
cv_scores_final = cross_val_score(final_svm, X_scaled, y, cv=10, scoring='accuracy')
print(f"Cross-validation scores: {[f'{score:.4f}' for score in cv_scores_final]}")
print(f"Mean CV accuracy: {cv_scores_final.mean():.4f} (+/- {cv_scores_final.std() * 2:.4f})")

# Plot cross-validation results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), cv_scores_final, marker='o', linewidth=2, markersize=8)
plt.axhline(y=cv_scores_final.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores_final.mean():.4f}')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.title('10-Fold Cross-Validation Scores')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../img/svm/cross_validation_scores.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAll images have been successfully saved in the '../img/svm/' directory:")
print("- svm_kernels_comparison.png")
print("- confusion_matrix.png")
if best_kernel == 'linear':
    print("- feature_importance.png")
if best_kernel == 'rbf':
    print("- rbf_hyperparameter_tuning.png")
print("- learning_curve.png")
print("- cross_validation_scores.png")

print(f"\nSVM implementation completed successfully!")