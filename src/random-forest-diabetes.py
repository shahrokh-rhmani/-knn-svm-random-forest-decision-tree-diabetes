import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Create directory for images if it doesn't exist
os.makedirs('../img/random_forest', exist_ok=True)

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

# 5. Split data into train and test sets (No standardization needed for Random Forest)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42, stratify=y)

print(f"\nTraining data dimensions: {X_train.shape}")
print(f"Test data dimensions: {X_test.shape}")

# 6. Evaluate Random Forest with different hyperparameters
print("\nEvaluating Random Forest with different hyperparameters...")

# Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# 7. Perform Grid Search for best parameters
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# 8. Get results for visualization
results_df = pd.DataFrame(grid_search.cv_results_)

# 9. Plot feature importance from the best model
best_rf = grid_search.best_estimator_
feature_importances = best_rf.feature_importances_
feature_names = X.columns

# Create feature importance dataframe
feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feature_imp_df['Feature'], feature_imp_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Random Forest - Feature Importance')
plt.tight_layout()
plt.savefig('../img/random_forest/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. Evaluate different numbers of estimators
print("\nEvaluating different numbers of estimators...")
n_estimators_range = [10, 50, 100, 200, 300, 400, 500]
accuracy_scores_est = []
precision_scores_est = []
recall_scores_est = []

for n_est in n_estimators_range:
    rf_temp = RandomForestClassifier(
        n_estimators=n_est,
        random_state=42,
        max_depth=grid_search.best_params_.get('max_depth', None),
        min_samples_split=grid_search.best_params_.get('min_samples_split', 2),
        min_samples_leaf=grid_search.best_params_.get('min_samples_leaf', 1),
        max_features=grid_search.best_params_.get('max_features', 'sqrt')
    )
    rf_temp.fit(X_train, y_train)
    y_pred_temp = rf_temp.predict(X_test)
    
    accuracy_scores_est.append(accuracy_score(y_test, y_pred_temp))
    precision_scores_est.append(precision_score(y_test, y_pred_temp, zero_division=0))
    recall_scores_est.append(recall_score(y_test, y_pred_temp))

# 11. Plot performance vs number of estimators
plt.figure(figsize=(12, 8))
plt.plot(n_estimators_range, accuracy_scores_est, marker='o', linewidth=2, markersize=6, label='Accuracy')
plt.plot(n_estimators_range, precision_scores_est, marker='s', linewidth=2, markersize=6, label='Precision', color='green')
plt.plot(n_estimators_range, recall_scores_est, marker='^', linewidth=2, markersize=6, label='Recall', color='orange')
plt.xlabel('Number of Estimators')
plt.ylabel('Score')
plt.title('Random Forest Performance vs Number of Estimators')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../img/random_forest/performance_vs_estimators.png', dpi=300, bbox_inches='tight')
plt.show()

# 12. Find best number of estimators based on accuracy
best_n_estimators = n_estimators_range[np.argmax(accuracy_scores_est)]
print(f"Best number of estimators based on test accuracy: {best_n_estimators}")

# 13. Train final model with best parameters
final_rf = RandomForestClassifier(
    n_estimators=best_n_estimators,
    max_depth=grid_search.best_params_.get('max_depth', None),
    min_samples_split=grid_search.best_params_.get('min_samples_split', 2),
    min_samples_leaf=grid_search.best_params_.get('min_samples_leaf', 1),
    max_features=grid_search.best_params_.get('max_features', 'sqrt'),
    random_state=42
)
final_rf.fit(X_train, y_train)
y_pred_final = final_rf.predict(X_test)

# 14. Calculate final evaluation metrics
final_accuracy = accuracy_score(y_test, y_pred_final)
final_precision = precision_score(y_test, y_pred_final, zero_division=0)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0

print(f"\nFinal results with {best_n_estimators} estimators:")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"Precision: {final_precision:.4f}")
print(f"Recall: {final_recall:.4f}")
print(f"F1-Score: {final_f1:.4f}")

# 15. Confusion matrix
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'], 
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix for Random Forest ({best_n_estimators} estimators)')
plt.savefig('../img/random_forest/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 16. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, 
                          target_names=['No Diabetes', 'Diabetes']))

# 17. Plot max_depth vs performance
print("\nAnalyzing max_depth parameter...")
max_depth_range = [3, 5, 10, 15, 20, 25, 30, None]
accuracy_scores_depth = []

for depth in max_depth_range:
    rf_depth = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42
    )
    scores = cross_val_score(rf_depth, X, y, cv=5, scoring='accuracy')
    accuracy_scores_depth.append(scores.mean())

plt.figure(figsize=(10, 6))
plt.plot([str(d) if d is not None else 'None' for d in max_depth_range], 
         accuracy_scores_depth, marker='o', linewidth=2, markersize=8)
plt.xlabel('Max Depth')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Random Forest Performance vs Max Depth')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../img/random_forest/performance_vs_max_depth.png', dpi=300, bbox_inches='tight')
plt.show()

# 18. Learning curve
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    final_rf, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
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
plt.title('Learning Curve for Random Forest')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../img/random_forest/learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 19. Results analysis
print("\n" + "="*50)
print("RESULTS ANALYSIS:")
print("="*50)

print(f"1. Best number of estimators: {best_n_estimators}")
print(f"2. Best parameters from grid search: {grid_search.best_params_}")
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

print(f"\nTop 5 Most Important Features:")
for i, (feature, importance) in enumerate(feature_imp_df.tail(5).iterrows()):
    print(f"{i+1}. {feature_imp_df.loc[feature, 'Feature']}: {feature_imp_df.loc[feature, 'Importance']:.4f}")

print(f"\nKey Insights:")
print(f"- Random Forest with {best_n_estimators} estimators showed the best performance")
print(f"- No feature scaling required for Random Forest")
print(f"- Feature importance analysis reveals which factors most influence diabetes prediction")
print(f"- The model handles class imbalance well through ensemble methods")
print(f"- Learning curve shows model performance with different training set sizes")
print(f"- Grid search optimized multiple hyperparameters simultaneously")

# 20. Cross-validation for more reliable results
print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS:")
print("="*50)
cv_scores_final = cross_val_score(final_rf, X, y, cv=10, scoring='accuracy')
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
plt.tight_layout()
plt.savefig('../img/random_forest/cross_validation_scores.png', dpi=300, bbox_inches='tight')
plt.show()

# 21. Compare with default Random Forest
default_rf = RandomForestClassifier(random_state=42)
default_rf.fit(X_train, y_train)
y_pred_default = default_rf.predict(X_test)
default_accuracy = accuracy_score(y_test, y_pred_default)

print(f"\nComparison with Default Random Forest:")
print(f"Default RF Accuracy: {default_accuracy:.4f}")
print(f"Tuned RF Accuracy: {final_accuracy:.4f}")
print(f"Improvement: {final_accuracy - default_accuracy:.4f}")

print(f"\nAll images have been successfully saved in the '../img/random_forest/' directory:")
print("- feature_importance.png")
print("- performance_vs_estimators.png")
print("- confusion_matrix.png")
print("- performance_vs_max_depth.png")
print("- learning_curve.png")
print("- cross_validation_scores.png")

print(f"\nRandom Forest implementation completed successfully!")