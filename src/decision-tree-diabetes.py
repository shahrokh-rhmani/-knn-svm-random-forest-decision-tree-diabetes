import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Create directory for images if it doesn't exist
os.makedirs('../img/decision_tree', exist_ok=True)

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

# 5. Split data into train and test sets (No standardization needed for Decision Tree)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42, stratify=y)

print(f"\nTraining data dimensions: {X_train.shape}")
print(f"Test data dimensions: {X_test.shape}")

# 6. Evaluate Decision Tree with different max_depth values
print("\nEvaluating Decision Tree with different max_depth values...")
max_depth_range = range(1, 21)
accuracy_scores_depth = []
precision_scores_depth = []
recall_scores_depth = []

for depth in max_depth_range:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    accuracy_scores_depth.append(accuracy_score(y_test, y_pred))
    precision_scores_depth.append(precision_score(y_test, y_pred, zero_division=0))
    recall_scores_depth.append(recall_score(y_test, y_pred))

# 7. Find the best max_depth
best_depth_acc = max_depth_range[np.argmax(accuracy_scores_depth)]
best_depth_prec = max_depth_range[np.argmax(precision_scores_depth)]
best_depth_rec = max_depth_range[np.argmax(recall_scores_depth)]

print(f"\nBest max_depth based on accuracy: {best_depth_acc}")
print(f"Best max_depth based on precision: {best_depth_prec}")
print(f"Best max_depth based on recall: {best_depth_rec}")

# 8. Plot evaluation metrics vs max_depth
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(max_depth_range, accuracy_scores_depth, marker='o', linewidth=2, markersize=6)
plt.axvline(x=best_depth_acc, color='red', linestyle='--', alpha=0.7, label=f'Best depth={best_depth_acc}')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Max Depth')
plt.grid(True, alpha=0.3)
plt.legend()

# Precision plot
plt.subplot(1, 3, 2)
plt.plot(max_depth_range, precision_scores_depth, marker='o', linewidth=2, markersize=6, color='green')
plt.axvline(x=best_depth_prec, color='red', linestyle='--', alpha=0.7, label=f'Best depth={best_depth_prec}')
plt.xlabel('Max Depth')
plt.ylabel('Precision')
plt.title('Precision vs Max Depth')
plt.grid(True, alpha=0.3)
plt.legend()

# Recall plot
plt.subplot(1, 3, 3)
plt.plot(max_depth_range, recall_scores_depth, marker='o', linewidth=2, markersize=6, color='orange')
plt.axvline(x=best_depth_rec, color='red', linestyle='--', alpha=0.7, label=f'Best depth={best_depth_rec}')
plt.xlabel('Max Depth')
plt.ylabel('Recall')
plt.title('Recall vs Max Depth')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('../img/decision_tree/depth_metrics_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. Perform Grid Search for best parameters
print("\nPerforming grid search for optimal parameters...")
param_grid = {
    'max_depth': [3, 5, 7, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2']
}

dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# 10. Train final model with best parameters
best_dt = grid_search.best_estimator_
best_dt.fit(X_train, y_train)
y_pred_final = best_dt.predict(X_test)

# 11. Calculate final evaluation metrics
final_accuracy = accuracy_score(y_test, y_pred_final)
final_precision = precision_score(y_test, y_pred_final, zero_division=0)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0

print(f"\nFinal results with optimized parameters:")
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
plt.title('Confusion Matrix for Decision Tree')
plt.savefig('../img/decision_tree/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 13. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, 
                          target_names=['No Diabetes', 'Diabetes']))

# 14. Feature importance
print("\nFeature Importance Analysis:")
feature_importances = best_dt.feature_importances_
feature_names = X.columns

feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feature_imp_df['Feature'], feature_imp_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Decision Tree - Feature Importance')
plt.tight_layout()
plt.savefig('../img/decision_tree/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 15. Visualize the decision tree (pruned version for readability)
plt.figure(figsize=(20, 10))
plot_tree(best_dt, 
          feature_names=feature_names,
          class_names=['No Diabetes', 'Diabetes'],
          filled=True,
          rounded=True,
          fontsize=10,
          max_depth=3)  # Limit depth for readability
plt.title('Decision Tree Visualization (First 3 Levels)')
plt.tight_layout()
plt.savefig('../img/decision_tree/tree_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 16. Additional plot: Comparison of all metrics vs max_depth
plt.figure(figsize=(12, 8))
plt.plot(max_depth_range, accuracy_scores_depth, marker='o', linewidth=2, markersize=6, label='Accuracy')
plt.plot(max_depth_range, precision_scores_depth, marker='s', linewidth=2, markersize=6, label='Precision', color='green')
plt.plot(max_depth_range, recall_scores_depth, marker='^', linewidth=2, markersize=6, label='Recall', color='orange')
plt.axvline(x=best_depth_acc, color='red', linestyle='--', alpha=0.7, label=f'Best depth={best_depth_acc}')
plt.xlabel('Max Depth')
plt.ylabel('Score')
plt.title('Decision Tree Performance Metrics vs Max Depth')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../img/decision_tree/all_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 17. Analyze min_samples_split parameter
print("\nAnalyzing min_samples_split parameter...")
min_samples_split_range = [2, 5, 10, 15, 20, 25, 30]
accuracy_scores_split = []

for min_split in min_samples_split_range:
    dt_split = DecisionTreeClassifier(
        min_samples_split=min_split,
        random_state=42
    )
    scores = cross_val_score(dt_split, X, y, cv=5, scoring='accuracy')
    accuracy_scores_split.append(scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(min_samples_split_range, accuracy_scores_split, marker='o', linewidth=2, markersize=8)
plt.xlabel('min_samples_split')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Decision Tree Performance vs min_samples_split')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../img/decision_tree/performance_vs_min_samples_split.png', dpi=300, bbox_inches='tight')
plt.show()

# 18. Learning curve
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_dt, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
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
plt.title('Learning Curve for Decision Tree')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../img/decision_tree/learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 19. Results analysis
print("\n" + "="*50)
print("RESULTS ANALYSIS:")
print("="*50)

print(f"1. Best max_depth (simple): {best_depth_acc}")
print(f"2. Best parameters (grid search): {grid_search.best_params_}")
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
print(f"- Decision Tree achieved {final_accuracy*100:.2f}% accuracy with optimized parameters")
print(f"- No feature scaling required for Decision Trees")
print(f"- max_depth controls tree complexity and prevents overfitting")
print(f"- Feature importance reveals Glucose as the most predictive feature")
print(f"- The tree visualization shows the decision-making process")
print(f"- Grid search optimized multiple hyperparameters for best performance")

# 20. Cross-validation for more reliable results
print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS:")
print("="*50)
cv_scores_final = cross_val_score(best_dt, X, y, cv=10, scoring='accuracy')
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
plt.savefig('../img/decision_tree/cross_validation_scores.png', dpi=300, bbox_inches='tight')
plt.show()

# 21. Compare with default Decision Tree
default_dt = DecisionTreeClassifier(random_state=42)
default_dt.fit(X_train, y_train)
y_pred_default = default_dt.predict(X_test)
default_accuracy = accuracy_score(y_test, y_pred_default)

print(f"\nComparison with Default Decision Tree:")
print(f"Default DT Accuracy: {default_accuracy:.4f}")
print(f"Tuned DT Accuracy: {final_accuracy:.4f}")
print(f"Improvement: {final_accuracy - default_accuracy:.4f}")

print(f"\nAll images have been successfully saved in the '../img/decision_tree/' directory:")
print("- depth_metrics_plot.png")
print("- confusion_matrix.png")
print("- feature_importance.png")
print("- tree_visualization.png")
print("- all_metrics_comparison.png")
print("- performance_vs_min_samples_split.png")
print("- learning_curve.png")
print("- cross_validation_scores.png")

print(f"\nDecision Tree implementation completed successfully!")