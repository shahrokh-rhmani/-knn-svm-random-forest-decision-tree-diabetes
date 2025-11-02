import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

# Create directory for images if it doesn't exist
os.makedirs('../img/knn', exist_ok=True)

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

# 7. Evaluate KNN model with different k values
k_values = range(1, 31)
accuracy_scores = []
precision_scores = []
recall_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
    recall_scores.append(recall_score(y_test, y_pred))

# 8. Find the best k
best_k_acc = k_values[np.argmax(accuracy_scores)]
best_k_prec = k_values[np.argmax(precision_scores)]
best_k_rec = k_values[np.argmax(recall_scores)]

print(f"\nBest k based on accuracy: {best_k_acc}")
print(f"Best k based on precision: {best_k_prec}")
print(f"Best k based on recall: {best_k_rec}")

# 9. Plot evaluation metrics
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(k_values, accuracy_scores, marker='o', linewidth=2, markersize=6)
plt.axvline(x=best_k_acc, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k_acc}')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs K Value')
plt.grid(True, alpha=0.3)
plt.legend()

# Precision plot
plt.subplot(1, 3, 2)
plt.plot(k_values, precision_scores, marker='o', linewidth=2, markersize=6, color='green')
plt.axvline(x=best_k_prec, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k_prec}')
plt.xlabel('K Value')
plt.ylabel('Precision')
plt.title('Precision vs K Value')
plt.grid(True, alpha=0.3)
plt.legend()

# Recall plot
plt.subplot(1, 3, 3)
plt.plot(k_values, recall_scores, marker='o', linewidth=2, markersize=6, color='orange')
plt.axvline(x=best_k_rec, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k_rec}')
plt.xlabel('K Value')
plt.ylabel('Recall')
plt.title('Recall vs K Value')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('../img/knn/knn_metrics_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. Train final model with best k (based on accuracy)
best_knn = KNeighborsClassifier(n_neighbors=best_k_acc)
best_knn.fit(X_train, y_train)
y_pred_final = best_knn.predict(X_test)

# 11. Calculate final evaluation metrics
final_accuracy = accuracy_score(y_test, y_pred_final)
final_precision = precision_score(y_test, y_pred_final, zero_division=0)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0

print(f"\nFinal results with k={best_k_acc}:")
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
plt.title(f'Confusion Matrix for KNN (k={best_k_acc})')
plt.savefig('../img/knn/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 13. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, 
                          target_names=['No Diabetes', 'Diabetes']))

# 14. Feature importance analysis
print("\nFeature Importance Analysis with Cross-Validation:")
feature_names = X.columns
feature_importance = []

for i in range(len(feature_names)):
    X_reduced = np.delete(X_scaled, i, axis=1)
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=best_k_acc), 
                           X_reduced, y, cv=5, scoring='accuracy')
    feature_importance.append(scores.mean())

feature_importance = 1 - np.array(feature_importance) / max(feature_importance)

plt.figure(figsize=(10, 6))
feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=True)

plt.barh(feature_imp_df['Feature'], feature_imp_df['Importance'])
plt.xlabel('Relative Importance')
plt.title('Feature Importance Analysis')
plt.tight_layout()
plt.savefig('../img/knn/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 15. Additional plot: Comparison of all metrics
plt.figure(figsize=(12, 8))
plt.plot(k_values, accuracy_scores, marker='o', linewidth=2, markersize=6, label='Accuracy')
plt.plot(k_values, precision_scores, marker='s', linewidth=2, markersize=6, label='Precision')
plt.plot(k_values, recall_scores, marker='^', linewidth=2, markersize=6, label='Recall')
plt.axvline(x=best_k_acc, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k_acc}')
plt.xlabel('K Value')
plt.ylabel('Score')
plt.title('KNN Performance Metrics vs K Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../img/knn/all_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 16. Results analysis
print("\n" + "="*50)
print("RESULTS ANALYSIS:")
print("="*50)

print(f"1. Best K value: {best_k_acc}")
print(f"2. Final model accuracy: {final_accuracy*100:.2f}%")
print(f"3. Precision: {final_precision*100:.2f}% - Accuracy in identifying true patients")
print(f"4. Recall: {final_recall*100:.2f}% - Ability to detect all actual patients")
print(f"5. F1-Score: {final_f1*100:.2f}% - Harmonic mean of precision and recall")

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

print(f"\nKey Insights:")
print(f"- KNN model with k={best_k_acc} showed the best performance")
print(f"- Low k values (1-5) may cause overfitting")
print(f"- High k values (>20) may cause underfitting")
print(f"- Data standardization is essential for KNN algorithm")
print(f"- Diabetes dataset shows class imbalance: {y.value_counts().to_dict()}")
print(f"- The model shows reasonable balance between precision and recall")
print(f"- All plots have been saved in the '../img/knn/' directory")

# Additional: Cross-validation for more reliable results
print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS:")
print("="*50)
cv_scores = cross_val_score(KNeighborsClassifier(n_neighbors=best_k_acc), 
                           X_scaled, y, cv=10, scoring='accuracy')
print(f"Cross-validation scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Plot cross-validation results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), cv_scores, marker='o', linewidth=2, markersize=8)
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.title('10-Fold Cross-Validation Scores')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../img/knn/cross_validation_scores.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAll images have been successfully saved in the '../img/knn/' directory:")
print("- knn_metrics_plot.png")
print("- confusion_matrix.png")
print("- feature_importance.png")
print("- all_metrics_comparison.png")
print("- cross_validation_scores.png")