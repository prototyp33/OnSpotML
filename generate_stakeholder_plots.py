import matplotlib.pyplot as plt
import numpy as np
import json

# Load the metrics to get feature importance
with open('reports/metrics/main/manual_tscv_evaluation_metrics.json', 'r') as f:
    metrics = json.load(f)

# Extract average feature importances
feature_imp = metrics['average_feature_importances']

# Get top 15 features
top_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:15]
features, importances = zip(*top_features)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(features))

# Create horizontal bar chart
bars = ax.barh(y_pos, importances, color='steelblue', alpha=0.8)

# Customize the plot
ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.invert_yaxis()  # Highest importance at top
ax.set_xlabel('Feature Importance Score', fontsize=12)
ax.set_title('Top 15 Most Important Features\nBarcelona Parking Occupancy Prediction Model', fontsize=14, fontweight='bold')

# Add value labels on bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + max(importances)*0.01, bar.get_y() + bar.get_height()/2, 
            f'{int(width):,}', ha='left', va='center', fontsize=10)

# Improve layout
plt.tight_layout()
plt.grid(axis='x', alpha=0.3)

# Save the plot
plt.savefig('feature_importance_stakeholder_report.png', dpi=300, bbox_inches='tight')
print('Feature importance plot saved as feature_importance_stakeholder_report.png')
plt.close()

# Also create a summary metrics plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: CV Accuracy across folds
fold_names = [m['fold'] for m in metrics['fold_metrics']]
accuracies = [m['accuracy'] for m in metrics['fold_metrics']]
ax1.bar(range(len(fold_names)), accuracies, color='lightcoral', alpha=0.8)
ax1.set_xlabel('CV Fold')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy Across CV Folds')
ax1.set_xticks(range(len(fold_names)))
ax1.set_xticklabels([f'Fold {i+1}' for i in range(len(fold_names))], rotation=45)
for i, acc in enumerate(accuracies):
    ax1.text(i, acc + 0.005, f'{acc:.3f}', ha='center', va='bottom')

# Plot 2: F1 scores across folds
f1_scores = [m['f1_weighted'] for m in metrics['fold_metrics']]
ax2.bar(range(len(fold_names)), f1_scores, color='lightgreen', alpha=0.8)
ax2.set_xlabel('CV Fold')
ax2.set_ylabel('Weighted F1 Score')
ax2.set_title('Model F1 Score Across CV Folds')
ax2.set_xticks(range(len(fold_names)))
ax2.set_xticklabels([f'Fold {i+1}' for i in range(len(fold_names))], rotation=45)
for i, f1 in enumerate(f1_scores):
    ax2.text(i, f1 + 0.005, f'{f1:.3f}', ha='center', va='bottom')

# Plot 3: Average metrics with error bars
metrics_names = ['Accuracy', 'F1 Score']
means = [metrics['average_accuracy'], metrics['average_f1_weighted']]
stds = [metrics['std_accuracy'], metrics['std_f1_weighted']]
ax3.bar(metrics_names, means, yerr=stds, capsize=5, color=['orange', 'purple'], alpha=0.7)
ax3.set_ylabel('Score')
ax3.set_title('Average Model Performance with Standard Deviation')
for i, (mean, std) in enumerate(zip(means, stds)):
    ax3.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')

# Plot 4: Feature category distribution (top features only)
feature_categories = {
    'Temporal Lag': ['actual_state_lag_1h', 'actual_state_lag_6h', 'actual_state_lag_12h', 'actual_state_lag_24h', 'actual_state_lag_48h', 'actual_state_lag_168h'],
    'Temporal Cycle': ['hour', 'hour_cos', 'hour_sin', 'dayofweek', 'dayofweek_cos', 'dayofweek_sin', 'dayofyear', 'month_cos', 'month_sin'],
    'Occupancy Derived': ['VALOR', 'occupancy_acceleration'],
    'Calendar/Holiday': ['days_to_holiday', 'is_holiday', 'is_near_holiday'],
    'Peak/Event': ['super_peak', 'is_morning_peak', 'is_evening_peak', 'is_lunch_hours']
}

category_importance = {}
for category, feature_list in feature_categories.items():
    total_imp = sum(feature_imp.get(f, 0) for f in feature_list if f in feature_imp)
    category_importance[category] = total_imp

sorted_categories = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
cat_names, cat_values = zip(*sorted_categories)

ax4.pie(cat_values, labels=cat_names, autopct='%1.1f%%', startangle=90)
ax4.set_title('Feature Importance by Category')

plt.tight_layout()
plt.savefig('model_performance_summary_stakeholder_report.png', dpi=300, bbox_inches='tight')
print('Model performance summary saved as model_performance_summary_stakeholder_report.png')
plt.close()

print('\n=== FINAL MODEL PERFORMANCE METRICS ===')
print(f'Average Accuracy: {metrics["average_accuracy"]:.4f} ± {metrics["std_accuracy"]:.4f}')
print(f'Average F1 Score: {metrics["average_f1_weighted"]:.4f} ± {metrics["std_f1_weighted"]:.4f}')
print('\nNote: These are classification metrics for 7-class occupancy prediction (0-6 scale)') 