import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Parse the CSV file generated from autoencoder evaluations, change path if needed
csv_file = 'evaluation_results.csv'
df = pd.read_csv(csv_file)

# Extract data for the three autoencoders
models_of_interest = ['Deep Convolutional Autoencoder', 
                      'Simple Convolutional Autoencoder', 'Simple Sparse Autoencoder (L1)']
df_filtered = df[df['Model'].isin(models_of_interest)].reset_index(drop=True)

# Extract metrics
models = df_filtered['Model'].values
accuracy = df_filtered['Combined_Accuracy(%)'].values
tpr = df_filtered['Combined_TPR(%)'].values
fpr = df_filtered['Combined_FPR(%)'].values
specificity = df_filtered['Combined_Specificity(%)'].values

# Create figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Green color palette
colors = ['#1b5e20', '#388e3c', '#66bb6a', '#a5d6a7']

# Plot for each model
for idx, (ax, model) in enumerate(zip(axes, models)):
    # X positions for metrics
    x = np.arange(4)
    width = 0.6
    
    # Data for this model
    metrics = [accuracy[idx], tpr[idx], fpr[idx], specificity[idx]]
    metric_names = ['Accuracy', 'TPR', 'FPR', 'Specificity']
    
    # Plot bars
    bars = ax.bar(x, metrics, width, 
                  color=[colors[0], colors[1], colors[2]], alpha=0.8)
    
    # Labels and formatting
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title(model.replace(' Autoencoder', '').replace(' (L1)', ''), fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Model Performance Metrics Comparison', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('model_comparison_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary
for i, model in enumerate(models):
    print(f'\n{model}:')
    print(f'  Accuracy: {accuracy[i]:.2f}%')
    print(f'  TPR (True Positive Rate): {tpr[i]:.2f}%')
    print(f'  FPR (False Positive Rate): {fpr[i]:.2f}%')
    print(f'  Specificity: {specificity[i]:.2f}%')
