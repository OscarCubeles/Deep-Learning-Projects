import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Parse the CSV file generated from autoencoder evaluations, change path if needed
csv_file = 'evaluation_results.csv'
df = pd.read_csv(csv_file)

# Extract data for the three models
models_of_interest = ['Deep Convolutional Autoencoder', 'Simple Convolutional Autoencoder', 
                      'Simple Sparse Autoencoder (L1)']
df_filtered = df[df['Model'].isin(models_of_interest)].reset_index(drop=True)

# Function to extract mean and std from format "0.002362 ± 0.002804"
def parse_error_string(error_str):
    match = re.match(r'([\d.]+)\s*±\s*([\d.]+)', error_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

# Extract mean and std for each model
models = df_filtered['Model'].values
val_means = []
val_stds = []
unseen_means = []
unseen_stds = []
seen_means = []
seen_stds = []

for idx, row in df_filtered.iterrows():
    # Validation
    val_mean, val_std = parse_error_string(row['Val_Mean_Error±Std'])
    val_means.append(val_mean)
    val_stds.append(val_std)
    
    # Test Unseen
    unseen_mean, unseen_std = parse_error_string(row['Test_Unseen_Mean_Error±Std'])
    unseen_means.append(unseen_mean)
    unseen_stds.append(unseen_std)
    
    # Test Seen
    seen_mean, seen_std = parse_error_string(row['Test_Seen_Mean_Error±Std'])
    seen_means.append(seen_mean)
    seen_stds.append(seen_std)

# Create figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Error bar colors (green scale)
colors = ['#1b5e20', '#388e3c', '#66bb6a']

# Plot for each model
for idx, (ax, model) in enumerate(zip(axes, models)):
    # X positions for error categories
    x = np.arange(3)
    width = 0.6
    
    # Data for this model
    means = [val_means[idx], unseen_means[idx], seen_means[idx]]
    stds = [val_stds[idx], unseen_stds[idx], seen_stds[idx]]
    categories = ['Validation', 'Test Unseen\n(OOD)', 'Test Seen\n(In-Dist)']
    
    # Plot error bars
    bars = ax.bar(x, means, width, yerr=stds, label='', 
                  color=[colors[0], colors[1], colors[2]], capsize=5, alpha=0.8, 
                  error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Labels and formatting
    ax.set_ylabel('Reconstruction Error (MSE)', fontsize=11, fontweight='bold')
    ax.set_title(model.replace(' Autoencoder', '').replace(' (L1)', ''), fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('error_bars_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary
for i, model in enumerate(models):
    print(f'\n{model}:')
    print(f'  Validation:        {val_means[i]:.6f} ± {val_stds[i]:.6f}')
    print(f'  Test Unseen (OOD): {unseen_means[i]:.6f} ± {unseen_stds[i]:.6f}')
    print(f'  Test Seen (In-Dist): {seen_means[i]:.6f} ± {seen_stds[i]:.6f}')
    print(f'  Error Separation (Unseen/Seen ratio): {unseen_means[i]/seen_means[i]:.2f}x')
