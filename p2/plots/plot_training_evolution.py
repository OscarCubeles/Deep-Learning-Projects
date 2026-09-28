import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV files generated at train time with training history, change path if needed
simple_conv_df = pd.read_csv('simple_conv_training_history.csv')
deep_conv_df = pd.read_csv('deep_conv_training_history.csv')
simple_sparse_df = pd.read_csv('simple_sparse_training_history.csv')

# Create figure for Simple Convolutional Autoencoder (1 row, 2 cols)
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle('Simple Convolutional Autoencoder - Training Evolution', fontsize=14, fontweight='bold')

# Simple Conv - Plot 1: Loss (Train and Val)
ax = axes1[0]
ax.plot(simple_conv_df['Epoch'], simple_conv_df['Loss'], label='Training Loss', 
        color='#1b5e20', linewidth=2.5, marker='o', markersize=4, markevery=5)
ax.plot(simple_conv_df['Epoch'], simple_conv_df['Val_Loss'], label='Validation Loss', 
        color='#a5d6a7', linewidth=2.5, marker='s', markersize=4, markevery=5)
ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax.set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
ax.set_title('Loss Evolution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

# Simple Conv - Plot 2: MAE (Train and Val)
ax = axes1[1]
ax.plot(simple_conv_df['Epoch'], simple_conv_df['MAE'], label='Training MAE', 
        color='#1b5e20', linewidth=2.5, marker='o', markersize=4, markevery=5)
ax.plot(simple_conv_df['Epoch'], simple_conv_df['Val_MAE'], label='Validation MAE', 
        color='#a5d6a7', linewidth=2.5, marker='s', markersize=4, markevery=5)
ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax.set_ylabel('MAE', fontsize=11, fontweight='bold')
ax.set_title('MAE Evolution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_conv_training_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

# Create figure for Deep Convolutional Autoencoder (1 row, 2 cols)
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Deep Convolutional Autoencoder - Training Evolution', fontsize=14, fontweight='bold')

# Deep Conv - Plot 1: Loss (Train and Val)
ax = axes2[0]
ax.plot(deep_conv_df['Epoch'], deep_conv_df['Loss'], label='Training Loss', 
        color='#388e3c', linewidth=2.5, marker='o', markersize=4, markevery=5)
ax.plot(deep_conv_df['Epoch'], deep_conv_df['Val_Loss'], label='Validation Loss', 
        color='#c8e6c9', linewidth=2.5, marker='s', markersize=4, markevery=5)
ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax.set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
ax.set_title('Loss Evolution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

# Deep Conv - Plot 2: MAE (Train and Val)
ax = axes2[1]
ax.plot(deep_conv_df['Epoch'], deep_conv_df['MAE'], label='Training MAE', 
        color='#388e3c', linewidth=2.5, marker='o', markersize=4, markevery=5)
ax.plot(deep_conv_df['Epoch'], deep_conv_df['Val_MAE'], label='Validation MAE', 
        color='#c8e6c9', linewidth=2.5, marker='s', markersize=4, markevery=5)
ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax.set_ylabel('MAE', fontsize=11, fontweight='bold')
ax.set_title('MAE Evolution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('deep_conv_training_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

# Create figure for Simple Sparse Autoencoder (1 row, 2 cols)
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Simple Sparse Autoencoder (L1) - Training Evolution', fontsize=14, fontweight='bold')

# Simple Sparse - Plot 1: Loss (Train and Val)
ax = axes3[0]
ax.plot(simple_sparse_df['Epoch'], simple_sparse_df['Loss'], label='Training Loss', 
        color='#66bb6a', linewidth=2.5, marker='o', markersize=4, markevery=5)
ax.plot(simple_sparse_df['Epoch'], simple_sparse_df['Val_Loss'], label='Validation Loss', 
        color='#c8e6c9', linewidth=2.5, marker='s', markersize=4, markevery=5)
ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax.set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
ax.set_title('Loss Evolution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

# Simple Sparse - Plot 2: MAE (Train and Val)
ax = axes3[1]
ax.plot(simple_sparse_df['Epoch'], simple_sparse_df['MAE'], label='Training MAE', 
        color='#66bb6a', linewidth=2.5, marker='o', markersize=4, markevery=5)
ax.plot(simple_sparse_df['Epoch'], simple_sparse_df['Val_MAE'], label='Validation MAE', 
        color='#c8e6c9', linewidth=2.5, marker='s', markersize=4, markevery=5)
ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax.set_ylabel('MAE', fontsize=11, fontweight='bold')
ax.set_title('MAE Evolution', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_sparse_training_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print(f'\nSimple Convolutional Autoencoder:')
print(f'  Epochs: {len(simple_conv_df)}')
print(f'  Initial Loss: {simple_conv_df["Loss"].iloc[0]:.6f}')
print(f'  Final Loss: {simple_conv_df["Loss"].iloc[-1]:.6f}')
print(f'  Loss Reduction: {((simple_conv_df["Loss"].iloc[0] - simple_conv_df["Loss"].iloc[-1]) / simple_conv_df["Loss"].iloc[0] * 100):.2f}%')
print(f'  Initial Val_Loss: {simple_conv_df["Val_Loss"].iloc[0]:.6f}')
print(f'  Final Val_Loss: {simple_conv_df["Val_Loss"].iloc[-1]:.6f}')
print(f'  Best Val_Loss: {simple_conv_df["Val_Loss"].min():.6f} (Epoch {simple_conv_df["Val_Loss"].idxmin() + 1})')
print(f'  Final MAE: {simple_conv_df["MAE"].iloc[-1]:.6f}')
print(f'  Final Val_MAE: {simple_conv_df["Val_MAE"].iloc[-1]:.6f}')

print(f'\nDeep Convolutional Autoencoder:')
print(f'  Epochs: {len(deep_conv_df)}')
print(f'  Initial Loss: {deep_conv_df["Loss"].iloc[0]:.6f}')
print(f'  Final Loss: {deep_conv_df["Loss"].iloc[-1]:.6f}')
print(f'  Loss Reduction: {((deep_conv_df["Loss"].iloc[0] - deep_conv_df["Loss"].iloc[-1]) / deep_conv_df["Loss"].iloc[0] * 100):.2f}%')
print(f'  Initial Val_Loss: {deep_conv_df["Val_Loss"].iloc[0]:.6f}')
print(f'  Final Val_Loss: {deep_conv_df["Val_Loss"].iloc[-1]:.6f}')
print(f'  Best Val_Loss: {deep_conv_df["Val_Loss"].min():.6f} (Epoch {deep_conv_df["Val_Loss"].idxmin() + 1})')
print(f'  Final MAE: {deep_conv_df["MAE"].iloc[-1]:.6f}')
print(f'  Final Val_MAE: {deep_conv_df["Val_MAE"].iloc[-1]:.6f}')

print(f'\nSimple Sparse Autoencoder (L1):')
print(f'  Epochs: {len(simple_sparse_df)}')
print(f'  Initial Loss: {simple_sparse_df["Loss"].iloc[0]:.6f}')
print(f'  Final Loss: {simple_sparse_df["Loss"].iloc[-1]:.6f}')
print(f'  Loss Reduction: {((simple_sparse_df["Loss"].iloc[0] - simple_sparse_df["Loss"].iloc[-1]) / simple_sparse_df["Loss"].iloc[0] * 100):.2f}%')
print(f'  Initial Val_Loss: {simple_sparse_df["Val_Loss"].iloc[0]:.6f}')
print(f'  Final Val_Loss: {simple_sparse_df["Val_Loss"].iloc[-1]:.6f}')
print(f'  Best Val_Loss: {simple_sparse_df["Val_Loss"].min():.6f} (Epoch {simple_sparse_df["Val_Loss"].idxmin() + 1})')
print(f'  Final MAE: {simple_sparse_df["MAE"].iloc[-1]:.6f}')
print(f'  Final Val_MAE: {simple_sparse_df["Val_MAE"].iloc[-1]:.6f}')
