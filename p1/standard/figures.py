import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves_comparison_single_row():
    # Models and configurations
    models = ['AlexNet', 'StandardArchitectureModel']
    configs = ['balanced', 'underfitting', 'overfitting']

    # Create figure with 1 row, 2 columns (one column per model)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Define colors for each configuration (train/val pairs)
    config_colors = {
        'balanced': {
            'train': '#1f4e79',   # Deep Navy Blue
            'val':   '#4a86c5'    # Medium Steel Blue
        },
        'underfitting': {
            'train': '#5e3c99',   # Deep Academic Purple
            'val':   '#b2abd2'    # Soft Lilac
        },
        'overfitting': {
            'train': '#2ca02c',   # Deep Green
            'val':   '#8dd18d'    # Light Green
        }
    }

    # Define line styles
    line_styles = {
        'train': '-',
        'val': '--'
    }

    # Plot for each model (column)
    for i, model in enumerate(models):
        ax = axes[i]
        
        for config in configs:
            try:
                df = pd.read_csv(f'results/{model}/{config}.csv')
                
                # Filter out rows where Epoch is NaN or contains non-numeric data
                df = df[pd.to_numeric(df['Epoch'], errors='coerce').notna()]
                df['Epoch'] = pd.to_numeric(df['Epoch'])
                
                # Sort by epoch to handle unsorted data
                df = df.sort_values('Epoch')
                
                # Remove duplicate epochs (keep first occurrence)
                df = df.drop_duplicates(subset='Epoch', keep='first')
                
                # Plot training accuracy
                ax.plot(df['Epoch'], df['Train Accuracy'], 
                    color=config_colors[config]['train'], 
                    linestyle=line_styles['train'],
                    linewidth=2.5,
                    label=f'{config.capitalize()} Train')
                
                # Plot validation accuracy
                ax.plot(df['Epoch'], df['Val Accuracy'], 
                    color=config_colors[config]['val'], 
                    linestyle=line_styles['val'],
                    linewidth=2.5,
                    label=f'{config.capitalize()} Val')
                
            except Exception as e:
                print(f"Warning: Could not load {model}/{config}.csv - {e}")
        
        # Title for each subplot
        ax.set_title(f'{model}', fontsize=13, fontweight='bold')
        
        # Labels
        ax.set_xlabel('Epoch', fontsize=11)
        if i == 0:
            ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_ylim([0, 1])
        ax.set_xlim([0, None])

    plt.tight_layout()
    plt.savefig('learning_curves_comparison_single_row.png', dpi=600, bbox_inches='tight')
    plt.show()

def plot_loss_accuracy_lr_single_model_linear(model='ResNet18'):
    
    configs = ['balanced', 'underfitting', 'overfitting']
    
    # Create 2x3 subplot
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f'{model} - All Configurations (Linear Scale)', fontsize=16, fontweight='bold')
    
    # Define colors (same as plot_loss_accuracy_single_model)
    train_loss_color = '#1f4e79'   # Deep Navy Blue
    val_loss_color   = '#4a86c5'   # Medium Steel Blue
    train_acc_color  = '#2ca02c'   # Deep Green
    val_acc_color    = '#8dd18d'   # Light Green
    lr_color         = '#f39c12'   # Orange
        
    for i, config in enumerate(configs):
        try:
            df = pd.read_csv(f'results/{model}/{config}.csv')
            
            # Filter out rows where Epoch is NaN or contains non-numeric data
            df = df[pd.to_numeric(df['Epoch'], errors='coerce').notna()]
            df['Epoch'] = pd.to_numeric(df['Epoch'])
            
            # Sort by epoch
            df = df.sort_values('Epoch')
            
            # Remove duplicate epochs
            df = df.drop_duplicates(subset='Epoch', keep='first')
            
            # Row 1: Loss and Accuracy with dual y-axes
            ax1 = axes[0, i]
            ax2 = ax1.twinx()
            
            # Plot losses on left y-axis
            line1 = ax1.plot(df['Epoch'], df['Train Loss'], 
                            color=train_loss_color, linewidth=2.5, linestyle='-',
                            label='Train Loss')
            line2 = ax1.plot(df['Epoch'], df['Val Loss'], 
                            color=val_loss_color, linewidth=2.5, linestyle='-',
                            label='Val Loss')
            
            # Plot accuracies on right y-axis
            line3 = ax2.plot(df['Epoch'], df['Train Accuracy'], 
                            color=train_acc_color, linewidth=2.5, linestyle='--',
                            label='Train Accuracy')
            line4 = ax2.plot(df['Epoch'], df['Val Accuracy'], 
                            color=val_acc_color, linewidth=2.5, linestyle='--',
                            label='Val Accuracy')
            
            ax1.set_title(f'{config.capitalize()}', fontsize=13, fontweight='bold')
            ax1.set_ylabel('Loss', fontsize=11)
            ax2.set_ylabel('Accuracy', fontsize=11)
            ax2.set_ylim([0, 1])
            
            # Combine legends
            lines = line1 + line2 + line3 + line4
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='best', fontsize=9)
            ax1.grid(alpha=0.3, linestyle=':', linewidth=0.5)
            
            # Row 2: Learning Rate (LINEAR scale)
            ax_lr = axes[1, i]
            ax_lr.plot(df['Epoch'], df['lr'], 
                      color=lr_color, linewidth=2.5, linestyle='-',
                      label='Learning Rate')
            ax_lr.set_xlabel('Epoch', fontsize=11)
            ax_lr.set_ylabel('Learning Rate', fontsize=11)
            ax_lr.legend(loc='best', fontsize=9)
            ax_lr.grid(alpha=0.3, linestyle=':', linewidth=0.5)
            ax_lr.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
        except Exception as e:
            print(f"Warning: Could not load {model}/{config}.csv - {e}")
            axes[0, i].text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=axes[0, i].transAxes, fontsize=12)
            axes[0, i].set_title(f'{config.capitalize()}', fontsize=13, fontweight='bold')
            axes[1, i].text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=axes[1, i].transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{model}_all_configs_loss_accuracy_lr_linear.png', dpi=600, bbox_inches='tight')
    plt.show()

plot_learning_curves_comparison_single_row()
plot_loss_accuracy_lr_single_model_linear(model='StandardArchitectureModel')