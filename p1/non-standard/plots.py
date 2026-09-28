import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

def plot_training_history(csv_path, output_filename="training_metrics.png", model_name="Model"):
    try:
        df = pd.read_csv(csv_path)
        df = df.iloc[:-1]

    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}. Please check the path.")
        return

    df['Epoch'] = pd.to_numeric(df['Epoch'], errors='coerce')
    df['Val Accuracy'] = pd.to_numeric(df['Val Accuracy'], errors='coerce')

    # Find epoch with best validation accuracy
    best_val_acc = df['Val Accuracy'].max()
    best_epoch_row = df.loc[df['Val Accuracy'].idxmax()]
    best_epoch = int(best_epoch_row['Epoch'])
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'{model_name} | Max Val Accuracy: {best_val_acc:.4f} at epoch {best_epoch}', 
        fontsize=16
    )

    # Plot loss
    ax1 = axes[0]
    ax1.plot(df['Epoch'], df['Train Loss'], label='Train Loss', color='darkred', linestyle='--')
    ax1.plot(df['Epoch'], df['Val Loss'], label='Val Loss', color='goldenrod', linestyle='-')
    ax1.axvline(x=best_epoch, color='gray', linestyle=':', alpha=0.7) # Highlight best epoch
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot accuracy
    ax2 = axes[1]
    ax2.plot(df['Epoch'], df['Train Accuracy'], label='Train Accuracy', color='darkgreen', linestyle='--')
    ax2.plot(df['Epoch'], df['Val Accuracy'], label='Val Accuracy', color='teal', linestyle='-')
    ax2.axvline(x=best_epoch, color='gray', linestyle=':', alpha=0.7)
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    
    # Plot learning rate
    ax3 = axes[2]
    ax3.plot(df['Epoch'], df['lr'], color='navy', linewidth=2)
    ax3.set_title('Learning Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    print(f"Plot saved successfully as {output_filename}")


csv_path = "results/inception_underfitting.csv" 
plot_training_history(csv_path, "results/inception_underfitting.png", model_name="InceptionV3")