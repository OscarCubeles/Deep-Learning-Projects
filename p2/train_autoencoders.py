import os
import csv
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from keras import layers

# Import all autoencoder models
from configurations.configurations_autoencoders import CONFIGS
from german_dataset import build_datasets_from_zip

# Data paths
root = r'/home/nct/nct01150/dl_openset/data/archive.zip'
path_unseen = r'/home/nct/nct01150/dl_openset/data'

def add_noise(images, noise_factor):
    """Add Gaussian noise to images for denoising autoencoder."""
    noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=noise_factor)
    noisy_images = images + noise
    noisy_images = tf.clip_by_value(noisy_images, 0.0, 1.0)
    return noisy_images


def train_standard_autoencoder(config, model, train_ds, val_ds, out_dir):
    """Train standard autoencoder (including denoising)."""
    ae_config = config.get('autoencoder', {})
    model_type = config.get('model_type', 'standard')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=ae_config.get('learning_rate', 0.001)),
        loss='mse',
        metrics=['mae']
    )
    
    num_epochs = ae_config.get('num_epochs', 50)
    
    # Prepare datasets
    if model_type == 'denoising': # DAE was discarded but kept for reference
        noise_factor = ae_config.get('noise_factor', 0.3)
        def prepare_data(ds):
            def add_noise_wrapper(img, label):
                noisy_img = add_noise(img, noise_factor)
                return noisy_img, img
            return ds.map(lambda img, label: add_noise_wrapper(img, label))
    else:
        def prepare_data(ds):
            return ds.map(lambda img, label: (img, img))
    
    train_ae = prepare_data(train_ds)
    val_ae = prepare_data(val_ds)
    
    # Train
    history = model.fit(
        train_ae,
        validation_data=val_ae,
        epochs=num_epochs,
        verbose=1
    )
    
    return model, history


def save_training_history(history, ae_name, out_dir):
    """Save training history to CSV file."""
    history_path = os.path.join(out_dir, f'{ae_name}_training_history.csv')
    
    # Extract history data
    epochs = len(history.history['loss'])
    
    with open(history_path, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Loss', 'Val_Loss', 'MAE', 'Val_MAE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for epoch in range(epochs):
            row = {
                'Epoch': epoch + 1,
                'Loss': history.history['loss'][epoch],
                'Val_Loss': history.history['val_loss'][epoch],
                'MAE': history.history.get('mae', ['N/A'])[epoch] if 'mae' in history.history else 'N/A',
                'Val_MAE': history.history.get('val_mae', ['N/A'])[epoch] if 'val_mae' in history.history else 'N/A',
            }
            writer.writerow(row)
    
    print(f'Training history saved to: {history_path}')
    return history_path


def compute_reconstruction_errors(model, dataset):
    """Compute reconstruction error for each sample."""
    errors = []
    
    for images, _ in dataset:
        reconstructed = model(images, training=False)
        mse_per_sample = tf.reduce_mean(tf.square(images - reconstructed), axis=[1, 2, 3])
        errors.extend(mse_per_sample.numpy())
    
    return np.array(errors)


def evaluate_ood_detection(model, model_name, val_ds, test_ds, out_dir):
    """Evaluate OOD detection performance."""
    print(f'\n{"="*60}')
    print(f'EVALUATING {model_name}')
    print(f'{"="*60}')
    print('\nComputing reconstruction errors...')
    
    val_errors = compute_reconstruction_errors(model, val_ds)
    test_errors = compute_reconstruction_errors(model, test_ds)
    
    # Statistics
    stats = {
        'model_name': model_name,
        'val_mean': np.mean(val_errors),
        'val_std': np.std(val_errors),
        'val_min': np.min(val_errors),
        'val_max': np.max(val_errors),
        'test_mean': np.mean(test_errors),
        'test_std': np.std(test_errors),
        'test_min': np.min(test_errors),
        'test_max': np.max(test_errors),
    }
    
    # Find optimal threshold
    k = 1.0
    threshold = stats['val_mean'] + k * stats['val_std']
    stats['threshold'] = threshold
    
    # Evaluate separation
    val_ood = (val_errors > threshold).sum()
    test_ood = (test_errors > threshold).sum()
    
    stats['val_ood_detected'] = int(val_ood)
    stats['val_total'] = len(val_errors)
    stats['test_ood_detected'] = int(test_ood)
    stats['test_total'] = len(test_errors)
    stats['false_positive_rate'] = val_ood / len(val_errors)
    stats['true_positive_rate'] = test_ood / len(test_errors)
    
    # Print results
    print(f'\nKnown Classes (Validation):')
    print(f'  Mean Error: {stats["val_mean"]:.6f} ± {stats["val_std"]:.6f}')
    print(f'  Range: [{stats["val_min"]:.6f}, {stats["val_max"]:.6f}]')
    print(f'  Flagged as OOD: {val_ood}/{len(val_errors)} ({stats["false_positive_rate"]*100:.2f}%)')
    
    print(f'\nUnknown Classes (Test):')
    print(f'  Mean Error: {stats["test_mean"]:.6f} ± {stats["test_std"]:.6f}')
    print(f'  Range: [{stats["test_min"]:.6f}, {stats["test_max"]:.6f}]')
    print(f'  Flagged as OOD: {test_ood}/{len(test_errors)} ({stats["true_positive_rate"]*100:.2f}%)')
    
    print(f'\nThreshold: {threshold:.6f}')
    print(f'FPR: {stats["false_positive_rate"]*100:.2f}% | TPR: {stats["true_positive_rate"]*100:.2f}%')
    
    return stats


def main():
    """Main training and evaluation pipeline for all autoencoders."""
    
    print('='*80)
    print('TRAINING ALL AUTOENCODERS FOR OOD DETECTION')
    print('='*80)
    
    # Setup main output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_out_dir = os.path.join('outputs', f'all_autoencoders_{timestamp}')
    os.makedirs(main_out_dir, exist_ok=True)
    

    
    # Store all results
    all_results = []
    
    # Train each autoencoder
    for ae_name, config in CONFIGS.items():
        print(f'\n\n{"#"*80}')
        print(f'# {ae_name.upper()}: {config["name"]}')
        print(f'{"#"*80}')
        
        # Create specific output directory
        ae_out_dir = os.path.join(main_out_dir, ae_name)
        os.makedirs(ae_out_dir, exist_ok=True)
        
        # Load datasets
        print('\nLoading datasets...')
        extract_dir = os.path.join(ae_out_dir, 'extracted_data')
        train_ds, val_ds, test_ds, n_train, n_val, n_test, num_classes = build_datasets_from_zip(
            root,
            img_size=config['autoencoder']['input_size'],
            batch_size=config['batch_size'],
            val_split=config['val_split'],
            extract_to=extract_dir,
            path_unseen=path_unseen,
        )
        
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
        
        print(f'Training: {n_train}, Validation: {n_val}, Test: {n_test}')
        
        # Build and train model
        print(f'\nTraining {config["name"]}...')
        model_type = config.get('model_type', 'standard')
        
        try:
            if model_type != 'sparse_kl':
                # Build standard autoencoder
                model = config['model_fn']()
                print('Model built successfully')
                
                # Train standard
                model, history = train_standard_autoencoder(config, model, train_ds, val_ds, ae_out_dir)
            
            # Print training summary
            print(f'\nTraining completed!')
            print(f'Final Train Loss: {history.history["loss"][-1]:.6f}')
            print(f'Final Val Loss: {history.history["val_loss"][-1]:.6f}')
            
            # Save training history to CSV
            save_training_history(history, ae_name, ae_out_dir)
            
            # Save model
            model_path = os.path.join(ae_out_dir, f'{ae_name}_model.keras')
            model.save(model_path)
            print(f'Model saved to {model_path}')
            
            # Evaluate OOD detection
            stats = evaluate_ood_detection(model, config['name'], val_ds, test_ds, ae_out_dir)
            
            # Add training info to stats
            stats['final_train_loss'] = float(history.history["loss"][-1])
            stats['final_val_loss'] = float(history.history["val_loss"][-1])
            
            all_results.append(stats)
            
            # Save individual stats
            stats_path = os.path.join(ae_out_dir, f'{ae_name}_stats.txt')
            with open(stats_path, 'w') as f:
                f.write(f'{config["name"]} OOD DETECTION RESULTS\n')
                f.write('='*60 + '\n\n')
                for key, value in stats.items():
                    f.write(f'{key}: {value}\n')
            
            print(f'\n✓ {ae_name} completed successfully!')
            
        except Exception as e:
            print(f'\n✗ Error training {ae_name}: {str(e)}')
            import traceback
            traceback.print_exc()
            continue
    
    # Save comparison results
    print(f'\n\n{"="*80}')
    print('SUMMARY OF ALL AUTOENCODERS')
    print(f'{"="*80}')
    
    comparison_path = os.path.join(main_out_dir, 'comparison_results.txt')
    with open(comparison_path, 'w') as f:
        f.write('COMPARISON OF ALL AUTOENCODERS FOR OOD DETECTION\n')
        f.write('='*80 + '\n\n')
        
        # Header
        f.write(f'{"Model":<35} {"TPR":<10} {"FPR":<10} {"Val Error":<15} {"Test Error":<15}\n')
        f.write('-'*85 + '\n')
        
        print(f'\n{"Model":<35} {"TPR":<10} {"FPR":<10} {"Val Error":<15} {"Test Error":<15}')
        print('-'*85)
        
        for stats in all_results:
            line = (f'{stats["model_name"]:<35} '
                   f'{stats["true_positive_rate"]*100:<10.2f} '
                   f'{stats["false_positive_rate"]*100:<10.2f} '
                   f'{stats["val_mean"]:<15.6f} '
                   f'{stats["test_mean"]:<15.6f}')
            f.write(line + '\n')
            print(line)
        
        f.write('\n' + '='*80 + '\n')
        f.write('\nBest TPR (True Positive Rate): ')
        best_tpr = max(all_results, key=lambda x: x['true_positive_rate'])
        f.write(f'{best_tpr["model_name"]} ({best_tpr["true_positive_rate"]*100:.2f}%)\n')
        
        f.write('Best FPR (False Positive Rate): ')
        best_fpr = min(all_results, key=lambda x: x['false_positive_rate'])
        f.write(f'{best_fpr["model_name"]} ({best_fpr["false_positive_rate"]*100:.2f}%)\n')
        
        f.write('Best Separation (Test/Val ratio): ')
        best_sep = max(all_results, key=lambda x: x['test_mean'] / x['val_mean'] if x['val_mean'] > 0 else 0)
        separation_ratio = best_sep['test_mean'] / best_sep['val_mean'] if best_sep['val_mean'] > 0 else 0
        f.write(f'{best_sep["model_name"]} ({separation_ratio:.2f}x)\n')
    
    print(f'\n{"="*80}')
    print(f'All results saved to: {main_out_dir}')
    print(f'Comparison table: {comparison_path}')
    print('Done!')


if __name__ == '__main__':
    main()
