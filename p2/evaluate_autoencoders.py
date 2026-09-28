import os
import numpy as np
from datetime import datetime

import tensorflow as tf
from keras import models

from german_dataset import build_datasets_from_zip_eval

# Data paths
root = r'/home/nct/nct01150/dl_openset/data/archive.zip'
path_unseen = r'/home/nct/nct01150/dl_openset/data'

def compute_reconstruction_errors(model, dataset):
    """Compute reconstruction error for each sample."""
    errors = []
    
    for images, _ in dataset:
        reconstructed = model(images, training=False)
        mse_per_sample = tf.reduce_mean(tf.square(images - reconstructed), axis=[1, 2, 3])
        errors.extend(mse_per_sample.numpy())
    
    return np.array(errors)


def evaluate_ood_detection(model, model_name, val_ds, test_ds_unseen, test_ds_seen, out_dir):
    """Evaluate OOD detection performance with separate unseen and seen test sets."""
    print(f'EVALUATING {model_name}')
    print('\nComputing reconstruction errors...')
    
    val_errors = compute_reconstruction_errors(model, val_ds)
    test_errors_unseen = compute_reconstruction_errors(model, test_ds_unseen)
    test_errors_seen = compute_reconstruction_errors(model, test_ds_seen)
    
    # Combine all test errors for overall metrics
    test_errors_all = np.concatenate([test_errors_unseen, test_errors_seen])
    
    # Statistics
    stats = {
        'model_name': model_name,
        # Validation (known classes)
        'val_mean': np.mean(val_errors),
        'val_std': np.std(val_errors),
        'val_min': np.min(val_errors),
        'val_max': np.max(val_errors),
        # Test Unseen (OOD classes)
        'test_unseen_mean': np.mean(test_errors_unseen),
        'test_unseen_std': np.std(test_errors_unseen),
        'test_unseen_min': np.min(test_errors_unseen),
        'test_unseen_max': np.max(test_errors_unseen),
        # Test Seen (In-distribution classes from test set)
        'test_seen_mean': np.mean(test_errors_seen),
        'test_seen_std': np.std(test_errors_seen),
        'test_seen_min': np.min(test_errors_seen),
        'test_seen_max': np.max(test_errors_seen),
        # Test All Combined
        'test_all_mean': np.mean(test_errors_all),
        'test_all_std': np.std(test_errors_all),
        'test_all_min': np.min(test_errors_all),
        'test_all_max': np.max(test_errors_all),
    }
    
    # Find optimal threshold based on validation set
    k = 1.0
    threshold = stats['val_mean'] + k * stats['val_std']
    stats['threshold'] = threshold
    
    # Evaluate separation on validation (known classes)
    val_ood = (val_errors > threshold).sum()
    
    # Evaluate separation on test sets
    test_ood_unseen = (test_errors_unseen > threshold).sum()
    test_ood_seen = (test_errors_seen > threshold).sum()
    test_ood_all = (test_errors_all > threshold).sum()
    
    # Calculate metrics
    stats['val_ood_detected'] = int(val_ood)
    stats['val_total'] = len(val_errors)
    
    # Test unseen metrics (OOD - higher detection rate is better)
    stats['test_unseen_ood_detected'] = int(test_ood_unseen)
    stats['test_unseen_total'] = len(test_errors_unseen)
    stats['true_positive_rate'] = test_ood_unseen / len(test_errors_unseen)  # Detect unseen classes
    
    # Test seen metrics (In-distribution - lower detection rate is better)
    stats['test_seen_ood_detected'] = int(test_ood_seen)
    stats['test_seen_total'] = len(test_errors_seen)
    stats['false_positive_rate'] = test_ood_seen / len(test_errors_seen)  # Incorrectly flag seen as OOD
    
    # Overall test metrics (combined)
    stats['test_all_ood_detected'] = int(test_ood_all)
    stats['test_all_total'] = len(test_errors_all)
    
    # Combined metrics
    # TP = unseen correctly detected as OOD
    # TN = seen correctly identified as in-distribution
    # FP = seen incorrectly flagged as OOD
    # FN = unseen not flagged as OOD
    tp = test_ood_unseen
    tn = len(test_errors_seen) - test_ood_seen
    fp = test_ood_seen
    fn = len(test_errors_unseen) - test_ood_unseen
    
    stats['combined_tp'] = int(tp)
    stats['combined_tn'] = int(tn)
    stats['combined_fp'] = int(fp)
    stats['combined_fn'] = int(fn)
    
    # Accuracy: (TP + TN) / (TP + TN + FP + FN)
    stats['combined_accuracy'] = (tp + tn) / len(test_errors_all)
    
    # Combined TPR (True Positive Rate on unseen) - same as true_positive_rate
    stats['combined_tpr'] = stats['true_positive_rate']
    
    # Combined FPR (False Positive Rate on seen) - same as false_positive_rate
    stats['combined_fpr'] = stats['false_positive_rate']
    
    # Combined Specificity/TNR (True Negative Rate on seen)
    stats['combined_specificity'] = tn / len(test_errors_seen) if len(test_errors_seen) > 0 else 0
    
    # Print results
    print(f'\nValidation Set (Known Classes):')
    print(f'  Mean Error: {stats["val_mean"]:.6f} ± {stats["val_std"]:.6f}')
    print(f'  Range: [{stats["val_min"]:.6f}, {stats["val_max"]:.6f}]')
    print(f'  Flagged as OOD: {val_ood}/{len(val_errors)} ({(val_ood/len(val_errors))*100:.2f}%)')
    
    print(f'\nTest Set - Unseen Classes (OOD):')
    print(f'  Mean Error: {stats["test_unseen_mean"]:.6f} ± {stats["test_unseen_std"]:.6f}')
    print(f'  Range: [{stats["test_unseen_min"]:.6f}, {stats["test_unseen_max"]:.6f}]')
    print(f'  Flagged as OOD: {test_ood_unseen}/{len(test_errors_unseen)} (TPR: {stats["true_positive_rate"]*100:.2f}%)')
    
    print(f'\nTest Set - Seen Classes (In-Distribution):')
    print(f'  Mean Error: {stats["test_seen_mean"]:.6f} ± {stats["test_seen_std"]:.6f}')
    print(f'  Range: [{stats["test_seen_min"]:.6f}, {stats["test_seen_max"]:.6f}]')
    print(f'  Flagged as OOD: {test_ood_seen}/{len(test_errors_seen)} (FPR: {stats["false_positive_rate"]*100:.2f}%)')
    
    print(f'\nTest Set - All Combined:')
    print(f'  Mean Error: {stats["test_all_mean"]:.6f} ± {stats["test_all_std"]:.6f}')
    print(f'  Range: [{stats["test_all_min"]:.6f}, {stats["test_all_max"]:.6f}]')
    print(f'  Flagged as OOD: {test_ood_all}/{len(test_errors_all)} ({(test_ood_all/len(test_errors_all))*100:.2f}%)')
    
    print(f'\nThreshold: {threshold:.6f}')
    print(f'FPR (False Positives on Seen): {stats["false_positive_rate"]*100:.2f}%')
    print(f'TPR (True Positives on Unseen): {stats["true_positive_rate"]*100:.2f}%')
    
    return stats


def main():
    """Main evaluation pipeline for pre-trained autoencoders."""
    
    print('EVALUATING PRE-TRAINED AUTOENCODERS FOR OOD DETECTION')
    
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_out_dir = os.path.join('.', f'evaluation_{timestamp}')
    os.makedirs(main_out_dir, exist_ok=True)
    

    
    # Model definitions from the.keras files generated at train time
    models_to_evaluate = {
        'deep_conv_model.keras': 'Deep Convolutional Autoencoder',
        'simple_conv_model.keras': 'Simple Convolutional Autoencoder',
        'simple_sparse_model.keras': 'Simple Sparse Autoencoder (L1)',
    }
    
    # Common configuration 
    input_size = (64, 64)
    batch_size = 32
    val_split = 0.2
    
    # Load datasets unseen test data
    print('\nLoading datasets...')
    extract_dir = os.path.join(main_out_dir, 'extracted_data')
    train_ds, val_ds, test_ds_unseen, n_train, n_val, n_test, num_classes = build_datasets_from_zip_eval(
        root,
        img_size=input_size,
        batch_size=batch_size,
        val_split=val_split,
        extract_to=extract_dir,
        path_unseen=None,
        test_path="data_mix_unseen"
    )

    # Load datasets seen test data
    train_ds, val_ds, test_ds_seen, n_train, n_val, n_test, num_classes = build_datasets_from_zip_eval(
        root,
        img_size=input_size,
        batch_size=batch_size,
        val_split=val_split,
        extract_to=extract_dir,
        path_unseen=None,
        test_path="data_mix_seen"
    )
    
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds_unseen = test_ds_unseen.prefetch(tf.data.AUTOTUNE)
    test_ds_seen = test_ds_seen.prefetch(tf.data.AUTOTUNE)
    
    print(f'Training: {n_train}, Validation: {n_val}, Test: {n_test}')
    
    # Store all results
    all_results = []
    
    # Evaluate each pre-trained model
    for model_filename, model_name in models_to_evaluate.items():
        print(f'# {model_name.upper()}')
        
        # Check if model file exists
        if not os.path.exists(model_filename):
            continue
        
        try:
            # Load pre-trained model
            print(f'\nLoading model from {model_filename}...')
            model = models.load_model(model_filename)
            
            # Evaluate OOD detection
            stats = evaluate_ood_detection(model, model_name, val_ds, test_ds_unseen, test_ds_seen, main_out_dir)
            all_results.append(stats)
            
            # Save individual stats
            model_base = os.path.splitext(model_filename)[0]
            stats_path = os.path.join(main_out_dir, f'{model_base}_evaluation_stats.txt')
            with open(stats_path, 'w') as f:
                f.write(f'{model_name} OOD DETECTION EVALUATION RESULTS\n')
                for key, value in stats.items():
                    f.write(f'{key}: {value}\n')
                        
        except Exception as e:
            print(f'\nError evaluating {model_name}: {str(e)}')
            import traceback
            traceback.print_exc()
            continue
    
    # Save comparison results
    print('SUMMARY OF ALL EVALUATED MODELS')
    
    comparison_path = os.path.join(main_out_dir, 'evaluation_comparison_results.txt')
    with open(comparison_path, 'w') as f:
        f.write('COMPARISON OF PRE-TRAINED AUTOENCODERS FOR OOD DETECTION\n')
        
        # Header
        f.write(f'{"Model":<35} {"Accuracy":<12} {"TPR":<12} {"FPR":<12} {"Specificity":<12} {"Val Error":<15} {"Unseen Error":<15} {"Seen Error":<15}\n')
        print(f'\n{"Model":<35} {"Accuracy":<12} {"TPR":<12} {"FPR":<12} {"Specificity":<12} {"Val Error":<15} {"Unseen Error":<15} {"Seen Error":<15}')
        
        for stats in all_results:
            line = (f'{stats["model_name"]:<35} '
                   f'{stats["combined_accuracy"]*100:<12.2f} '
                   f'{stats["combined_tpr"]*100:<12.2f} '
                   f'{stats["combined_fpr"]*100:<12.2f} '
                   f'{stats["combined_specificity"]*100:<12.2f} '
                   f'{stats["val_mean"]:<15.6f} '
                   f'{stats["test_unseen_mean"]:<15.6f} '
                   f'{stats["test_seen_mean"]:<15.6f}')
            f.write(line + '\n')
            print(line)
        
        if all_results:
            f.write('\nBest Combined Accuracy: ')
            best_acc = max(all_results, key=lambda x: x['combined_accuracy'])
            f.write(f'{best_acc["model_name"]} ({best_acc["combined_accuracy"]*100:.2f}%)\n')
            
            f.write('Best TPR on Unseen Classes (OOD Detection): ')
            best_tpr = max(all_results, key=lambda x: x['combined_tpr'])
            f.write(f'{best_tpr["model_name"]} ({best_tpr["combined_tpr"]*100:.2f}%)\n')
            
            f.write('Best FPR on Seen Classes (Specificity): ')
            best_fpr = min(all_results, key=lambda x: x['combined_fpr'])
            f.write(f'{best_fpr["model_name"]} ({best_fpr["combined_fpr"]*100:.2f}%)\n')
            
            f.write('Best Separation (Unseen/Seen error ratio): ')
            best_sep = max(all_results, key=lambda x: x['test_unseen_mean'] / x['test_seen_mean'] if x['test_seen_mean'] > 0 else 0)
            separation_ratio = best_sep['test_unseen_mean'] / best_sep['test_seen_mean'] if best_sep['test_seen_mean'] > 0 else 0
            f.write(f'{best_sep["model_name"]} ({separation_ratio:.2f}x)\n')
            
            f.write('\nDetailed Per-Model Results:\n')
            for stats in all_results:
                f.write(f'{stats["model_name"]}\n')
                f.write(f'  Validation (Known Classes):\n')
                f.write(f'    Mean Error: {stats["val_mean"]:.6f} ± {stats["val_std"]:.6f}\n')
                f.write(f'    Range: [{stats["val_min"]:.6f}, {stats["val_max"]:.6f}]\n')
                f.write(f'  Test Unseen (OOD Classes):\n')
                f.write(f'    Mean Error: {stats["test_unseen_mean"]:.6f} ± {stats["test_unseen_std"]:.6f}\n')
                f.write(f'    Range: [{stats["test_unseen_min"]:.6f}, {stats["test_unseen_max"]:.6f}]\n')
                f.write(f'    Detected as OOD: {stats["test_unseen_ood_detected"]}/{stats["test_unseen_total"]} (TPR: {stats["true_positive_rate"]*100:.2f}%)\n')
                f.write(f'  Test Seen (In-Distribution Classes):\n')
                f.write(f'    Mean Error: {stats["test_seen_mean"]:.6f} ± {stats["test_seen_std"]:.6f}\n')
                f.write(f'    Range: [{stats["test_seen_min"]:.6f}, {stats["test_seen_max"]:.6f}]\n')
                f.write(f'    Incorrectly Flagged as OOD: {stats["test_seen_ood_detected"]}/{stats["test_seen_total"]} (FPR: {stats["false_positive_rate"]*100:.2f}%)\n')
                f.write(f'  Combined Test Set Performance:\n')
                f.write(f'    Accuracy: {stats["combined_accuracy"]*100:.2f}%\n')
                f.write(f'    TPR (Unseen Detection): {stats["combined_tpr"]*100:.2f}%\n')
                f.write(f'    FPR (Seen False Alarms): {stats["combined_fpr"]*100:.2f}%\n')
                f.write(f'    Specificity (Seen Correctly Identified): {stats["combined_specificity"]*100:.2f}%\n')
                f.write(f'    TP: {stats["combined_tp"]}, TN: {stats["combined_tn"]}, FP: {stats["combined_fp"]}, FN: {stats["combined_fn"]}\n')
                f.write(f'  Threshold: {stats["threshold"]:.6f}\n')
                f.write('\n')
    
    print(f'All results saved to: {main_out_dir}')
    print(f'Comparison table: {comparison_path}')
    print('Done!')


if __name__ == '__main__':
    main()
