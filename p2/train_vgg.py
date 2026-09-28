import os
import tensorflow as tf
import keras
from keras import layers
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from german_dataset import build_datasets_from_zip
from models.evaluation import Evaluator
from configurations.configurations_open_set import CONFIG
from keras.applications import VGG16

#root = r'C:\Users\Claudia Boixader\Desktop\master\Semestre3\DL\gtsrb_open_set\data\archive.zip'
#out_dir = r'C:\Users\Claudia Boixader\Desktop\master\Semestre3\DL\gtsrb_open_set\results'

# no has de canviar res d'aquests paths, només el num d'experiment a les execucions
root = r'/home/nct/nct01150/dl_openset/data/archive.zip'
out_dir = r'/home/nct/nct01150/dl_openset/exp_vgg_3/results'
out_data_dir = r'/home/nct/nct01150/dl_openset/data'
unseen_test = r'/home/nct/nct01150/dl_openset/data'

def train_vgg(root, path):
    cfg = CONFIG['vgg']
    input_size = cfg['input_size']

    os.makedirs(path, exist_ok=True)

    extract_dir = os.path.join(path, 'extracted_data')
    train_ds, val_ds, test_ds, n_train, n_val, n_test, num_classes = build_datasets_from_zip(
        root,
        img_size=input_size,
        batch_size=cfg['batch_size'],
        val_split=0.2,
        extract_to=extract_dir,
        path_unseen=unseen_test,
    )

    print('train samples:', n_train)
    print('val samples:', n_val)
    print('num classes:', num_classes)

    base_model = VGG16(
        include_top=False,
        weights=None,
        input_shape=(input_size[0], input_size[1], 3)
    )
    base_model.trainable = True
    # Custom classification head
    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes)(x)

    model = keras.Model(inputs=base_model.input, outputs=outputs)
    optimizer = keras.optimizers.Adam(learning_rate=cfg['lr'])

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg['epochs'],
    )

    model.save(os.path.join(path, "vgg16_pretrained.keras"))

    return history, model, train_ds, val_ds, test_ds, n_test, num_classes

def evaluate_vgg_seen_classes(model, val_ds, num_classes, path):
    """
    Evaluate VGG11 performance on seen classes (validation set).
    This function:
    1) Assesses per-class prediction performance (accuracy, recall)
    2) Analyzes per-class prediction confidence (softmax probabilities and entropy)
    3) Identifies classes that are more difficult to recognize
    """
    evaluator = Evaluator(model, num_classes)
    seen_eval = evaluator.evaluate_seen_classes(val_ds)

    # Print comprehensive report
    evaluator.print_evaluation_report(seen_eval)

    # Save text report
    with open(os.path.join(path, 'vgg11_evaluation_report.txt'), 'w') as f:
        f.write("VGG11 EVALUATION REPORT - SEEN CLASSES\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall Accuracy:  {seen_eval['overall_accuracy']:.4f}\n")
        f.write(f"Overall Recall:    {seen_eval['overall_recall']:.4f}\n\n")

        f.write("PER-CLASS METRICS:\n")
        f.write(f"{'Class':<10} {'Accuracy':<12} {'Recall':<12} {'Confidence':<15} {'Entropy':<12} {'Samples':<10}\n")
        f.write("-" * 95 + "\n")

        for class_id, metrics in sorted(seen_eval['per_class_metrics'].items()):
            f.write(f"{class_id:<10} {metrics['accuracy']:<12.4f} {metrics['recall']:<12.4f} "
                   f"{metrics['mean_confidence']:<15.4f} "
                   f"{metrics['mean_entropy']:<12.4f} {metrics['num_samples']:<10}\n")

        if seen_eval['difficult_classes']:
            f.write("\nDIFFICULT CLASSES (Accuracy < 0.7):\n")
            for class_id, accuracy, entropy_val in seen_eval['difficult_classes']:
                f.write(f"  Class {class_id}: Accuracy={accuracy:.4f}, Mean Entropy={entropy_val:.4f}\n")

    return seen_eval


def evaluate_vgg_unseen_classes(model, test_ds, num_classes, path):
    """
    Evaluate VGG11 performance on unseen classes.
    This function analyzes the behavior of the model when predicting instances
    from unseen classes, taking into account:
    - Predicted class distribution
    - Confidence levels (softmax probabilities)
    - Entropy measurements
    """
    evaluator = Evaluator(model, num_classes)

    unseen_eval = evaluator.evaluate_unseen_classes(test_ds)

    # Print comprehensive report
    evaluator.print_evaluation_report({
        'overall_accuracy': 0.0,
        'overall_recall': 0.0,
        'per_class_metrics': {},
        'difficult_classes': [],
        'confusion_matrix': None,
        'predictions': None,
        'labels': None,
        'confidences': None,
        'entropies': None,
    }, unseen_eval)

    # Save text report
    with open(os.path.join(path, 'vgg11_evaluation_unseen_report.txt'), 'w') as f:
        f.write("VGG11 EVALUATION REPORT - UNSEEN CLASSES\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total unseen samples: {unseen_eval['num_unseen_samples']}\n\n")

        f.write("CONFIDENCE STATISTICS:\n")
        f.write(f"  Mean: {unseen_eval['confidence_stats']['mean']:.4f}\n")
        f.write(f"  Std:  {unseen_eval['confidence_stats']['std']:.4f}\n")
        f.write(f"  Min:  {unseen_eval['confidence_stats']['min']:.4f}\n")
        f.write(f"  Max:  {unseen_eval['confidence_stats']['max']:.4f}\n\n")

        f.write("ENTROPY STATISTICS:\n")
        f.write(f"  Mean: {unseen_eval['entropy_stats']['mean']:.4f}\n")
        f.write(f"  Std:  {unseen_eval['entropy_stats']['std']:.4f}\n")
        f.write(f"  Min:  {unseen_eval['entropy_stats']['min']:.4f}\n")
        f.write(f"  Max:  {unseen_eval['entropy_stats']['max']:.4f}\n\n")

        f.write(f"Confidence-Entropy correlation: {unseen_eval['confidence_entropy_correlation']:.4f}\n")
        f.write(f"High confidence (>0.9) samples: {unseen_eval['high_confidence_percentage']:.2f}%\n\n")

        f.write("TOP PREDICTED KNOWN CLASSES FOR UNSEEN SAMPLES:\n")
        for i, (class_id, count) in enumerate(unseen_eval['top_predicted_classes'][:10], 1):
            percentage = (count / unseen_eval['num_unseen_samples']) * 100
            f.write(f"  {i}. Class {class_id}: {count} samples ({percentage:.2f}%)\n")
    return unseen_eval

def plot_training_metrics(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    # ---- Loss ----
    axes[0].plot(history.history['loss'], label='train_loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='val_loss')
    axes[0].set_title('Loss over epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # ---- Accuracy ----
    axes[1].plot(history.history['accuracy'], label='train_acc')
    if 'val_accuracy' in history.history:
        axes[1].plot(history.history['val_accuracy'], label='val_acc')
    axes[1].set_title('Accuracy over epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    out_path = os.path.join(out_dir, "training_metrics.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved training metrics to {out_path}")

def plot_accuracy_vs_confidence_entropy(per_class_metrics, out_dir, seen=True):
    os.makedirs(out_dir, exist_ok=True)

    classes = sorted(per_class_metrics.keys())
    accuracies = [per_class_metrics[c]['accuracy'] for c in classes]
    confidences = [per_class_metrics[c]['mean_confidence'] for c in classes]
    entropies = [per_class_metrics[c]['mean_entropy'] for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Accuracy vs Confidence ----
    axes[0].scatter(confidences, accuracies)
    for i, cls in enumerate(classes):
        axes[0].text(confidences[i], accuracies[i], str(cls), fontsize=8)
    axes[0].set_xlabel('Mean Confidence')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Per-class Accuracy vs Confidence')
    axes[0].grid(True)

    # ---- Accuracy vs Entropy ----
    axes[1].scatter(entropies, accuracies)
    for i, cls in enumerate(classes):
        axes[1].text(entropies[i], accuracies[i], str(cls), fontsize=8)
    axes[1].set_xlabel('Mean Entropy')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Per-class Accuracy vs Entropy')
    axes[1].grid(True)

    plt.tight_layout()
    if seen:
        out_path = os.path.join(out_dir, "accuracy_vs_confidence_entropy_seen.png")
    else:
        out_path = os.path.join(out_dir, "accuracy_vs_confidence_entropy_unseen.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved accuracy vs confidence/entropy plot to {out_path}")

def visualize_baseline_unseen(vgg, train_ds, test_ds, num_classes, num_examples):
    # Initialize dictionary
    class_to_example = {}

    # Iterate over the training dataset
    for x_batch, y_batch in train_ds:
        # Convert labels to numpy if necessary
        y_batch_np = y_batch.numpy() if hasattr(y_batch, "numpy") else y_batch
        
        for i in range(len(x_batch)):
            class_id = int(y_batch_np[i])
            # If we don't have an example yet, save this image
            if class_id not in class_to_example:
                class_to_example[class_id] = x_batch[i].numpy() if hasattr(x_batch[i], "numpy") else x_batch[i]
        
        # Stop if we have all known classes
        if len(class_to_example) >= num_classes:
            break
            
    print(f"Collected example images for {len(class_to_example)} classes.")

    evaluator = Evaluator(vgg, num_classes)

    evaluator.visualize_unseen_with_training_examples(
        model=vgg,
        unseen_dataset=test_ds,
        class_to_example=class_to_example,
        num_examples=num_examples,
        out_dir="results"
    )

if __name__ == '__main__':
    # PART 1
    os.makedirs(out_dir, exist_ok=True)
    cfg = CONFIG['vgg']

    print('Training VGG16..')
    history, vgg, train_ds, val_ds, test_ds, n_test, num_classes = train_vgg(root, out_data_dir)
    plot_training_metrics(history, out_dir)

    print('\nEvaluating VGG16 on seen classes (validation set)...')
    seen_eval = evaluate_vgg_seen_classes(vgg, val_ds, num_classes, out_dir)
    plot_accuracy_vs_confidence_entropy(
        seen_eval['per_class_metrics'],
        out_dir="results",
        seen=True
    )

    print(f'\nEvaluating VGG16 on unseen classes ({n_test} samples)...')
    unseen_eval = evaluate_vgg_unseen_classes(vgg, test_ds, num_classes, out_dir)
    plot_accuracy_vs_confidence_entropy(
        seen_eval['per_class_metrics'],
        out_dir="results",
        seen=False
    )

    print('\nVisualizing unseen predictions...')
    visualize_baseline_unseen(vgg, train_ds, test_ds, num_classes, cfg['num_examples'])