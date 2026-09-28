import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from scipy.stats import entropy

import matplotlib
matplotlib.use("Agg")
import os
import matplotlib.pyplot as plt


class VGG11Evaluator:
    """Evaluation module for VGG11 model on seen and unseen classes."""

    def __init__(self, model, num_classes):
        """
        Initialize evaluator.

        Args:
            model: Trained Keras VGG11 model
            num_classes: Number of known classes
        """
        self.model = model
        self.num_classes = num_classes

    def evaluate_seen_classes(self, validation_dataset, class_labels=None):
        """
        Evaluate prediction performance on seen classes (validation set).

        Computes:
        - Per-class accuracy, precision, and recall
        - Per-class confidence via softmax probability scores
        - Entropy measurements for each class
        - Identifies difficult-to-recognize classes

        Args:
            validation_dataset: TensorFlow dataset with (images, labels)
            class_labels: Optional list of class names for better readability

        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        all_predictions = []
        all_labels = []
        all_confidences = []
        all_entropies = []

        # Collect predictions and probabilities
        for x_batch, y_batch in validation_dataset:
            logits = self.model(x_batch, training=False)
            probabilities = tf.nn.softmax(logits, axis=-1).numpy()
            predicted_classes = np.argmax(probabilities, axis=1)

            all_predictions.append(predicted_classes)
            all_labels.append(y_batch.numpy())
            all_confidences.append(np.max(probabilities, axis=1))

            # Compute entropy for each sample
            sample_entropies = np.array([entropy(probs) for probs in probabilities])
            all_entropies.append(sample_entropies)

        # Concatenate all batches
        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)
        confidences = np.concatenate(all_confidences)
        entropies = np.concatenate(all_entropies)

        # Compute per-class metrics
        per_class_metrics = {}
        difficult_classes = []

        for class_id in range(self.num_classes):
            mask = labels == class_id
            if np.sum(mask) == 0:
                continue

            class_predictions = predictions[mask]
            class_labels = labels[mask]
            class_confidences = confidences[mask]
            class_entropies = entropies[mask]

            # Per-class accuracy
            class_accuracy = accuracy_score(class_labels, class_predictions)

            # Per-class precision and recall
            class_precision = precision_score(
                class_labels, class_predictions, average='weighted', zero_division=0
            )
            class_recall = recall_score(
                class_labels, class_predictions, average='weighted', zero_division=0
            )

            # Confidence statistics
            mean_confidence = np.mean(class_confidences)
            std_confidence = np.std(class_confidences)
            min_confidence = np.min(class_confidences)

            # Entropy statistics
            mean_entropy = np.mean(class_entropies)
            std_entropy = np.std(class_entropies)
            # Safe version
            if class_labels is None or class_id >= len(class_labels):
                class_name = f"Class_{class_id}"
            else:
                class_name = class_labels[class_id]
            #class_name = class_labels[class_id] if class_labels else f"Class_{class_id}"

            per_class_metrics[class_id] = {
                'accuracy': class_accuracy,
                'precision': class_precision,
                'recall': class_recall,
                'num_samples': np.sum(mask),
                'mean_confidence': mean_confidence,
                'std_confidence': std_confidence,
                'min_confidence': min_confidence,
                'mean_entropy': mean_entropy,
                'std_entropy': std_entropy,
                'class_name': class_name,
            }

            # Identify difficult classes (low accuracy or high entropy)
            if class_accuracy < 0.7:
                difficult_classes.append((class_id, class_accuracy, mean_entropy))

        # Sort difficult classes by accuracy (ascending)
        difficult_classes.sort(key=lambda x: x[1])

        # Overall metrics
        overall_accuracy = accuracy_score(labels, predictions)
        overall_precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        overall_recall = recall_score(labels, predictions, average='weighted', zero_division=0)

        # Confusion matrix
        conf_matrix = confusion_matrix(labels, predictions, labels=range(self.num_classes))

        return {
            'overall_accuracy': overall_accuracy,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'per_class_metrics': per_class_metrics,
            'difficult_classes': difficult_classes,
            'confusion_matrix': conf_matrix,
            'predictions': predictions,
            'labels': labels,
            'confidences': confidences,
            'entropies': entropies,
        }

    def evaluate_unseen_classes(self, test_dataset):
        """
        Evaluate prediction performance on unseen classes.

        Analyzes behavior when predicting instances from unknown classes:
        - Predicted class distribution
        - Confidence level statistics
        - Entropy measurements
        - Which known classes are most commonly predicted

        Args:
            test_dataset: TensorFlow dataset with (images, labels)
            unseen_class_mask: Boolean array indicating which samples are from unseen classes

        Returns:
            Dictionary containing unseen class evaluation metrics
        """
        all_predictions = []
        all_confidences = []
        all_entropies = []
        all_full_probabilities = []

        # Collect predictions
        for x_batch, y_batch in test_dataset:
            logits = self.model(x_batch, training=False)
            probabilities = tf.nn.softmax(logits, axis=-1).numpy()
            predicted_classes = np.argmax(probabilities, axis=1)

            all_predictions.append(predicted_classes)
            all_confidences.append(np.max(probabilities, axis=1))
            all_full_probabilities.append(probabilities)

            # Entropy for each sample
            sample_entropies = np.array([entropy(probs) for probs in probabilities])
            all_entropies.append(sample_entropies)

        unseen_predictions = np.concatenate(all_predictions)
        unseen_confidences = np.concatenate(all_confidences)
        unseen_entropies = np.concatenate(all_entropies)
        unseen_probabilities = np.concatenate(all_full_probabilities)

        # Predicted class distribution for unseen samples
        unique_predicted, counts = np.unique(unseen_predictions, return_counts=True)
        predicted_class_distribution = {
            int(class_id): int(count) for class_id, count in zip(unique_predicted, counts)
        }

        # Top predicted classes
        top_predicted = sorted(predicted_class_distribution.items(), key=lambda x: x[1], reverse=True)

        # Confidence statistics for unseen class predictions
        mean_confidence = np.mean(unseen_confidences)
        std_confidence = np.std(unseen_confidences)
        min_confidence = np.min(unseen_confidences)
        max_confidence = np.max(unseen_confidences)

        # Entropy statistics for unseen class predictions
        mean_entropy = np.mean(unseen_entropies)
        std_entropy = np.std(unseen_entropies)
        min_entropy = np.min(unseen_entropies)
        max_entropy = np.max(unseen_entropies)

        # Analyze confidence vs entropy correlation
        if np.std(unseen_entropies) > 0:
            correlation = np.corrcoef(unseen_confidences, unseen_entropies)[0, 1]
        else:
            correlation = 0.0
        # Percentage of unseen samples with high confidence (> 0.9)
        high_confidence_count = np.sum(unseen_confidences > 0.9)
        high_confidence_percentage = (high_confidence_count / len(unseen_confidences)) * 100

        return {
            'num_unseen_samples': len(unseen_predictions),
            'predicted_class_distribution': predicted_class_distribution,
            'top_predicted_classes': top_predicted,
            'confidence_stats': {
                'mean': mean_confidence,
                'std': std_confidence,
                'min': min_confidence,
                'max': max_confidence,
            },
            'entropy_stats': {
                'mean': mean_entropy,
                'std': std_entropy,
                'min': min_entropy,
                'max': max_entropy,
            },
            'confidence_entropy_correlation': correlation,
            'high_confidence_percentage': high_confidence_percentage,
            'predictions': unseen_predictions,
            'confidences': unseen_confidences,
            'entropies': unseen_entropies,
            'full_probabilities': unseen_probabilities,
        }

    def print_evaluation_report(self, seen_eval, unseen_eval=None):
        """
        Print a comprehensive evaluation report.

        Args:
            seen_eval: Dictionary from evaluate_seen_classes()
            unseen_eval: Dictionary from evaluate_unseen_classes()
        """
        if unseen_eval is None:
            print("EVALUATION REPORT: SEEN CLASSES (VALIDATION SET)")
            print("="*80)

            print(f"\nOVERALL PERFORMANCE:")
            print(f"  Accuracy:  {seen_eval['overall_accuracy']:.4f}")
            print(f"  Precision: {seen_eval['overall_precision']:.4f}")
            print(f"  Recall:    {seen_eval['overall_recall']:.4f}")

            print(f"\nPER-CLASS METRICS:")
            print(f"{'Class':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'Confidence':<15} {'Entropy':<12} {'Samples':<10}")
            print("-" * 95)

            for class_id, metrics in sorted(seen_eval['per_class_metrics'].items()):
                print(f"{class_id:<10} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                    f"{metrics['recall']:<12.4f} {metrics['mean_confidence']:<15.4f} "
                    f"{metrics['mean_entropy']:<12.4f} {metrics['num_samples']:<10}")

            if seen_eval['difficult_classes']:
                print(f"\nDIFFICULT CLASSES (Accuracy < 0.7):")
                for class_id, accuracy, entropy_val in seen_eval['difficult_classes']:
                    print(f"  Class {class_id}: Accuracy={accuracy:.4f}, Mean Entropy={entropy_val:.4f}")
            else:
                print(f"\nNo difficult classes identified (all classes with accuracy >= 0.7)")
        else:
            print("\n" + "="*80)
            print("EVALUATION REPORT: UNSEEN CLASSES")
            print("="*80)

            print(f"\nUNSEEN CLASS DETECTION STATISTICS:")
            print(f"  Total unseen samples: {unseen_eval['num_unseen_samples']}")
            print(f"  Confidence (mean ± std): {unseen_eval['confidence_stats']['mean']:.4f} ± {unseen_eval['confidence_stats']['std']:.4f}")
            print(f"  Confidence range: [{unseen_eval['confidence_stats']['min']:.4f}, {unseen_eval['confidence_stats']['max']:.4f}]")
            print(f"  Entropy (mean ± std): {unseen_eval['entropy_stats']['mean']:.4f} ± {unseen_eval['entropy_stats']['std']:.4f}")
            print(f"  Entropy range: [{unseen_eval['entropy_stats']['min']:.4f}, {unseen_eval['entropy_stats']['max']:.4f}]")
            print(f"  Confidence-Entropy correlation: {unseen_eval['confidence_entropy_correlation']:.4f}")
            print(f"  High confidence (>0.9) samples: {unseen_eval['high_confidence_percentage']:.2f}%")

            print(f"\nTOP PREDICTED KNOWN CLASSES FOR UNSEEN SAMPLES:")
            for i, (class_id, count) in enumerate(unseen_eval['top_predicted_classes'][:5], 1):
                percentage = (count / unseen_eval['num_unseen_samples']) * 100
                print(f"  {i}. Class {class_id}: {count} samples ({percentage:.2f}%)")

        print("\n" + "="*80)

    def visualize_unseen_with_training_examples(
        self,
        model,
        unseen_dataset,
        class_to_example,
        num_examples=6,
        out_dir="results"
    ):
        """
        Save one image per unseen example:
        results/example_0000.png, example_0001.png, ...
        """
        os.makedirs(out_dir, exist_ok=True)

        example_id = 0

        for x_batch, _ in unseen_dataset:
            logits = model(x_batch, training=False)
            probs = tf.nn.softmax(logits, axis=-1).numpy()
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)
            ents = np.array([entropy(p) for p in probs])

            for i in range(len(x_batch)):
                img = x_batch[i].numpy()
                pred = preds[i]
                conf = confs[i]
                ent = ents[i]

                # Create ONE figure per example
                plt.figure(figsize=(6, 3))

                # Unseen image
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.axis("off")
                plt.title(f"Unseen\nConf:{conf:.2f}\nEnt:{ent:.2f}", fontsize=9)

                # Training example of predicted class
                plt.subplot(1, 2, 2)
                plt.imshow(class_to_example[pred])
                plt.axis("off")
                plt.title(f"Class {pred}", fontsize=9)

                plt.tight_layout()

                out_path = os.path.join(out_dir, f"example_{example_id:04d}.png")
                plt.savefig(out_path, dpi=200, bbox_inches="tight")
                plt.close()

                example_id += 1
                if example_id >= num_examples:
                    print(f"[INFO] Saved {example_id} examples to {out_dir}")
                    return