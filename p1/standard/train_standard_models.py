import tensorflow as tf
from tensorflow import keras
from data_loader import MAMeDataset
from standard_models import AlexNet, StandardArchitectureModel, StandardArchitectureModelNoAttention, ResNet18, ResNet50TransferLearning
import numpy as np
import csv
import os

# Path configuration
home_path_old = r'C:\Users\CuOs364\OneDrive - HP Inc\Documents\Projects\DL_CNN_CLASSIFICATION'
home_path = '/home/nct/nct01150/oscar/standard'
csv_file_path = '/home/nct/nct01150/oscar/data/MAMe_metadata/MAMe_dataset.csv'
images_dir_path = '/home/nct/nct01150/oscar/data/MAMe_data_256'

def train(model, train_loader, val_loader, epochs, csv_filename, lr, step_size, gamma, betas):
    print("\n=== Setting up training ===")
    print(f"Learning rate: {lr}, Step size: {step_size}, Gamma: {gamma}, Betas: {betas}")
    
    # Define loss function and optimizer
    criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Adam optimizer with custom beta values
    optimizer = keras.optimizers.Adam(
        learning_rate=lr,
        beta_1=betas[0],
        beta_2=betas[1]
    )
    
    # Learning rate scheduler
    def lr_schedule(epoch):
        """Step decay learning rate schedule"""
        return lr * (gamma ** (epoch // step_size))
    
    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    
    # Compile model
    print("Compiling model...")
    model.compile(
        optimizer=optimizer,
        loss=criterion,
        metrics=['accuracy']
    )
    print("Model compiled successfully")
    
    # Create CSV logger callback
    class CSVMetricsLogger(keras.callbacks.Callback):
        def __init__(self, filename):
            super().__init__()
            self.filename = filename
            # Create CSV file with headers
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy', 'lr'])
        
        def on_epoch_end(self, epoch, logs=None):
            with open(self.filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    round(logs.get('loss', 0), 4),
                    round(logs.get('val_loss', 0), 4),
                    round(logs.get('accuracy', 0), 4),
                    round(logs.get('val_accuracy', 0), 4),
                    round(float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)), 6)
                ])
    
    csv_logger = CSVMetricsLogger(csv_filename)
    print(f"Metrics will be logged to: {csv_filename}")
    
    # Train the model
    print(f"\n=== Starting training for {epochs} epochs ===")
    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=epochs,
        callbacks=[lr_scheduler, csv_logger],
        verbose=1
    )
    print("\n=== Training completed ===")
    
    return history


def evaluate(model, test_loader):
    print("\n=== Evaluating on test set ===")
    # Evaluate the model on the test set
    results = model.evaluate(test_loader, verbose=1)
    test_loss = results[0]
    test_acc = results[1]
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    return test_acc


def get_model(model_name, num_classes):
    # Select and return model architecture based on name
    if model_name == 'AlexNet':
        input_size = 256
        model = AlexNet(num_classes=num_classes)
    elif model_name == 'StandardArchitectureModel':
        input_size = 256
        model = StandardArchitectureModelNoAttention(num_classes=num_classes)
    elif model_name == 'StandardArchitectureModelNoAttention':
        input_size = 256
        model = StandardArchitectureModelNoAttention(num_classes=num_classes)
    elif model_name == 'ResNet18':
        input_size = 224
        model = ResNet18(num_classes=num_classes)
    
    else:
        raise ValueError(f"Model '{model_name}' not found.")
    return model, input_size

def calculate_mean_std(dataset):
    print("Calculating dataset mean and std...")
    # Calculate per-channel mean and standard deviation for normalization
    mean = np.array([0.0, 0.0, 0.0])
    std = np.array([0.0, 0.0, 0.0])
    total_images = 0
    
    # Temporarily set batch_size to dataset length to process all at once
    original_batch_size = dataset.batch_size
    dataset.batch_size = len(dataset.data)
    
    for images, _ in dataset:
        batch_samples = images.shape[0]
        # Calculate mean and std across spatial dimensions
        for i in range(3):  # RGB channels
            channel_data = images[:, :, :, i]
            mean[i] += np.mean(channel_data) * batch_samples
            std[i] += np.std(channel_data) * batch_samples
        total_images += batch_samples
    
    mean /= total_images
    std /= total_images
    
    # Restore original batch size
    dataset.batch_size = original_batch_size
    
    return mean, std


def create_transform(mean, std, input_size):
    """Create a preprocessing function for images"""
    def transform(image):
        # Resize
        image = tf.image.resize(image, [input_size, input_size])
        # Normalize
        image = (image - mean) / std
        return image
    return transform


def main(model_name, batch_size, epochs, csv_filename, lr, step_size, gamma, betas):
    print("\n" + "="*60)
    print(f"Starting experiment: {os.path.basename(csv_filename)}")
    print("="*60)
    
    # Use the new path variables
    csv_file = csv_file_path
    img_dir = images_dir_path

    print("\n=== Loading dataset to determine classes ===")
    # Load a temporary dataset to get the number of classes
    temp_dataset = MAMeDataset(csv_file, img_dir, subset='train', batch_size=1, transform=None, shuffle=False)
    num_classes = len(temp_dataset.medium_to_idx)
    print(f"Number of classes: {num_classes}")

    # Get the model architecture and input size
    print(f"\n=== Creating {model_name} model ===")
    model, input_size = get_model(model_name, num_classes=num_classes)
    print(f"Model created with input size: {input_size}x{input_size}")

    # Compute dataset-wide mean and std for normalization
    print("\n=== Computing normalization statistics ===")
    mean, std = calculate_mean_std(temp_dataset)
    print(f'Mean (RGB): [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]')
    print(f'Std  (RGB): [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]')

    # Define preprocessing transform
    transform = create_transform(mean, std, input_size)

    # Load full datasets with transformations
    print("\n=== Loading full datasets ===")
    print(f"Batch size: {batch_size}")
    train_dataset = MAMeDataset(csv_file, img_dir, subset='train', batch_size=batch_size, transform=transform, shuffle=True)
    val_dataset = MAMeDataset(csv_file, img_dir, subset='val', batch_size=batch_size, transform=transform, shuffle=False)
    test_dataset = MAMeDataset(csv_file, img_dir, subset='test', batch_size=batch_size, transform=transform, shuffle=False)

    print(f'Train: {train_dataset.get_num_samples()} images ({len(train_dataset)} batches)')
    print(f'Val:   {val_dataset.get_num_samples()} images ({len(val_dataset)} batches)')
    print(f'Test:  {test_dataset.get_num_samples()} images ({len(test_dataset)} batches)')

    # Build the model with proper input shape
    print("\n=== Building model ===")
    model.build(input_shape=(None, input_size, input_size, 3))
    print(f"Model built successfully")
    
    # Print model summary
    print("\n=== Model Architecture ===")
    model.summary()

    # Train the model
    train(model, train_dataset, val_dataset, epochs=epochs, csv_filename=csv_filename, lr=lr, step_size=step_size, gamma=gamma, betas=betas)

    # Evaluate the model on the test set
    test_acc = evaluate(model, test_dataset)

    # Save final test accuracy
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(['Test Accuracy', round(test_acc, 4)])
    
    # Save the trained model in the same folder as the CSV file
    # Extract the folder path from csv_filename (e.g., models/AlexNet/)
    model_folder = os.path.dirname(csv_filename)
    
    # Extract configuration name from csv filename (e.g., "ok.csv" -> "ok.keras")
    model_filename = os.path.basename(csv_filename).replace('.csv', '.keras')
    model_path = os.path.join(model_folder, model_filename)
    
    print(f"\n=== Saving model ===")
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    print("\n" + "="*60)
    print(f"Experiment completed: {os.path.basename(csv_filename)}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Model saved: {model_path}")
    print("="*60 + "\n")
