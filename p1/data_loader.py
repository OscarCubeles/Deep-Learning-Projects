import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

class MAMeDataset(keras.utils.Sequence):
    def __init__(self, csv_file, img_dir, subset, batch_size=32, transform=None, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['Subset'] == subset].reset_index(drop=True)

        # Create the Medium-to-index dictionary
        all_data = pd.read_csv(csv_file)
        self.medium_to_idx = {medium: idx for idx, medium in enumerate(sorted(all_data['Medium'].unique()))}
        self.idx_to_medium = {idx: medium for medium, idx in self.medium_to_idx.items()}

        self.img_dir = img_dir
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Returns the number of batches per epoch"""
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate data
        images = []
        labels = []
        
        for idx in batch_indexes:
            row = self.data.iloc[idx]
            img_path = os.path.join(self.img_dir, row['Image file'])
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            # Apply transform if provided
            if self.transform:
                image = self.transform(image)
            else:
                image = image / 255.0  # Default normalization
            
            images.append(image)
            label = self.medium_to_idx[row['Medium']]
            labels.append(label)
        
        return np.array(images), np.array(labels)
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def get_num_samples(self):
        """Returns total number of samples"""
        return len(self.data)
