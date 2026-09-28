from keras import layers
from tensorflow import keras


def deep_conv_autoencoder():
    """
    Build an elaborate autoencoder with BatchNorm, Dropout, and deeper architecture.
    
    Returns:
        Keras Sequential model
    """
    model = keras.Sequential([
        # Encoder
        layers.Input(shape=(64, 64, 3)),
        
        # Block 1
        layers.Conv2D(32, 3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, 3, strides=2, padding='same'),  # 32x32
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        # Block 2
        layers.Conv2D(64, 3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, 3, strides=2, padding='same'),  # 16x16
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        # Block 3
        layers.Conv2D(128, 3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, 3, strides=2, padding='same'),  # 8x8
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Block 4
        layers.Conv2D(256, 3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, 3, strides=2, padding='same'),  # 4x4
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Bottleneck
        layers.Conv2D(16, 1, strides=1, padding='same'),  # Very tight bottleneck
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        # Decoder
        # Block 5
        layers.Conv2D(256, 3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2DTranspose(128, 3, strides=2, padding='same'),  # 8x8
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        # Block 6
        layers.Conv2D(128, 3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2DTranspose(64, 3, strides=2, padding='same'),  # 16x16
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        # Block 7
        layers.Conv2D(64, 3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2DTranspose(32, 3, strides=2, padding='same'),  # 32x32
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        # Block 8
        layers.Conv2D(32, 3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2DTranspose(16, 3, strides=2, padding='same'),  # 64x64
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        # Output
        layers.Conv2D(3, 3, strides=1, padding='same', activation='sigmoid'),
    ], name='elaborate_autoencoder')
    
    return model