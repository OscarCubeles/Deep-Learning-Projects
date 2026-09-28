import tensorflow as tf
import keras
from keras import layers


def simpleconv_autoencoder():
    """
    Build a simple sequential autoencoder without sparsity constraint.
    
    Returns:
        Keras Sequential model
    """
    model = keras.Sequential([
        # Encoder
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'),  # 32x32
        layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),  # 16x16
        layers.Conv2D(128, 3, strides=2, padding='same', activation='relu'),  # 8x8
        layers.Conv2D(256, 3, strides=2, padding='same', activation='relu'),  # 4x4
        layers.Conv2D(64, 3, strides=1, padding='same', activation='relu'),  # bottleneck
        
        # Decoder
        layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),  # 8x8
        layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),   # 16x16
        layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),   # 32x32
        layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='sigmoid'),  # 64x64
    ], name='simple_autoencoder')
    
    return model