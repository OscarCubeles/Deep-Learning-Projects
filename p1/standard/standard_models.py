import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AlexNet(keras.Sequential):
    """    
    AlexNet adapted for 256x256 images and 29 classes.
    
    Architecture: Conv → Pool → Conv → Pool → Conv → Conv → Conv → Pool → Dense → Dense → Dense
    """
    def __init__(self, num_classes=29, input_channels=3, **kwargs):
        layers_list = []
        
        # First Convolutional Block: Conv + BatchNorm + ReLU + MaxPool
        layers_list.append(layers.Conv2D(96, kernel_size=11, strides=4, padding='valid'))
        layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Activation('relu'))
        layers_list.append(layers.MaxPooling2D(pool_size=3, strides=2))
        
        # Second Convolutional Block: Conv + BatchNorm + ReLU + MaxPool
        layers_list.append(layers.Conv2D(256, kernel_size=5, strides=1, padding='same'))
        layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Activation('relu'))
        layers_list.append(layers.MaxPooling2D(pool_size=3, strides=2))
        
        # Third Convolutional Block: Conv + BatchNorm + ReLU
        layers_list.append(layers.Conv2D(384, kernel_size=3, strides=1, padding='same'))
        layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Activation('relu'))
        
        # Fourth Convolutional Block: Conv + BatchNorm + ReLU
        layers_list.append(layers.Conv2D(384, kernel_size=3, strides=1, padding='same'))
        layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Activation('relu'))
        
        # Fifth Convolutional Block: Conv + BatchNorm + ReLU + MaxPool
        layers_list.append(layers.Conv2D(256, kernel_size=3, strides=1, padding='same'))
        layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Activation('relu'))
        layers_list.append(layers.MaxPooling2D(pool_size=3, strides=2))
        
        # Flatten
        layers_list.append(layers.Flatten())
        
        # First Fully Connected Layer: Dense + ReLU + Dropout
        layers_list.append(layers.Dense(4096))
        layers_list.append(layers.Activation('relu'))
        layers_list.append(layers.Dropout(0.5))
        
        # Second Fully Connected Layer: Dense + ReLU + Dropout
        layers_list.append(layers.Dense(4096))
        layers_list.append(layers.Activation('relu'))
        layers_list.append(layers.Dropout(0.5))
        
        # Output Layer: Dense (no activation, logits for training)
        layers_list.append(layers.Dense(num_classes))
        
        # Initialize Sequential with all layers
        super(AlexNet, self).__init__(layers_list, **kwargs)


class ResidualBlock(layers.Layer):
    """Residual block with bottleneck design"""
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # Main path - Bottleneck design (1x1 -> 3x3 -> 1x1)
        self.conv1 = layers.Conv2D(mid_channels, kernel_size=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(mid_channels, kernel_size=3, strides=stride,
                                   padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        self.conv3 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()
        
        # Shortcut path
        self.use_shortcut = (stride != 1 or in_channels != out_channels)
        if self.use_shortcut:
            self.shortcut_conv = layers.Conv2D(out_channels, kernel_size=1, 
                                               strides=stride, use_bias=False)
            self.shortcut_bn = layers.BatchNormalization()
    
    def call(self, x, training=False):
        residual = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = tf.nn.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out, training=training)
        
        # Shortcut path
        if self.use_shortcut:
            residual = self.shortcut_conv(residual)
            residual = self.shortcut_bn(residual, training=training)
        
        # Add residual connection
        out = layers.Add()([out, residual])
        out = tf.nn.relu(out)
        
        return out


class ChannelAttention(layers.Layer):
    """
    Channel attention mechanism as a reusable layer.
    
    Applies channel-wise attention weights to focus on important features.
    """
    def __init__(self, channels, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.channels = channels
        
        # Attention layers
        self.conv1 = layers.Conv2D(128, kernel_size=1)
        self.bn = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(channels, kernel_size=1)
    
    def call(self, x, training=False):
        # Generate attention weights
        attention = self.conv1(x)
        attention = self.bn(attention, training=training)
        attention = tf.nn.relu(attention)
        attention = self.conv2(attention)
        attention = tf.nn.sigmoid(attention)
        
        # Apply attention (element-wise multiplication)
        return layers.Multiply()([x, attention])


class StandardArchitectureModel(keras.Sequential):
    """
    Modified architecture 
    
    Architecture:
    - Initial Conv (7x7) + MaxPool
    - Layer1: 2 Residual Blocks (64 -> 256 channels)
    - Layer2: 3 Residual Blocks (256 -> 512 channels, stride=2)
    - Layer3: 3 Residual Blocks (512 -> 512 channels, stride=2)
    - Attention mechanism
    - Global Average Pooling
    - Classifier: Dense(512->256) -> Dense(256->29)
    """
    def __init__(self, num_classes=29, input_channels=3, **kwargs):
        # Build the sequential model
        layers_list = []
        
        # Initial convolution layer
        layers_list.append(layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False))
        layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Activation('relu'))
        layers_list.append(layers.MaxPooling2D(pool_size=3, strides=2, padding='same'))
        
        # Residual blocks - Layer 1 (2 blocks: 64 -> 256 channels)
        layers_list.append(ResidualBlock(64, 64, 256, stride=1))
        layers_list.append(ResidualBlock(256, 64, 256, stride=1))
        
        # Residual blocks - Layer 2 (3 blocks: 256 -> 512 channels)
        layers_list.append(ResidualBlock(256, 128, 512, stride=2))
        layers_list.append(ResidualBlock(512, 128, 512, stride=1))
        layers_list.append(ResidualBlock(512, 128, 512, stride=1))
        
        # Residual blocks - Layer 3 (3 blocks: 512 -> 512 channels)
        layers_list.append(ResidualBlock(512, 256, 512, stride=2))
        layers_list.append(ResidualBlock(512, 256, 512, stride=1))
        layers_list.append(ResidualBlock(512, 256, 512, stride=1))
        
        # Attention mechanism (channel attention) - using Lambda layer for custom operations
        layers_list.append(ChannelAttention(512))
        
        # Global average pooling
        layers_list.append(layers.GlobalAveragePooling2D())
        
        # Classifier with dropout
        layers_list.append(layers.Dropout(0.3))
        layers_list.append(layers.Dense(256))
        layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Activation('relu'))
        layers_list.append(layers.Dropout(0.3))
        layers_list.append(layers.Dense(num_classes))
        
        # Initialize Sequential with all layers
        super(StandardArchitectureModel, self).__init__(layers_list, **kwargs)


class BasicBlock(layers.Layer):
    """
    Basic residual block for ResNet18/34.
    
    Structure: Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN -> Add -> ReLU
    """
    def __init__(self, filters, stride=1, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        
        self.filters = filters
        self.stride = stride
        
        # Main path
        self.conv1 = layers.Conv2D(filters, kernel_size=3, strides=stride, 
                                   padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(filters, kernel_size=3, strides=1,
                                   padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        # Shortcut path
        self.use_shortcut = (stride != 1)
        if self.use_shortcut:
            self.shortcut_conv = layers.Conv2D(filters, kernel_size=1,
                                               strides=stride, use_bias=False)
            self.shortcut_bn = layers.BatchNormalization()
    
    def call(self, x, training=False):
        residual = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        
        # Shortcut path
        if self.use_shortcut:
            residual = self.shortcut_conv(residual)
            residual = self.shortcut_bn(residual, training=training)
        
        # Add residual
        out = layers.Add()([out, residual])
        out = tf.nn.relu(out)
        
        return out


class StandardArchitectureModelNoAttention(keras.Sequential):
    """
    Same architecture as StandardArchitectureModel but without the channel attention layer.
    
    Architecture:
    - Initial Conv (7x7) + MaxPool
    - Layer1: 2 Residual Blocks (64 -> 256 channels)
    - Layer2: 3 Residual Blocks (256 -> 512 channels, stride=2)
    - Layer3: 3 Residual Blocks (512 -> 512 channels, stride=2)
    - Global Average Pooling (no attention)
    - Classifier: Dense(512->256) -> Dense(256->29)
    """
    def __init__(self, num_classes=29, input_channels=3, **kwargs):
        # Build the sequential model
        layers_list = []
        
        # Initial convolution layer
        layers_list.append(layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False))
        layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Activation('relu'))
        layers_list.append(layers.MaxPooling2D(pool_size=3, strides=2, padding='same'))
        
        # Residual blocks - Layer 1 (2 blocks: 64 -> 256 channels)
        layers_list.append(ResidualBlock(64, 64, 256, stride=1))
        layers_list.append(ResidualBlock(256, 64, 256, stride=1))
        
        # Residual blocks - Layer 2 (3 blocks: 256 -> 512 channels)
        layers_list.append(ResidualBlock(256, 128, 512, stride=2))
        layers_list.append(ResidualBlock(512, 128, 512, stride=1))
        layers_list.append(ResidualBlock(512, 128, 512, stride=1))
        
        # Residual blocks - Layer 3 (3 blocks: 512 -> 512 channels)
        layers_list.append(ResidualBlock(512, 256, 512, stride=2))
        layers_list.append(ResidualBlock(512, 256, 512, stride=1))
        layers_list.append(ResidualBlock(512, 256, 512, stride=1))
        
        # No attention mechanism - directly to global average pooling
        layers_list.append(layers.GlobalAveragePooling2D())
        
        # Classifier with dropout
        layers_list.append(layers.Dropout(0.3))
        layers_list.append(layers.Dense(256))
        layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Activation('relu'))
        layers_list.append(layers.Dropout(0.3))
        layers_list.append(layers.Dense(num_classes))
        
        # Initialize Sequential with all layers
        super(StandardArchitectureModelNoAttention, self).__init__(layers_list, **kwargs)


class ResNet18(keras.Sequential):
    """

    Structure:
    - Initial Conv 7x7 + MaxPool
    - Layer1: 2 BasicBlocks (64 filters)
    - Layer2: 2 BasicBlocks (128 filters, stride=2)
    - Layer3: 2 BasicBlocks (256 filters, stride=2)
    - Layer4: 2 BasicBlocks (512 filters, stride=2)
    - Global Average Pooling
    - Fully Connected layer (29 classes)
    """
    def __init__(self, num_classes=29, **kwargs):
        layers_list = []
        
        # Initial convolution
        layers_list.append(layers.Conv2D(64, kernel_size=7, strides=2, 
                                         padding='same', use_bias=False))
        layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Activation('relu'))
        layers_list.append(layers.MaxPooling2D(pool_size=3, strides=2, padding='same'))
        
        # Layer 1: 2 blocks, 64 filters, stride=1
        layers_list.append(BasicBlock(64, stride=1))
        layers_list.append(BasicBlock(64, stride=1))
        
        # Layer 2: 2 blocks, 128 filters, first with stride=2
        layers_list.append(BasicBlock(128, stride=2))
        layers_list.append(BasicBlock(128, stride=1))
        
        # Layer 3: 2 blocks, 256 filters, first with stride=2
        layers_list.append(BasicBlock(256, stride=2))
        layers_list.append(BasicBlock(256, stride=1))
        
        # Layer 4: 2 blocks, 512 filters, first with stride=2
        layers_list.append(BasicBlock(512, stride=2))
        layers_list.append(BasicBlock(512, stride=1))
        
        # Global average pooling
        layers_list.append(layers.GlobalAveragePooling2D())
        
        # Classifier
        layers_list.append(layers.Dense(num_classes))
        
        # Initialize Sequential with all layers
        super(ResNet18, self).__init__(layers_list, **kwargs)
