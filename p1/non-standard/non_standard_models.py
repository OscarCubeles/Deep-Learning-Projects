import tensorflow as tf
import tensorflow as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2 

# Fixed parameters
WEIGHT_DECAY = 5e-4
PATCH_SIZE = 16  
EMBED_DIM = 192 
NUM_HEADS = 6    
FF_DIM = 768     
NUM_BLOCKS = 12
DROPOUT_RATE = 0.1

def InceptionV3(model_name, num_classes, input_size):

    if model_name == 'InceptionV3':
        
        # L2 regularizer
        l2_reg = L2(WEIGHT_DECAY)
        
        base_model = tf.keras.applications.InceptionV3(
            weights=None,
            include_top=False,
            input_shape=(input_size, input_size, 3)
        )
        
        # Custom Head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)

        # Middle Layers
        x = layers.Dense(1024, activation='relu', kernel_regularizer=l2_reg)(x)
        x = layers.Dropout(0.4)(x) 
        
        # Output Layer
        predictions = layers.Dense(num_classes, activation='softmax', name='predictions', kernel_regularizer=l2_reg)(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
    else:
        raise ValueError(f"Model '{model_name}' not found.")
        
    print(f"Using {model_name} with {num_classes} classes.")
    return model




class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        # Extracts patches from the image
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # Flattens the patches for sequence processing
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])

# Patch Encoding (Embedding + Positional Encoding)
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.num_patches = num_patches
        # Patch Embedding
        self.projection = layers.Dense(units=embed_dim) 
        # Positional Embeddings
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=embed_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class ViTBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs) 
                
        # Multi-Head Attention
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        
        # Feed-Forward Network 
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        # Normalization and Dropout layers
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False): 
        
        # Attention with Residual connection
        x = self.layernorm1(inputs)
        x = self.att(x, x)

        x = self.dropout1(x, training=training) 
        att_output = x + inputs

        # FFN with Residual connection
        x = self.layernorm2(att_output)
        x = self.ffn(x)

        x = self.dropout2(x, training=training) 
        return x + att_output

def ViTModel(model_name, num_classes, input_size):
    if model_name != 'ViTModel':
        raise ValueError(f"Model '{model_name}' not found.")
    
    num_patches = (input_size // PATCH_SIZE) ** 2
    inputs = layers.Input(shape=(input_size, input_size, 3))
    
    # Patch and encoding
    patches = Patches(PATCH_SIZE)(inputs)

    # Create embeddings and add positional information
    encoded_patches = PatchEncoder(num_patches, EMBED_DIM)(patches)
    
    # Apply Transformer blocks
    x = encoded_patches
    for _ in range(NUM_BLOCKS):
        x = ViTBlock(EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT_RATE)(x)
        
    # Classification head
    representation = layers.GlobalAveragePooling1D()(x) 
    
    representation = layers.Dropout(0.5)(representation)
    output = layers.Dense(num_classes, activation="softmax")(representation)

    model = Model(inputs=inputs, outputs=output)
    return model