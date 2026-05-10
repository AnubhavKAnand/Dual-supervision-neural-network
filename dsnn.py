import tensorflow as tf
from tensorflow.keras import layers, Model

def build_dsnn(input_shape=(256, 256, 1)): # 256x256 as per the pre-processing specs
    inputs = layers.Input(shape=input_shape)

    # ---------------------------------------------------------
    # 1. Feature Extraction Backbone (5 Hidden Layers)
    # ---------------------------------------------------------
    # Layer 1: 128 neurons
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(inputs)
    
    # Layer 2: 256 neurons
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
    
    # Layer 3: 256 neurons (Extracting high-level features here for the 2nd supervisor)
    high_level_feat = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
    
    # Layer 4: 128 neurons
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(high_level_feat)
    
    # Layer 5: 64 neurons (Low-level structural features)
    low_level_feat = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)

    # ---------------------------------------------------------
    # 2. Multi-scale Dilated Convolutions (Receptive Field Expansion)
    # ---------------------------------------------------------
    # Using different dilation rates to capture multi-scale information
    d1 = layers.Conv2D(64, kernel_size=3, padding='same', dilation_rate=1, activation='relu')(low_level_feat)
    
    concat_1 = layers.Concatenate()([low_level_feat, d1])
    d2 = layers.Conv2D(64, kernel_size=3, padding='same', dilation_rate=6, activation='relu')(concat_1)
    
    concat_2 = layers.Concatenate()([low_level_feat, d2])
    d3 = layers.Conv2D(64, kernel_size=3, padding='same', dilation_rate=12, activation='relu')(concat_2)
    
    concat_3 = layers.Concatenate()([low_level_feat, d3])
    d4 = layers.Conv2D(64, kernel_size=3, padding='same', dilation_rate=18, activation='relu')(concat_3)

    # Stitching operation to combine all scales
    f_d = layers.Concatenate()([d1, d2, d3, d4])

    # ---------------------------------------------------------
    # 3. Channel Attention Mechanism
    # ---------------------------------------------------------
    # Squeeze spatial dimensions to get channel descriptors
    ca = layers.GlobalAveragePooling2D()(f_d)
    
    # Generate attention weights using Softmax
    ca = layers.Dense(f_d.shape[-1] // 4, activation='relu')(ca)
    attention_weights = layers.Dense(f_d.shape[-1], activation='softmax')(ca)
    attention_weights = layers.Reshape((1, 1, f_d.shape[-1]))(attention_weights)
    
    # Re-weight the stitched features and add residual connection
    weighted_f_d = layers.Multiply()([f_d, attention_weights])
    f_z = layers.Add()([weighted_f_d, f_d]) 

    # ---------------------------------------------------------
    # 4. Dual Supervision Output Heads
    # ---------------------------------------------------------
    # Head 1: Final image enhancement/mask prediction (Linear activation)
    out_fuse = layers.Conv2D(input_shape[-1], kernel_size=1, padding='same', activation='linear', name='out_fuse')(f_z)
    
    # Head 2: High-level feature prediction for boundary/derivative supervision
    out_high = layers.Conv2D(input_shape[-1], kernel_size=1, padding='same', activation='linear', name='out_high')(high_level_feat)

    # Compile Model
    model = Model(inputs=inputs, outputs=[out_fuse, out_high], name="NC_FODE_DSNN")
    
    return model

# Instantiate and check the architecture
dsnn_model = build_dsnn()
dsnn_model.summary()