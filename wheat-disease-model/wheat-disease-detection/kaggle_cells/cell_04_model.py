# Cell 4: Build Model Architecture
"""
Define the CNN model architecture with batch normalization and dropout.
This is an improved architecture compared to the basic model.
"""

def build_improved_model(input_shape, num_classes):
    """
    Build a deep CNN model with:
    - 3 convolutional blocks (each with 2 conv layers)
    - Batch Normalization (speeds up training)
    - Dropout (prevents overfitting)
    - Dense layers for classification
    """
    model = Sequential([
        # ========== Block 1: Extract basic features ==========
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),  # Normalizes activations
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),  # Reduces spatial dimensions by half
        Dropout(0.25),         # Randomly drops 25% of connections
        
        # ========== Block 2: Learn intermediate features ==========
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # ========== Block 3: Learn complex features ==========
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # ========== Fully Connected Layers ==========
        Flatten(),             # Convert 2D feature maps to 1D
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),          # Higher dropout for dense layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # ========== Output Layer ==========
        Dense(num_classes, activation='softmax')  # Probability for each class
    ])
    
    # Compile model with optimizer and loss function
    model.compile(
        optimizer='adam',                      # Adaptive learning rate optimizer
        loss='categorical_crossentropy',       # Loss for multi-class classification
        metrics=['accuracy']                   # Track accuracy during training
    )
    
    return model


print("✅ Model architecture function defined!")
print("\nModel Architecture Overview:")
print("  • 3 Convolutional Blocks (32→64→128 filters)")
print("  • Batch Normalization after each Conv layer")
print("  • Dropout for regularization (prevents overfitting)")
print("  • 2 Dense layers (256→128 neurons)")
print("  • Output layer with softmax activation")
