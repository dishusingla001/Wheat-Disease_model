# Cell 7: Build Model
"""
Create the improved CNN model for wheat disease classification.
The model architecture includes:
- 3 convolutional blocks with batch normalization and dropout
- 2 fully connected layers with dropout
- Output layer with softmax for multi-class classification
"""

print("🏗️  Building improved CNN model...")
print("="*70)

# Build the model using our custom function
model = build_improved_model(NUM_CLASSES)

# Display model architecture
print("\n📋 Model Architecture:")
model.summary()

# Display key model information
total_params = model.count_params()
print(f"\n📊 Model Statistics:")
print(f"   Total Parameters: {total_params:,}")
print(f"   Input Shape:      {model.input_shape}")
print(f"   Output Shape:     {model.output_shape}")
print(f"   Number of Layers: {len(model.layers)}")

print("="*70)
print("✅ Model built successfully!")
