# Cell 8: Setup Training Configuration
"""
Configure callbacks and data augmentation for robust training:
- EarlyStopping: Stop training if no improvement (prevent overfitting)
- ModelCheckpoint: Save best model during training
- ReduceLROnPlateau: Reduce learning rate when stuck on plateau
- Data Augmentation: Generate variations of training images
"""

print("⚙️  Setting up training configuration...")
print("="*70)

# Define callbacks for training
callbacks = [
    # Stop training if validation loss doesn't improve for 10 epochs
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Save the best model based on validation loss
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    # Reduce learning rate if stuck on plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

print("✅ Callbacks configured:")
print("   • EarlyStopping (patience=10)")
print("   • ModelCheckpoint (save best model)")
print("   • ReduceLROnPlateau (factor=0.5, patience=5)")

# Data augmentation for training set
# This creates variations of images to make model more robust
datagen = ImageDataGenerator(
    rotation_range=30,        # Rotate images by up to 30 degrees
    width_shift_range=0.2,    # Shift horizontally by up to 20%
    height_shift_range=0.2,   # Shift vertically by up to 20%
    horizontal_flip=True,     # Flip images horizontally
    vertical_flip=True,       # Flip images vertically
    zoom_range=0.2,           # Zoom in/out by up to 20%
    shear_range=0.2,          # Shear transformation
    fill_mode='nearest'       # Fill empty pixels with nearest value
)

print("\n✅ Data augmentation configured:")
print("   • Rotation: ±30°")
print("   • Shifts: ±20% (horizontal & vertical)")
print("   • Flips: horizontal & vertical")
print("   • Zoom: ±20%")
print("   • Shear: 20%")

# Fit data generator on training data
print("\n📊 Fitting data generator on training data...")
datagen.fit(train_images)

# Create data generators
train_generator = datagen.flow(
    train_images, train_labels,
    batch_size=BATCH_SIZE
)

val_generator = ImageDataGenerator().flow(
    val_images, val_labels,
    batch_size=BATCH_SIZE
)

print(f"✅ Training generator: {len(train_generator)} batches")
print(f"✅ Validation generator: {len(val_generator)} batches")

print("="*70)
print("✅ Training configuration complete!")
