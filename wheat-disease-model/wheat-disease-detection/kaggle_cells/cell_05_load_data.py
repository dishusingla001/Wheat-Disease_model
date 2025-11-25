# Cell 5: Load and Preprocess Dataset
"""
Load all images from the dataset and prepare them for training.
This cell will take a few minutes depending on dataset size.
"""

print("🔄 Starting dataset loading process...")
print("="*70)

# Convert any GIF files to PNG (if needed)
if os.path.exists(DATASET_PATH):
    convert_gif_to_png(DATASET_PATH)
else:
    print(f"❌ ERROR: Dataset path not found: {DATASET_PATH}")
    print("Please update DATASET_PATH in Cell 2 to match your Kaggle dataset!")
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

# Load all images and labels
print("\n📥 Loading images from dataset...")
images, labels, label_binarizer = load_dataset(DATASET_PATH)

# Normalize images (convert to 0-1 range)
print("\n🔄 Normalizing images...")
images = preprocess_images(images)
print("✅ Images normalized to [0, 1] range")

# Get number of classes
NUM_CLASSES = len(label_binarizer.classes_)

# Display dataset statistics
print("\n" + "="*70)
print("📊 DATASET SUMMARY")
print("="*70)
print(f"Number of Classes: {NUM_CLASSES}")
print(f"Classes: {list(label_binarizer.classes_)}")
print(f"Total Images: {len(images)}")
print(f"Image Shape: {images[0].shape}")
print(f"Label Shape: {labels[0].shape}")
print("="*70)

# Check if we have data
if NUM_CLASSES == 0:
    raise RuntimeError("❌ No classes found in dataset!")
if len(images) == 0:
    raise RuntimeError("❌ No images loaded!")

print("\n✅ Dataset loaded and preprocessed successfully!")
