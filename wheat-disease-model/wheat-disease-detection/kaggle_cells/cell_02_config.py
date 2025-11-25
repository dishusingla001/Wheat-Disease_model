# Cell 2: Configuration and Constants
"""
Set up all configuration parameters for training.
IMPORTANT: Update DATASET_PATH to match your Kaggle dataset name!
"""

# Kaggle paths - UPDATE THIS to match your dataset name
DATASET_PATH = '/kaggle/input/wheat-disease-dataset/cropDiseaseDataset'
OUTPUT_DIR = '/kaggle/working'

# Training parameters
IMG_SIZE = 64          # Image will be resized to 64x64 pixels
BATCH_SIZE = 32        # Number of images per training batch
EPOCHS = 50            # Number of complete passes through the dataset

# File names for saving
MODEL_FILENAME = 'wheatDiseaseModel.keras'
LB_FILENAME = 'label_binarizer.pkl'

print("="*70)
print("🌾 WHEAT DISEASE DETECTION - KAGGLE TRAINING")
print("="*70)
print(f"📂 Dataset Path: {DATASET_PATH}")
print(f"📂 Output Path: {OUTPUT_DIR}")
print(f"🖼️  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"📦 Batch Size: {BATCH_SIZE}")
print(f"🔄 Epochs: {EPOCHS}")
print("="*70)
