import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Paths
BASE_DIR = os.path.dirname(__file__)
MASK_DIR = os.path.join(BASE_DIR, 'segmentation model', 'mask_folder', 'generated_masks')
IMAGE_DIR = os.path.join(BASE_DIR, 'segmentation model', 'mask_folder', 'generated_images')
TEST_DIR = os.path.join(BASE_DIR, 'NewtestCdd')
PARENT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PARENT_DIR, 'wheat_unet_model.h5')

print("=" * 60)
print("🔍 U-Net Model Diagnostics")
print("=" * 60)

# 1. Check training masks
print("\n1️⃣ Checking training masks...")
mask_stats = []
for category in os.listdir(MASK_DIR):
    cat_path = os.path.join(MASK_DIR, category)
    if not os.path.isdir(cat_path):
        continue
    
    mask_files = [f for f in os.listdir(cat_path) if f.endswith(('.png', '.jpg'))][:5]
    
    for mask_file in mask_files:
        mask_path = os.path.join(cat_path, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            white_pixels = np.sum(mask > 0)
            total_pixels = mask.size
            white_percent = (white_pixels / total_pixels) * 100
            mask_stats.append({
                'category': category,
                'file': mask_file,
                'white_percent': white_percent,
                'is_blank': white_percent == 0
            })

if mask_stats:
    blank_count = sum(1 for m in mask_stats if m['is_blank'])
    print(f"   Checked {len(mask_stats)} masks:")
    print(f"   - All-black masks: {blank_count}")
    print(f"   - Masks with disease: {len(mask_stats) - blank_count}")
    
    # Show samples
    print("\n   Sample masks:")
    for m in mask_stats[:10]:
        status = "⚫ ALL BLACK" if m['is_blank'] else f"⚪ {m['white_percent']:.2f}% white"
        print(f"   {m['category']}/{m['file']}: {status}")
else:
    print("   ⚠️ No masks found!")

# 2. Check model
print(f"\n2️⃣ Checking model at {MODEL_PATH}...")
if os.path.exists(MODEL_PATH):
    print(f"   ✅ Model exists ({os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB)")
    
    try:
        model = load_model(MODEL_PATH, compile=False)
        print(f"   ✅ Model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        exit(1)
else:
    print(f"   ❌ Model not found!")
    exit(1)

# 3. Test on a training image
print("\n3️⃣ Testing on a training image (should detect disease)...")
test_img_path = None
test_mask_path = None

for category in os.listdir(IMAGE_DIR):
    cat_path = os.path.join(IMAGE_DIR, category)
    if not os.path.isdir(cat_path):
        continue
    
    img_files = [f for f in os.listdir(cat_path) if f.endswith(('.png', '.jpg'))]
    if img_files:
        test_img_path = os.path.join(cat_path, img_files[0])
        
        # Find corresponding mask
        base_name = os.path.splitext(img_files[0])[0]
        mask_cat_path = os.path.join(MASK_DIR, category)
        if os.path.exists(mask_cat_path):
            mask_files = [f for f in os.listdir(mask_cat_path) 
                         if os.path.splitext(f)[0] == base_name]
            if mask_files:
                test_mask_path = os.path.join(mask_cat_path, mask_files[0])
        break

if test_img_path:
    print(f"   Using: {os.path.basename(test_img_path)}")
    
    # Load and preprocess
    img = cv2.imread(test_img_path)
    img_resized = cv2.resize(img, (256, 256))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)
    
    # Predict
    pred = model.predict(img_input, verbose=0)[0]
    
    # Stats
    print(f"   Raw prediction range: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"   Mean prediction: {pred.mean():.4f}")
    print(f"   Prediction shape: {pred.shape}")
    
    # Threshold at different values
    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        binary = (pred > thresh).astype(np.uint8)
        white_percent = (np.sum(binary) / binary.size) * 100
        print(f"   At threshold {thresh}: {white_percent:.2f}% predicted as disease")
    
    # Check ground truth
    if test_mask_path:
        gt_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            gt_resized = cv2.resize(gt_mask, (256, 256))
            gt_white = (np.sum(gt_resized > 0) / gt_resized.size) * 100
            print(f"   Ground truth has: {gt_white:.2f}% disease pixels")
else:
    print("   ⚠️ No training images found!")

# 4. Test on testCDD images
print("\n4️⃣ Testing on testCDD images...")
test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))][:5]

for test_file in test_files:
    test_path = os.path.join(TEST_DIR, test_file)
    img = cv2.imread(test_path)
    if img is None:
        continue
    
    img_resized = cv2.resize(img, (256, 256))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)
    
    pred = model.predict(img_input, verbose=0)[0]
    
    print(f"\n   {test_file}:")
    print(f"   - Prediction range: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"   - Mean: {pred.mean():.4f}")
    
    binary_05 = (pred > 0.5).astype(np.uint8)
    binary_01 = (pred > 0.1).astype(np.uint8)
    
    white_05 = (np.sum(binary_05) / binary_05.size) * 100
    white_01 = (np.sum(binary_01) / binary_01.size) * 100
    
    print(f"   - At 0.5 threshold: {white_05:.2f}% disease")
    print(f"   - At 0.1 threshold: {white_01:.2f}% disease")

print("\n" + "=" * 60)
print("✅ Diagnosis complete!")
print("=" * 60)
