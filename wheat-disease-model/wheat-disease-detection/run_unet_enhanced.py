import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---------------------
# CONFIG
# ---------------------
IMG_HEIGHT = 256
IMG_WIDTH = 256
THRESHOLD = 0.15  # Lowered from 0.5 based on model behavior

BASE_DIR = os.path.dirname(__file__)
input_folder = os.path.join(BASE_DIR, 'testCDD')
output_folder = os.path.join(BASE_DIR, 'mask_by_unet')
overlay_folder = os.path.join(BASE_DIR, 'mask_by_unet', 'overlays')

os.makedirs(output_folder, exist_ok=True)
os.makedirs(overlay_folder, exist_ok=True)

# ---------------------
# LOAD MODEL
# ---------------------
PARENT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PARENT_DIR, 'wheat_unet_model.h5')
print(f"Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!\n")

# ---------------------
# FUNCTION TO PROCESS & PREDICT
# ---------------------
def predict_mask(image_path):
    # Read input image
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read {image_path}")
        return None, None, None

    # Store original for overlay
    original = img.copy()

    # Resize and normalize
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_resized = img_resized / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    # Predict mask
    pred_mask = model.predict(img_input, verbose=0)[0]  # (256,256,1)
    
    # Get prediction stats
    pred_min, pred_max, pred_mean = pred_mask.min(), pred_mask.max(), pred_mask.mean()
    
    # Apply threshold
    binary_mask = (pred_mask > THRESHOLD).astype(np.uint8) * 255
    binary_mask = np.squeeze(binary_mask, axis=-1)  # (256,256)

    # Resize mask back to original size
    mask_resized = cv2.resize(binary_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create overlay
    overlay = original.copy()
    # Make disease regions red with transparency
    red_mask = np.zeros_like(original)
    red_mask[:, :, 2] = mask_resized  # Red channel
    overlay = cv2.addWeighted(original, 0.7, red_mask, 0.3, 0)

    return mask_resized, overlay, (pred_min, pred_max, pred_mean)

# ---------------------
# LOOP THROUGH TEST IMAGES
# ---------------------
valid_exts = (".png", ".jpg", ".jpeg")
results = []

print("=" * 70)
print(f"{'Image':<20} {'Disease %':<12} {'Pred Range':<25} {'Status'}")
print("=" * 70)

for filename in sorted(os.listdir(input_folder)):
    if filename.lower().endswith(valid_exts):
        image_path = os.path.join(input_folder, filename)
        mask, overlay, stats = predict_mask(image_path)

        if mask is None:
            continue

        # Calculate disease percentage
        disease_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        disease_percent = (disease_pixels / total_pixels) * 100

        pred_min, pred_max, pred_mean = stats

        # Save predicted mask
        mask_path = os.path.join(output_folder, f"mask_{filename}")
        cv2.imwrite(mask_path, mask)

        # Save overlay
        overlay_path = os.path.join(overlay_folder, f"overlay_{filename}")
        cv2.imwrite(overlay_path, overlay)

        # Determine status
        if disease_percent < 1:
            status = "Healthy ✓"
        elif disease_percent < 10:
            status = "Minor 🟡"
        elif disease_percent < 30:
            status = "Moderate 🟠"
        else:
            status = "Severe 🔴"

        results.append({
            'filename': filename,
            'disease_percent': disease_percent,
            'pred_range': f"[{pred_min:.3f}, {pred_max:.3f}]",
            'pred_mean': pred_mean,
            'status': status
        })

        print(f"{filename:<20} {disease_percent:>6.2f}%     {pred_min:.3f}-{pred_max:.3f} (μ={pred_mean:.3f})   {status}")

print("=" * 70)
print(f"\n✅ Processed {len(results)} images")
print(f"\n📁 Outputs saved to:")
print(f"   • Masks: {output_folder}")
print(f"   • Overlays: {overlay_folder}")

# Summary statistics
if results:
    avg_disease = np.mean([r['disease_percent'] for r in results])
    max_disease = max(results, key=lambda x: x['disease_percent'])
    min_disease = min(results, key=lambda x: x['disease_percent'])
    
    print(f"\n📊 Summary:")
    print(f"   • Average disease: {avg_disease:.2f}%")
    print(f"   • Most affected: {max_disease['filename']} ({max_disease['disease_percent']:.2f}%)")
    print(f"   • Least affected: {min_disease['filename']} ({min_disease['disease_percent']:.2f}%)")
    
    # Count by severity
    healthy = sum(1 for r in results if r['disease_percent'] < 1)
    minor = sum(1 for r in results if 1 <= r['disease_percent'] < 10)
    moderate = sum(1 for r in results if 10 <= r['disease_percent'] < 30)
    severe = sum(1 for r in results if r['disease_percent'] >= 30)
    
    print(f"\n🏥 Disease Distribution:")
    print(f"   • Healthy: {healthy}")
    print(f"   • Minor: {minor}")
    print(f"   • Moderate: {moderate}")
    print(f"   • Severe: {severe}")

print("\n" + "=" * 70)
