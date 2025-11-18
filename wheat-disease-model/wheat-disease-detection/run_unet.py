import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------
# CONFIG
# ---------------------
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3


input_folder = r"D:\Wheat-diesease\wheat-disease-model\wheat-disease-detection\segmentation model\Manual Mask\Wheat_crown_root_rot"   # folder with test images
output_folder = r"D:\Wheat-diesease\wheat-disease-model\wheat-disease-detection\predicted_masks"    # folder to save masks

os.makedirs(output_folder, exist_ok=True)

# ---------------------
# LOAD MODEL
# ---------------------
model = load_model("wheat_unet_model.h5", compile=False)

# ---------------------
# FUNCTION TO PROCESS & PREDICT
# ---------------------
def predict_mask(image_path):
    # Read input image
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read {image_path}")
        return None

    # Resize and normalize
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    img_resized = img_resized / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    # Predict mask
    pred_mask = model.predict(img_input)[0]  # (128,128,1)
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    pred_mask = np.squeeze(pred_mask, axis=-1)  # (128,128)

    # Resize mask back to original size
    pred_mask_resized = cv2.resize(pred_mask, (img.shape[1], img.shape[0]))

    return pred_mask_resized

# ---------------------
# LOOP THROUGH TEST IMAGES
# ---------------------
valid_exts = (".png", ".jpg", ".jpeg")

for filename in os.listdir(input_folder):
    if filename.lower().endswith(valid_exts):
        image_path = os.path.join(input_folder, filename)
        mask = predict_mask(image_path)

        if mask is None:
            continue

        # Save predicted mask
        save_path = os.path.join(output_folder, f"mask_{filename}")
        cv2.imwrite(save_path, mask)

        print(f"✅ Saved mask for {filename} at {save_path}")
