import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(__file__);
IMAGE_DIR = os.path.join(BASE_DIR,'segmentation model','mask_folder','generated_images')
MASK_DIR = os.path.join(BASE_DIR,'segmentation model','mask_folder','generated_masks')

IMG_HEIGHT, IMG_WIDTH = 256, 256

# Model file and training config
PARENT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PARENT_DIR,'wheat_unet_model.h5')

EPOCHS = 20

# Load dataset (recursive: supports images in subfolders)
def load_data(image_dir, mask_dir, img_height, img_width):
    images = []
    masks = []

    # valid image/mask extensions (include common JPEG/JFIF variants)
    valid_exts = [".png", ".jpg", ".jpeg", ".jfif"]

    # loop over categories (subfolders)
    for category in os.listdir(image_dir):
        image_category_path = os.path.join(image_dir, category)
        mask_category_path = os.path.join(mask_dir, category)

        if not os.path.isdir(image_category_path):
            continue  # skip files

        # dictionary of masks ignoring extension
        mask_files = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(mask_category_path)
        }

        # loop over images in category
        for img_name in os.listdir(image_category_path):
            if os.path.splitext(img_name)[1].lower() not in valid_exts:
                continue

            image_path = os.path.join(image_category_path, img_name)
            base_name = os.path.splitext(img_name)[0]

            # check if corresponding mask exists
            if base_name not in mask_files:
                print(f"⚠️ Mask not found for: {img_name}")
                continue

            mask_path = os.path.join(mask_category_path, mask_files[base_name])

            # load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️ Could not read image: {image_path}")
                continue
            image = cv2.resize(image, (img_width, img_height))
            image = image / 255.0
            images.append(image)

            # load mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"⚠️ Could not read mask: {mask_path}")
                continue
            mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            mask = mask / 255.0
            mask = np.expand_dims(mask, axis=-1)  # make it (H, W, 1)
            masks.append(mask)

    return np.array(images), np.array(masks)
# (data existence check runs after loading X,Y below)



# Load data
X, Y = load_data(IMAGE_DIR,MASK_DIR,IMG_HEIGHT,IMG_WIDTH)
print(f"✅ Loaded {len(X)} images and masks.")

# If no images loaded, print a helpful message and exit early
if len(X) == 0:
    print("\nNo images were loaded. Check that: ")
    print(f" - IMAGE_DIR is correct: {IMAGE_DIR}")
    print(f" - MASK_DIR is correct: {MASK_DIR}")
    print(" - Each image has a matching mask file with the same base filename inside the corresponding subfolder under MASK_DIR.")
    print(" - Mask files can be .png/.jpg/.jpeg/.jfif.\n")
    print("You can run a quick check listing a few files, e.g:")
    print("  Get-ChildItem -Path 'D:\\Wheat-diesease\\wheat-disease-model\\wheat-disease-detection\\segmentation model\\wheat_images' -Recurse | Select-Object -First 20")
    raise SystemExit(1)

# Train-test split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

# U-Net Model
def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(c9)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# Load existing model if present, otherwise build a new one
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH, compile=False)
        print(f"🔁 Loaded existing model from {MODEL_PATH}, continuing training.")
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    except Exception as e:
        print(f"⚠️  Failed to load model at {MODEL_PATH} ({e}). Building a new model.")
        model = unet_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
else:
    model = unet_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train (continues training if model was loaded)
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=8, epochs=EPOCHS)

# Save model (overwrite/update)
model.save(MODEL_PATH)
print(f"💾 Model saved/updated at {MODEL_PATH}")

# Prediction + Disease Percentage

def disease_percentage(pred_mask):
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    diseased_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size
    return (diseased_pixels / total_pixels) * 100

# Example prediction (disabled GUI display on purpose; saves files instead)
if len(X_val) > 0:
    test_img = X_val[0].reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)
    pred = model.predict(test_img)[0]
    percentage = disease_percentage(pred)

    # Save example outputs
    os.makedirs('example_outputs', exist_ok=True)
    orig_save = os.path.join('example_outputs', 'original_example.png')
    mask_save = os.path.join('example_outputs', 'predicted_mask_example.png')
    cv2.imwrite(orig_save, (X_val[0] * 255).astype(np.uint8))
    cv2.imwrite(mask_save, ((pred > 0.5).astype(np.uint8) * 255).squeeze())
    print(f"🌿 Diseased Area: {percentage:.2f}%")
    print(f"Saved example images to {os.path.abspath('example_outputs')}")
else:
    print("No validation images available to run example prediction.")
