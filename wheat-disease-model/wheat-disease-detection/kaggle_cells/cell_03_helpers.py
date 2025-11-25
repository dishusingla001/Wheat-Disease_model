# Cell 3: Helper Functions
"""
Define utility functions for data loading and preprocessing.
These functions handle image loading, conversion, and label encoding.
"""

def convert_gif_to_png(dataset_dir):
    """
    Convert any .gif files to .png format.
    Usually not needed on Kaggle, but included for completeness.
    """
    count = 0
    for subdir, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".gif"):
                gif_path = os.path.join(subdir, file)
                try:
                    img = Image.open(gif_path).convert("RGB")
                    new_path = gif_path.replace(".gif", ".png")
                    img.save(new_path)
                    os.remove(gif_path)
                    count += 1
                    print(f"Converted: {file}")
                except Exception as e:
                    print(f"Failed to convert {file}: {e}")
    if count > 0:
        print(f"✅ Converted {count} GIF files to PNG")
    else:
        print("✅ No GIF files found to convert")


def load_dataset(dataset_path):
    """
    Load all images from the dataset directory.
    Returns: images array, labels array, and label binarizer object
    """
    images = []
    labels = []
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        raise ValueError(f"❌ Dataset path does not exist: {dataset_path}")
    
    # Get all class folders (disease types)
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"📁 Found {len(classes)} classes: {classes}")
    
    # Load images from each class
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif'))]
        
        print(f"\n📂 Loading class: {class_name}")
        print(f"   Found {len(image_files)} images")
        
        for idx, image_name in enumerate(image_files):
            image_path = os.path.join(class_path, image_name)
            try:
                # Read and resize image
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                    images.append(image)
                    labels.append(class_name)
                else:
                    print(f"   ⚠️ Skipped invalid image: {image_name}")
            except Exception as e:
                print(f"   ❌ Error loading {image_name}: {e}")
            
            # Progress indicator
            if (idx + 1) % 500 == 0:
                print(f"   Loaded {idx + 1}/{len(image_files)} images...")
    
    print(f"\n✅ Total images loaded: {len(images)}")
    
    # Encode labels (convert class names to numbers)
    lb_path = os.path.join(OUTPUT_DIR, LB_FILENAME)
    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(labels)
    
    # Save label binarizer for later use
    with open(lb_path, "wb") as f:
        pickle.dump(label_binarizer, f)
    print(f"✅ Label binarizer saved to {LB_FILENAME}")
    
    return np.array(images), np.array(labels), label_binarizer


def preprocess_images(images):
    """
    Normalize pixel values from [0, 255] to [0, 1].
    This helps the neural network train faster and more effectively.
    """
    return images.astype('float32') / 255.0


print("✅ Helper functions defined successfully!")
