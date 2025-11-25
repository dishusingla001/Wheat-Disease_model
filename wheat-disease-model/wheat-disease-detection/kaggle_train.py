import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from PIL import Image
import pickle

# Constants for Kaggle
# Kaggle datasets are typically mounted at /kaggle/input/
# Update this path to match your Kaggle dataset name
DATASET_PATH = '/kaggle/input/wheat-disease-dataset/cropDiseaseDataset'
# Output directory for Kaggle (results are saved to /kaggle/working/)
OUTPUT_DIR = '/kaggle/working'

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50  # Increased for better training on Kaggle GPU
MODEL_FILENAME = 'wheatDiseaseModel.keras'
LB_FILENAME = 'label_binarizer.pkl'

# Helper functions
def convert_gif_to_png(dataset_dir):
    """Convert any .gif files to .png format"""
    for subdir, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".gif"):
                gif_path = os.path.join(subdir, file)
                try:
                    img = Image.open(gif_path).convert("RGB")
                    new_path = gif_path.replace(".gif", ".png")
                    img.save(new_path)
                    os.remove(gif_path)
                    print(f"Converted: {gif_path} -> {new_path}")
                except Exception as e:
                    print(f"Failed to convert {gif_path}: {e}")


def load_dataset(dataset_path):
    """Load and preprocess the dataset"""
    images = []
    labels = []
    label_binarizer = None
    
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    classes = os.listdir(dataset_path)
    print(f"Found classes: {classes}")

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            image_files = os.listdir(class_path)
            print(f"Loading {len(image_files)} images from {class_name}...")
            
            for image_name in image_files:
                image_path = os.path.join(class_path, image_name)
                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                        images.append(image)
                        labels.append(class_name)
                    else:
                        print(f"Skipping invalid image: {image_path}")
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
    
    print(f"\nTotal images loaded: {len(images)}")
    
    # Label encoding
    lb_path = os.path.join(OUTPUT_DIR, LB_FILENAME)
    if os.path.exists(lb_path):
        try:
            with open(lb_path, "rb") as f:
                label_binarizer = pickle.load(f)
            labels = label_binarizer.transform(labels)
            print("Loaded existing label binarizer")
        except Exception as e:
            print(f"Failed to load existing label binarizer: {e}. Creating new one.")
            label_binarizer = LabelBinarizer()
            labels = label_binarizer.fit_transform(labels)
            with open(lb_path, "wb") as f:
                pickle.dump(label_binarizer, f)
    else:
        label_binarizer = LabelBinarizer()
        labels = label_binarizer.fit_transform(labels)
        with open(lb_path, "wb") as f:
            pickle.dump(label_binarizer, f)
        print("Created new label binarizer")

    return np.array(images), np.array(labels), label_binarizer


def preprocess_images(images):
    """Normalize images to [0, 1] range"""
    return images.astype('float32') / 255.0


def build_improved_model(input_shape, num_classes):
    """Build an improved CNN model with batch normalization and dropout"""
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Fully connected layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_history(history):
    """Plot training history"""
    plt.figure(figsize=(14, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='orange', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150)
    plt.show()
    print(f"Training history plot saved to {OUTPUT_DIR}/training_history.png")


# Main training logic
if __name__ == "__main__":
    print("="*70)
    print("🌾 WHEAT DISEASE DETECTION - KAGGLE TRAINING")
    print("="*70)
    
    # Check if running on Kaggle
    if not os.path.exists('/kaggle'):
        print("⚠️  Warning: Not running on Kaggle. Using local paths.")
        DATASET_PATH = os.path.join(os.path.dirname(__file__), 'cropDiseaseDataset')
        OUTPUT_DIR = os.path.dirname(__file__)
    
    print(f"📂 Dataset Path: {DATASET_PATH}")
    print(f"📂 Output Path: {OUTPUT_DIR}")
    
    # Convert GIFs if needed (usually not necessary on Kaggle)
    if os.path.exists(DATASET_PATH):
        convert_gif_to_png(DATASET_PATH)
    
    print("\n🔍 Loading dataset...")
    images, labels, label_binarizer = load_dataset(DATASET_PATH)
    
    print("\n🔄 Preprocessing images...")
    images = preprocess_images(images)
    
    NUM_CLASSES = len(label_binarizer.classes_)
    print(f"\n📊 Dataset Information:")
    print(f"   • Number of Classes: {NUM_CLASSES}")
    print(f"   • Classes: {list(label_binarizer.classes_)}")
    print(f"   • Total Images: {len(images)}")
    print(f"   • Image Size: {IMG_SIZE}x{IMG_SIZE}")
    
    if NUM_CLASSES == 0:
        raise RuntimeError(f"No classes found in {DATASET_PATH}")
    
    # Split dataset with validation set
    print("\n✂️  Splitting dataset...")
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"   • Training Images: {len(train_images)}")
    print(f"   • Validation Images: {len(val_images)}")
    print(f"   • Test Images: {len(test_images)}")
    
    # Build model
    print("\n🏗️  Building improved model...")
    model = build_improved_model((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)
    print(model.summary())
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, MODEL_FILENAME),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Data augmentation
    print("\n🔄 Setting up data augmentation...")
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()  # No augmentation for validation
    
    train_datagen.fit(train_images)
    
    # Train model
    print("\n🚀 Starting training...")
    print("="*70)
    history = model.fit(
        train_datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
        validation_data=val_datagen.flow(val_images, val_labels, batch_size=BATCH_SIZE),
        steps_per_epoch=max(1, math.ceil(len(train_images) / BATCH_SIZE)),
        validation_steps=max(1, math.ceil(len(val_images) / BATCH_SIZE)),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    print("\n📊 Plotting training history...")
    plot_history(history)
    
    # Save final model
    print("\n💾 Saving final model...")
    model_path = os.path.join(OUTPUT_DIR, MODEL_FILENAME)
    model.save(model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Evaluate on test set
    print("\n🔍 Evaluating model on test set...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    
    # Get best validation accuracy from history
    best_val_acc = max(history.history['val_accuracy'])
    best_val_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    
    # Final summary
    print("\n" + "="*70)
    print("🎯 TRAINING COMPLETED!")
    print("="*70)
    print(f"\n📈 Final Results:")
    print(f"   • Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) at epoch {best_val_epoch}")
    print(f"   • Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   • Test Loss: {test_loss:.4f}")
    print(f"   • Total Epochs Trained: {len(history.history['loss'])}")
    
    print(f"\n📁 Output Files:")
    print(f"   • Model: {model_path}")
    print(f"   • Label Binarizer: {os.path.join(OUTPUT_DIR, LB_FILENAME)}")
    print(f"   • Training Plot: {os.path.join(OUTPUT_DIR, 'training_history.png')}")
    
    # Performance assessment
    if test_acc >= 0.95:
        print(f"\n🌟 EXCELLENT! Your model achieved {test_acc*100:.2f}% test accuracy!")
    elif test_acc >= 0.90:
        print(f"\n🎉 GREAT! Your model achieved {test_acc*100:.2f}% test accuracy!")
    elif test_acc >= 0.80:
        print(f"\n👍 GOOD! Your model achieved {test_acc*100:.2f}% test accuracy!")
    elif test_acc >= 0.70:
        print(f"\n⚠️  FAIR! Your model achieved {test_acc*100:.2f}% test accuracy.")
    else:
        print(f"\n⚠️  Model achieved {test_acc*100:.2f}% test accuracy. Consider:")
        print("   • Adding more training data")
        print("   • Training for more epochs")
        print("   • Adjusting model architecture")
    
    print("\n" + "="*70)
    print("✨ Download the model files from /kaggle/working/ to use locally!")
    print("="*70)
