import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from PIL import Image
import pickle

# Constants
BASE_DIR = os.path.dirname(__file__)  # this is the directory in which now we are 
#  __file__ = give us currect directory 
DATASET_PATH = os.path.join(BASE_DIR,'cropDiseaseDataset')
# PREDICT_DIR = r'D:\Wheat-diesease\wheat-disease-detection\testCDD'
IMG_SIZE = 64
# NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 30
MODEL_FILENAME = 'wheatDiseaseModel.keras'
LB_FILENAME = 'label_binarizer.pkl'

# Helper functions this function change the .gif file or images to the .png 
def convert_gif_to_png(dataset_dir):
    for subdir, _ , files in os.walk(dataset_dir):
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


# this function load the dataset and all the images from the dataset 
# and also resize the images to 32x32 and convert the class name to one-hot endcoded array
def load_dataset(dataset_path):
    # this is the empty list which we used for the store valid images 
    # and in labels we store the class names
    images = []
    labels = []
    label_binarizer = None
    classes = os.listdir(dataset_path)
    print(classes)

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
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
    # Convert labels to one-hot encoding. If a previously saved
    # label binarizer exists, reuse it to preserve the mapping.
    if os.path.exists(LB_FILENAME):
        try:
            with open(LB_FILENAME, "rb") as f:
                label_binarizer = pickle.load(f)
            labels = label_binarizer.transform(labels)
        except Exception as e:
            print(f"Failed to load existing label binarizer: {e}. Re-fitting a new one.")
            label_binarizer = LabelBinarizer()
            labels = label_binarizer.fit_transform(labels)
            with open(LB_FILENAME, "wb") as f:
                pickle.dump(label_binarizer, f)
    else:
        label_binarizer = LabelBinarizer()
        labels = label_binarizer.fit_transform(labels)
        # Save label binarizer
        with open(LB_FILENAME, "wb") as f:
            pickle.dump(label_binarizer, f)

    return np.array(images), np.array(labels), label_binarizer

# It converts the pixel values of your images from integers (0 to 255) to floating-point numbers between 0 and 1.

# because the neural network work better and even faster when the input value are samll and normalized
def preprocess_images(images):
    return images.astype('float32') / 255.0

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='orange')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def predict_on_images(model, image_dir, label_binarizer):
    if not os.path.exists(image_dir):
        print(f"Error: The directory {image_dir} does not exist.")
        return

    image_names = os.listdir(image_dir)
    processed_images = []

    for name in image_names:
        img_path = os.path.join(image_dir, name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            processed_images.append(img)
        else:
            print(f"Skipping invalid image: {img_path}")

    processed_images = preprocess_images(np.array(processed_images))
    predictions = model.predict(processed_images)

    for i, prediction in enumerate(predictions):
        predicted_label = label_binarizer.inverse_transform([prediction])[0]
        print(f"Image: {image_names[i]} → Predicted Label: {predicted_label}")

# Main logic
if __name__ == "__main__":
    # Convert GIFs if needed
    convert_gif_to_png(DATASET_PATH)

    print("🔍 Loading dataset...")
    images, labels, label_binarizer = load_dataset(DATASET_PATH)

    images = preprocess_images(images)

    # derive NUM_CLASSES from the label binarizer (most robust)
    # before the num_classes is not dynamic and hardcoded to 4 as there are only 4 folders in our code now 
    # but now it is dynamic 
    NUM_CLASSES = len(label_binarizer.classes_)
    print(f"Detected classes ({NUM_CLASSES}): {list(label_binarizer.classes_)}")

    if NUM_CLASSES == 0:
        raise RuntimeError(f"No classes found in {DATASET_PATH}")

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    # Load existing model if present, otherwise build a new one. In either
    # case we continue training (this updates the model if it already exists).
    if os.path.exists(MODEL_FILENAME):
        print(f"📥 Loading existing model from '{MODEL_FILENAME}'...")
        model = load_model(MODEL_FILENAME)
    else:
        print("📦 Building a new model...")
        model = build_model((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)

    # Data augmentation / generator (used for both fresh training and continued training)
    data_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    data_gen.fit(train_images)

    history = model.fit(
        data_gen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
        steps_per_epoch=max(1, math.ceil(len(train_images) / BATCH_SIZE)),
        epochs=EPOCHS
    )
    plot_history(history)
    
    print("\n" + "="*60)
    print("💾 SAVING MODEL...")
    print("="*60)
    model.save(MODEL_FILENAME)
    print(f"✅ Model successfully saved to: {MODEL_FILENAME}")
    
    print("\n" + "="*60)
    print("🔍 EVALUATING MODEL ON TEST SET...")
    print("="*60)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"📊 Final Test Results:")
    print(f"   • Test Loss: {test_loss:.4f}")
    print(f"   • Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Training summary
    final_train_acc = history.history['accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    
    print("\n" + "="*60)
    print("🎯 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📈 Training Summary:")
    print(f"   • Total Epochs: {EPOCHS}")
    print(f"   • Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"   • Final Training Loss: {final_train_loss:.4f}")
    print(f"   • Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   • Final Test Loss: {test_loss:.4f}")
    print(f"   • Number of Classes: {NUM_CLASSES}")
    print(f"   • Classes: {list(label_binarizer.classes_)}")
    print(f"   • Training Images: {len(train_images)}")
    print(f"   • Test Images: {len(test_images)}")
    
    print(f"\n📁 Model Files Created/Updated:")
    print(f"   • Model: {MODEL_FILENAME}")
    print(f"   • Label Binarizer: {LB_FILENAME}")
    
    # Performance assessment
    if test_acc >= 0.90:
        print(f"\n🌟 EXCELLENT! Your model achieved {test_acc*100:.2f}% accuracy!")
    elif test_acc >= 0.80:
        print(f"\n👍 GOOD! Your model achieved {test_acc*100:.2f}% accuracy!")
    elif test_acc >= 0.70:
        print(f"\n⚠️  FAIR! Your model achieved {test_acc*100:.2f}% accuracy. Consider more training or data.")
    else:
        print(f"\n⚠️  LOW ACCURACY! Your model achieved only {test_acc*100:.2f}% accuracy. Consider:")
        print("   • Adding more training data")
        print("   • Increasing epochs")
        print("   • Adjusting model architecture")
        print("   • Checking data quality")
    
    print("\n" + "="*60)
    print("🎉 WHEAT DISEASE DETECTION MODEL READY FOR USE!")
    print("="*60)
    print("You can now use this model to predict wheat diseases on new images.")
    print(f"Use the predict.py script or load '{MODEL_FILENAME}' in your code.")
    print("="*60)

    # print("🧪 Predicting on new images...")
    # predict_on_images(model, PREDICT_DIR, label_binarizer)
