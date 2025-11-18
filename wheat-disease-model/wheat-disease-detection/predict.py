import os
import numpy as np
import cv2
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
from fertilizer_helper import load_fertilizer_data, get_fertilizer_info
import time
import mimetypes
import sys

# Constants
BASE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)
PREDICT_DIR = os.path.join(BASE_DIR,'testCDD')
IMG_SIZE = 64
MODEL_FILENAME = os.path.join(PARENT_DIR,'wheatDiseaseModel.keras')
LABEL_BINARIZER_FILE = os.path.join(PARENT_DIR,'label_binarizer.pkl')

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.jfif'}

# Validation functions
def validate_file_exists(filepath, file_type="file"):
    """Validate if a file or directory exists with detailed error messages"""
    if not os.path.exists(filepath):
        print(f"❌ ERROR: {file_type.title()} not found at: {filepath}")
        print(f"   Please ensure the {file_type} exists and the path is correct.")
        return False
    return True

def validate_image_format(filepath):
    """Validate if the file is a supported image format"""
    file_ext = os.path.splitext(filepath)[1].lower()
    if file_ext not in SUPPORTED_FORMATS:
        print(f"⚠️  WARNING: Unsupported format '{file_ext}' for file: {os.path.basename(filepath)}")
        print(f"   Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return False
    return True

def validate_image_integrity(image_path):
    """Validate if the image can be read and is not corrupted"""
    try:
        # Try reading with cv2
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ ERROR: Corrupted or unreadable image: {os.path.basename(image_path)}")
            return False, None
            
        # Check if image has valid dimensions
        if img.shape[0] == 0 or img.shape[1] == 0:
            print(f"❌ ERROR: Invalid image dimensions (0x0): {os.path.basename(image_path)}")
            return False, None
            
        # Check if image is too small (minimum 10x10)
        if img.shape[0] < 10 or img.shape[1] < 10:
            print(f"⚠️  WARNING: Image too small ({img.shape[1]}x{img.shape[0]}): {os.path.basename(image_path)}")
            print(f"   Minimum recommended size: 10x10 pixels")
            
        return True, img
        
    except Exception as e:
        print(f"❌ ERROR: Failed to read image '{os.path.basename(image_path)}': {str(e)}")
        return False, None

def validate_directory(dir_path):
    """Validate directory and check for valid images"""
    if not validate_file_exists(dir_path, "directory"):
        return False
        
    # Check if directory is empty
    files = os.listdir(dir_path)
    if not files:
        print(f"❌ ERROR: Directory is empty: {dir_path}")
        return False
        
    # Check if directory contains any supported image files
    valid_images = [f for f in files if validate_image_format(os.path.join(dir_path, f))]
    if not valid_images:
        print(f"❌ ERROR: No supported image files found in: {dir_path}")
        print(f"   Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return False
        
    print(f"✅ Found {len(valid_images)} valid image files in directory")
    return True

# Helper functions
def load_trained_model(model_filename):
    """Load trained model with comprehensive error handling"""
    try:
        # Validate file exists
        if not validate_file_exists(model_filename, "model file"):
            return None
            
        # Check file size
        file_size = os.path.getsize(model_filename)
        if file_size == 0:
            print(f"❌ ERROR: Model file is empty (0 bytes): {model_filename}")
            return None
        elif file_size < 1024:  # Less than 1KB is suspicious for a model
            print(f"⚠️  WARNING: Model file is very small ({file_size} bytes), might be corrupted")
            
        print(f"📦 Loading model from: {model_filename}")
        print(f"📊 Model file size: {file_size / (1024*1024):.2f} MB")
        
        model = load_model(model_filename)
        
        # Validate model structure
        if model is None:
            print(f"❌ ERROR: Failed to load model (returned None)")
            return None
            
        # Check if model has layers
        if len(model.layers) == 0:
            print(f"❌ ERROR: Model has no layers")
            return None
            
        print(f"✅ Model loaded successfully")
        print(f"🏗️  Model architecture: {len(model.layers)} layers")
        print(f"📥 Input shape: {model.input_shape}")
        print(f"📤 Output shape: {model.output_shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load model from {model_filename}")
        print(f"   Error details: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        if "tensorflow" in str(e).lower():
            print(f"   💡 This might be a TensorFlow compatibility issue")
        elif "keras" in str(e).lower():
            print(f"   💡 This might be a Keras version compatibility issue")
        return None

def preprocess_images(image_dir):
    """Preprocess images with comprehensive validation and error handling"""
    if not validate_directory(image_dir):
        return np.array([]), []
    
    images = []
    image_names = []
    skipped_files = []
    processed_count = 0
    
    print(f"\n🔍 Processing images from: {image_dir}")
    
    try:
        files = os.listdir(image_dir)
        print(f"📁 Found {len(files)} files in directory")
        
        for file_name in files:
            file_path = os.path.join(image_dir, file_name)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
                
            # Validate image format
            if not validate_image_format(file_path):
                skipped_files.append((file_name, "Unsupported format"))
                continue
                
            # Validate image integrity and read
            is_valid, img = validate_image_integrity(file_path)
            if not is_valid:
                skipped_files.append((file_name, "Corrupted or unreadable"))
                continue
                
            try:
                # Resize image
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                
                # Validate resized image
                if img_resized.shape != (IMG_SIZE, IMG_SIZE, 3):
                    print(f"⚠️  WARNING: Unexpected image shape after resize: {img_resized.shape}")
                    
                images.append(img_resized)
                image_names.append(file_name)
                processed_count += 1
                
            except Exception as e:
                print(f"❌ ERROR: Failed to process '{file_name}': {str(e)}")
                skipped_files.append((file_name, f"Processing error: {str(e)}"))
                continue
                
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to read directory {image_dir}: {str(e)}")
        return np.array([]), []
    
    # Summary report
    print(f"\n📊 PREPROCESSING SUMMARY:")
    print(f"   ✅ Successfully processed: {processed_count} images")
    print(f"   ❌ Skipped files: {len(skipped_files)}")
    
    if skipped_files:
        print(f"\n⚠️  SKIPPED FILES DETAILS:")
        for file_name, reason in skipped_files:
            print(f"   • {file_name}: {reason}")
    
    if processed_count == 0:
        print(f"❌ ERROR: No images were successfully processed")
        return np.array([]), []
        
    print(f"\n✅ Ready to process {processed_count} images")
    return np.array(images), image_names

def display_prediction_results(prediction_probs, class_names, predicted_label, confidence, image_name):
    """Display detailed prediction results with confidence scores for all classes"""
    print(f"\n{'='*60}")
    print(f"🖼️  IMAGE: {image_name}")
    print(f"{'='*60}")
    print(f"🎯 PREDICTED DISEASE: {predicted_label}")
    print(f"🔥 CONFIDENCE: {confidence:.2f}%")
    print(f"\n📊 PREDICTION PROBABILITIES FOR ALL CLASSES:")
    print(f"{'-'*60}")
    
    # Sort classes by probability (highest first)
    class_probs = list(zip(class_names, prediction_probs))
    class_probs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (class_name, prob) in enumerate(class_probs):
        percentage = prob * 100
        # Create a visual bar for probability
        bar_length = int(percentage / 5)  # Scale bar to max 20 characters
        bar = '█' * bar_length + '░' * (20 - bar_length)
        
        # Add emoji indicators
        if i == 0:  # Highest probability (predicted class)
            emoji = "🥇"
        elif percentage > 20:
            emoji = "⚠️"
        elif percentage > 10:
            emoji = "💡"
        else:
            emoji = "📉"
            
        print(f"{emoji} {class_name:<25} {percentage:6.2f}% [{bar}]")
    
    return predicted_label, confidence

def predict_on_images(model, image_dir, label_binarizer, fertilizer_data):
    images, image_names = preprocess_images(image_dir)
    if len(images) == 0:
        print("⚠️ No valid images found in the directory.")
        return
        
    images = images.astype('float32') / 255.0  # Normalize images
    
    print(f"\n🔄 Processing {len(images)} images...")
    start_time = time.time()
    
    predictions = model.predict(images)
    class_names = label_binarizer.classes_
    
    prediction_summary = []
    
    for i, prediction in enumerate(predictions):
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(prediction)
        predicted_label = class_names[predicted_class_idx]
        confidence = np.max(prediction) * 100
        
        # Display detailed results
        predicted_label, confidence = display_prediction_results(
            prediction, class_names, predicted_label, confidence, image_names[i]
        )
        
        # Store for summary
        prediction_summary.append({
            'image': image_names[i],
            'disease': predicted_label,
            'confidence': confidence
        })

        # Get fertilizer and dosage info with confidence-based recommendations
        fertilizer_info = get_fertilizer_info(predicted_label, fertilizer_data)
        print(f"\n💊 TREATMENT RECOMMENDATION:")
        print(f"{'-'*30}")
        
        if fertilizer_info:
            print(f"🌱 Fertilizer: {fertilizer_info['fertilizer']}")
            print(f"💊 Dosage: {fertilizer_info['dosage']}")
            
            # Confidence-based recommendation reliability
            if confidence >= 80:
                print(f"✅ High Confidence - Apply treatment as recommended")
            elif confidence >= 60:
                print(f"⚠️  Medium Confidence - Consider consulting expert")
            else:
                print(f"❌ Low Confidence - Further inspection recommended")
        else:
            print("⚠️ No treatment info found for this disease.")
            
        print(f"{'='*60}")
    
    # Display prediction summary
    processing_time = time.time() - start_time
    print(f"\n🏁 PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"📊 Total Images Processed: {len(prediction_summary)}")
    print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
    print(f"⚡ Average Time per Image: {processing_time/len(prediction_summary):.2f} seconds")
    
    # Count diseases
    disease_counts = {}
    high_confidence_count = 0
    
    for pred in prediction_summary:
        disease = pred['disease']
        confidence = pred['confidence']
        
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
        if confidence >= 80:
            high_confidence_count += 1
    
    print(f"\n📈 DISEASE DISTRIBUTION:")
    for disease, count in disease_counts.items():
        percentage = (count / len(prediction_summary)) * 100
        print(f"   • {disease}: {count} images ({percentage:.1f}%)")
    
    print(f"\n🎯 HIGH CONFIDENCE PREDICTIONS (≥80%): {high_confidence_count}/{len(prediction_summary)} ({(high_confidence_count/len(prediction_summary)*100):.1f}%)")
    print(f"{'='*60}")

# Main logic
if __name__ == "__main__":
    print("🚀 Starting Wheat Disease Prediction System")
    print("="*60)
    
    # Validate all required files before starting
    print("🔍 Validating system requirements...")
    
    # Check model file
    if not validate_file_exists(MODEL_FILENAME, "model file"):
        print("💡 Suggestion: Train the model first using wheatDisDet.py")
        sys.exit(1)
        
    # Check label binarizer file
    if not validate_file_exists(LABEL_BINARIZER_FILE, "label binarizer file"):
        print("💡 Suggestion: Train the model first using wheatDisDet.py to generate label binarizer")
        sys.exit(1)
        
    # Check prediction directory
    if not validate_directory(PREDICT_DIR):
        print(f"💡 Suggestion: Add image files to {PREDICT_DIR} directory")
        sys.exit(1)
    
    print("✅ All system requirements validated successfully\n")
    
    # Load model with enhanced error handling
    print("📦 Loading trained model...")
    model = load_trained_model(MODEL_FILENAME)
    if model is None:
        print("🛑 Cannot proceed without a valid model. Exiting...")
        sys.exit(1)

    # Load label binarizer with enhanced error handling
    print("\n🏷️  Loading label binarizer...")
    try:
        if not os.path.exists(LABEL_BINARIZER_FILE):
            print(f"❌ ERROR: Label binarizer file not found: {LABEL_BINARIZER_FILE}")
            sys.exit(1)
            
        with open(LABEL_BINARIZER_FILE, "rb") as f:
            label_binarizer = pickle.load(f)
            
        # Validate label binarizer
        if not hasattr(label_binarizer, 'classes_'):
            print(f"❌ ERROR: Invalid label binarizer - missing classes attribute")
            sys.exit(1)
            
        if len(label_binarizer.classes_) == 0:
            print(f"❌ ERROR: Label binarizer has no classes")
            sys.exit(1)
            
        print(f"✅ Label binarizer loaded successfully")
        print(f"📝 Available classes: {list(label_binarizer.classes_)}")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load label binarizer: {str(e)}")
        print(f"   File: {LABEL_BINARIZER_FILE}")
        sys.exit(1)

    # Load fertilizer data with error handling
    print("\n🌱 Loading fertilizer recommendations...")
    try:
        fertilizer_data = load_fertilizer_data()
        if not fertilizer_data:
            print("⚠️  WARNING: No fertilizer data available")
        else:
            print(f"✅ Fertilizer data loaded for {len(fertilizer_data)} disease types")
    except Exception as e:
        print(f"⚠️  WARNING: Failed to load fertilizer data: {str(e)}")
        fertilizer_data = {}

    # Start prediction process
    print(f"\n🎯 Starting prediction process...")
    try:
        predict_on_images(model, PREDICT_DIR, label_binarizer, fertilizer_data)
    except Exception as e:
        print(f"❌ CRITICAL ERROR during prediction: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        sys.exit(1)
        
    print("\n🎉 Prediction process completed successfully!")
