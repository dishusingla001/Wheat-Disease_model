# 🌾 Training Wheat Disease Detection Model on Kaggle

This guide explains how to train your wheat disease detection model on Kaggle's free GPU resources.

## 📋 Prerequisites

1. A Kaggle account (free at [kaggle.com](https://www.kaggle.com))
2. Your dataset organized in the `cropDiseaseDataset/` folder structure

## 🚀 Steps to Train on Kaggle

### Step 1: Upload Your Dataset to Kaggle

1. **Compress your dataset:**
   - Create a zip file of your `cropDiseaseDataset` folder
   - Make sure it contains subfolders: `Wheat_healthy`, `Wheat_Leaf_Rust`, `Wheat_Loose_Smut`, `Wheat_yellow_rust`

2. **Create a new dataset on Kaggle:**
   - Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
   - Click "New Dataset"
   - Upload your zip file
   - Name it something like `wheat-disease-dataset`
   - Set it to **Public** or **Private** (your choice)
   - Click "Create"

### Step 2: Create a New Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click "New Notebook"
3. In the notebook settings (right sidebar):
   - **Accelerator:** Select **GPU** (or **TPU** for faster training)
   - **Internet:** Turn **ON** if you need to install packages
   - **Language:** Python

### Step 3: Add Your Dataset to the Notebook

1. Click "+ Add data" in the right sidebar
2. Search for your dataset (the one you uploaded)
3. Click to add it to your notebook

### Step 4: Copy the Training Code

Copy the entire code from `kaggle_train.py` and paste it into a code cell in your Kaggle notebook.

**Important:** Update the dataset path in the code:
```python
DATASET_PATH = '/kaggle/input/YOUR-DATASET-NAME/cropDiseaseDataset'
```

Replace `YOUR-DATASET-NAME` with your actual dataset name from Kaggle.

### Step 5: Run the Training

1. Click "Run All" or press `Shift + Enter` to run the code
2. Training will start automatically
3. Monitor the progress in the output

## 📊 What Happens During Training

1. **Dataset Loading:** All images are loaded and preprocessed
2. **Data Split:** 70% training, 15% validation, 15% test
3. **Model Training:** CNN trains with data augmentation
4. **Callbacks Active:**
   - Early stopping (stops if no improvement for 10 epochs)
   - Model checkpoint (saves best model)
   - Learning rate reduction (reduces LR when stuck)
5. **Results:** Final accuracy and loss metrics displayed

## 📥 Download Your Trained Model

After training completes, download the model files from the `/kaggle/working/` directory:

1. Click the "Output" tab on the right
2. Download these files:
   - `wheatDiseaseModel.keras` (your trained model)
   - `label_binarizer.pkl` (class labels mapping)
   - `training_history.png` (training visualization)

## 🔧 Key Improvements in Kaggle Version

Compared to your local version, the Kaggle script includes:

1. **Better Model Architecture:**
   - Batch Normalization layers (faster training, better accuracy)
   - Dropout layers (prevents overfitting)
   - Deeper network with 3 conv blocks

2. **Advanced Callbacks:**
   - Early stopping to prevent overtraining
   - Model checkpoint to save best model
   - Learning rate reduction for fine-tuning

3. **Enhanced Data Augmentation:**
   - Vertical flips
   - Zoom
   - Shear transformations
   - More rotation variety

4. **Validation Set:**
   - Separate validation set during training
   - Better monitoring of model performance

5. **More Epochs:**
   - 50 epochs instead of 30 (Kaggle GPUs are free!)

## 📈 Expected Results

With the improved architecture and Kaggle's GPU:
- **Training time:** ~15-30 minutes (depends on dataset size)
- **Expected accuracy:** 85-95% (depends on data quality)
- **GPU memory used:** ~2-4 GB

## 🎯 Tips for Best Results

1. **Use GPU acceleration** - Essential for faster training
2. **Increase epochs** - Try 50-100 for better accuracy
3. **Monitor training plots** - Check for overfitting
4. **Try different architectures** - Experiment with model depth
5. **Balance your dataset** - Ensure similar number of images per class

## 🐛 Common Issues

### Issue: "Dataset path not found"
**Solution:** Update `DATASET_PATH` to match your Kaggle dataset name

### Issue: "Out of memory"
**Solution:** Reduce `BATCH_SIZE` from 32 to 16 or 8

### Issue: "Low accuracy"
**Solutions:**
- Train for more epochs (50-100)
- Check data quality (remove corrupted images)
- Balance classes (similar images per disease)
- Use more data augmentation

## 📝 Using the Model Locally

After downloading the model from Kaggle:

1. Place `wheatDiseaseModel.keras` and `label_binarizer.pkl` in your local project
2. Use your existing `predict.py` script to make predictions
3. Or create a new prediction script:

```python
from tensorflow.keras.models import load_model
import pickle
import cv2
import numpy as np

# Load model and label binarizer
model = load_model('wheatDiseaseModel.keras')
with open('label_binarizer.pkl', 'rb') as f:
    lb = pickle.load(f)

# Predict on new image
img = cv2.imread('test_image.jpg')
img = cv2.resize(img, (64, 64))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
label = lb.inverse_transform(prediction)[0]
print(f"Predicted: {label}")
```

## 🔗 Useful Links

- [Kaggle Notebooks Documentation](https://www.kaggle.com/docs/notebooks)
- [Kaggle Datasets Documentation](https://www.kaggle.com/docs/datasets)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)

## ✅ Checklist

- [ ] Dataset uploaded to Kaggle
- [ ] Notebook created with GPU enabled
- [ ] Dataset added to notebook
- [ ] Code copied and dataset path updated
- [ ] Training started
- [ ] Model downloaded from Kaggle
- [ ] Model tested locally

## 💡 Next Steps

After training your model on Kaggle:

1. **Test locally** with your `predict.py` script
2. **Create a web app** using Flask or Streamlit
3. **Deploy to cloud** (Heroku, AWS, Google Cloud)
4. **Share on Kaggle** - Publish your notebook for others!

---

**Happy Training! 🎉**

For questions or issues, refer to the Kaggle documentation or the project README.
