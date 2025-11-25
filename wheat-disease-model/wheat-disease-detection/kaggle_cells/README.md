# Kaggle Notebook Cells - README

This directory contains the complete wheat disease classification training pipeline broken down into **12 separate cells** for easy understanding and execution in Kaggle notebooks.

## 📋 Cell Execution Order

Run these cells **in order** in your Kaggle notebook:

### 1. **cell_01_imports.py** - Import Libraries
- Loads all required Python libraries (TensorFlow, OpenCV, scikit-learn, etc.)
- Checks TensorFlow version
- Verifies GPU availability

### 2. **cell_02_config.py** - Configuration
- Defines dataset paths (⚠️ **MUST UPDATE** the `DATASET_PATH`)
- Sets hyperparameters (image size, batch size, epochs)
- Specifies output file names

### 3. **cell_03_helpers.py** - Helper Functions
- `convert_gif_to_png()`: Converts GIF images to PNG format
- `load_dataset()`: Loads images from dataset folders
- `preprocess_images()`: Normalizes images and prepares labels

### 4. **cell_04_model.py** - Model Architecture
- `build_improved_model()`: Defines CNN architecture
- 3 convolutional blocks with batch normalization
- Dropout layers for regularization
- Dense layers for classification

### 5. **cell_05_load_data.py** - Load Dataset
- Executes data loading functions
- Displays dataset statistics
- Validates dataset structure

### 6. **cell_06_split_data.py** - Split Dataset
- Splits data into train (70%), validation (15%), test (15%)
- Uses stratified splitting to maintain class balance
- Shows class distribution across splits

### 7. **cell_07_build_model.py** - Build Model
- Instantiates the CNN model
- Displays model architecture summary
- Shows parameter counts

### 8. **cell_08_setup_training.py** - Training Configuration
- Configures callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
- Sets up data augmentation (rotation, shifts, flips, zoom)
- Creates training and validation generators

### 9. **cell_09_train_model.py** - Train Model
- Trains the model on the dataset
- Uses GPU acceleration if available
- Displays training progress and time

### 10. **cell_10_evaluate.py** - Evaluate Model
- Tests model on unseen test set
- Generates classification report
- Shows confusion matrix and per-class accuracy

### 11. **cell_11_visualize.py** - Visualize Results
- Plots training/validation accuracy curves
- Plots training/validation loss curves
- Analyzes overfitting
- Saves visualization to file

### 12. **cell_12_summary.py** - Summary
- Displays final results
- Provides download instructions
- Shows example prediction code
- Recommends next steps

---

## 🚀 How to Use in Kaggle

### Step 1: Create New Notebook
1. Go to [Kaggle](https://www.kaggle.com)
2. Click **"Code"** → **"New Notebook"**
3. Enable GPU: **Settings** → **Accelerator** → **GPU T4 x2**

### Step 2: Upload Dataset
1. Click **"Add Data"** → **"Upload Dataset"**
2. Upload your `cropDiseaseDataset` folder
3. Note the dataset path (usually `/kaggle/input/your-dataset-name/`)

### Step 3: Copy Cell Contents
1. Create a new code cell in Kaggle
2. Open `cell_01_imports.py`
3. Copy **all contents** and paste into the Kaggle cell
4. Click **"Run"** or press **Shift+Enter**
5. Repeat for cells 2-12 in order

### Step 4: Update Configuration
- In **Cell 2** (`cell_02_config.py`), update this line:
  ```python
  DATASET_PATH = "/kaggle/input/YOUR-DATASET-NAME/cropDiseaseDataset"
  ```
  Replace `YOUR-DATASET-NAME` with your actual dataset name.

### Step 5: Run All Cells
- Run each cell sequentially from 1 to 12
- Wait for each cell to complete before running the next
- Monitor progress through printed messages

### Step 6: Download Results
- After Cell 12 completes, go to **Output** tab
- Download:
  - `wheatDiseaseModel.keras` (trained model)
  - `label_binarizer.pkl` (label encoder)
  - `training_history.png` (visualization)

---

## 📊 Expected Outputs

### After Cell 5 (Load Data):
```
✅ Dataset loaded: 3,250 images, 4 classes
```

### After Cell 9 (Training):
```
✅ Training completed!
⏱️  Training time: 0h 25m 43s
📈 Best validation accuracy: 0.9234
```

### After Cell 10 (Evaluation):
```
📈 Test Results:
   Test Accuracy: 0.9180 (91.80%)
```

---

## 🎯 Why This Cell-by-Cell Approach?

✅ **Educational**: Understand each step of the ML pipeline  
✅ **Debuggable**: Isolate and fix issues easily  
✅ **Flexible**: Modify individual components without affecting others  
✅ **Transparent**: See exactly what happens at each stage  

---

## ⚙️ System Requirements

- **Python**: 3.7+
- **TensorFlow**: 2.x
- **RAM**: 16GB+ recommended
- **GPU**: Recommended for faster training
- **Storage**: 2GB+ for dataset and model

---

## 📚 Additional Resources

- **Full Script**: See `../kaggle_train.py` for complete single-file version
- **Guide**: See `../KAGGLE_TRAINING_GUIDE.md` for detailed documentation
- **Original Script**: See `../wheatDisDet.py` for the original local training code

---

## 🐛 Troubleshooting

### "No module named 'X'" Error
Run Cell 1 again to ensure all imports succeed.

### "Dataset path not found" Error
Update `DATASET_PATH` in Cell 2 with your correct Kaggle dataset path.

### Low Accuracy (<70%)
- Ensure dataset has balanced classes
- Increase training epochs (edit `EPOCHS` in Cell 2)
- Check image quality and variety

### Out of Memory Error
- Reduce `BATCH_SIZE` in Cell 2 (try 16 or 8)
- Reduce `IMG_SIZE` in Cell 2 (try 32x32)

### Training Too Slow
- Verify GPU is enabled: Runtime → Change runtime type → GPU
- Reduce `EPOCHS` for testing (increase later for production)

---

## 📞 Support

If you encounter issues:
1. Check the `KAGGLE_TRAINING_GUIDE.md` for detailed troubleshooting
2. Verify all cells run in sequence (don't skip cells)
3. Ensure dataset path is correctly configured in Cell 2
4. Check that GPU is enabled in Kaggle notebook settings

---

**Happy Training! 🌾🚀**
