# U-Net Segmentation - Problem & Solution

## 🐛 Problem: All-Black Masks

**Issue:** The U-Net model was producing all-black binary masks (no disease detected) even on diseased images.

## 🔍 Root Cause

**Model Under-confidence:**
- Model predictions range: `0.0 - 0.35` (typically `0.0 - 0.27`)
- Original threshold: `0.5`
- Result: **No pixels exceeded threshold** → all-black masks

## ✅ Solution

**Lowered threshold from 0.5 to 0.15**

### Files Updated:
1. `run_unet.py` - Fixed with lower threshold (0.15)
2. `run_unet_enhanced.py` - Added visualization, statistics, and overlays

### Results After Fix:
```
Image                Disease %    Status
crr_3.jpg             52.98%     Severe 🔴
h_2.jpg               30.99%     Severe 🔴
ls_3.jpg              17.47%     Moderate 🟠
lr_1.jpg              15.84%     Moderate 🟠
h_1.jpg                9.36%     Minor 🟡
...
```

## 📊 Diagnostic Tool

`diagnose_unet.py` - Analyzes:
- Training mask quality
- Model prediction ranges
- Threshold sensitivity
- Per-image statistics

**Usage:**
```powershell
python diagnose_unet.py
```

## 🚀 Usage

### Basic Prediction:
```powershell
python run_unet.py
```
Output: Binary masks saved to `mask_by_unet/`

### Enhanced Prediction (Recommended):
```powershell
python run_unet_enhanced.py
```
Output:
- Binary masks: `mask_by_unet/mask_*.jpg`
- Overlays (red highlights): `mask_by_unet/overlays/overlay_*.jpg`
- Detailed statistics and disease severity classification

## 🎯 Why This Happened

### Common Causes of Under-confident Models:

1. **Class Imbalance:**
   - Most pixels are healthy (black)
   - Few diseased pixels (white)
   - Model learns to predict low values

2. **Insufficient Training:**
   - Only 20 epochs
   - Small dataset
   - No data augmentation

3. **Loss Function:**
   - Binary Crossentropy alone may not penalize false negatives enough
   - Consider: Dice Loss, Focal Loss, or combined loss

4. **Architecture:**
   - Standard U-Net without batch normalization
   - No dropout for regularization

## 🔧 Future Improvements

### 1. Retrain with Better Configuration

**Recommended changes to `unet_model.py`:**

```python
# Increase epochs
EPOCHS = 50  # or more

# Add data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest'
)

# Use combined loss
def dice_loss(y_true, y_pred):
    smooth = 1.
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def combined_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy', dice_coefficient, iou_score])

# Add callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss'),
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
]

history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=8),
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    callbacks=callbacks
)
```

### 2. Add Batch Normalization

```python
from tensorflow.keras.layers import BatchNormalization

# After each Conv2D:
c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
c1 = BatchNormalization()(c1)
```

### 3. Adaptive Thresholding

Instead of fixed threshold, use Otsu's method:

```python
from skimage.filters import threshold_otsu

threshold = threshold_otsu(pred_mask)
binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
```

### 4. Post-processing

```python
# Remove small noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
```

### 5. Confidence Calibration

Train a calibration layer or use temperature scaling to map raw outputs to proper probabilities.

## 📈 Expected Performance After Improvements

| Metric | Current | After Improvements |
|--------|---------|-------------------|
| Prediction Range | 0.0 - 0.35 | 0.0 - 1.0 |
| Optimal Threshold | 0.15 | 0.5 |
| IoU Score | ~0.4 | ~0.7+ |
| Dice Score | ~0.5 | ~0.8+ |

## 🎓 Key Learnings

1. **Always check prediction ranges** before setting thresholds
2. **Use diagnostic tools** to understand model behavior
3. **Binary outputs don't guarantee binary predictions** - model may be under/over-confident
4. **Class imbalance** in segmentation requires special handling
5. **Combined loss functions** often work better than single metrics

## 📝 Quick Reference

### Current Settings:
- **Threshold:** 0.15
- **Input size:** 256x256
- **Model output range:** 0.0 - 0.35

### Severity Classification:
- **Healthy:** < 1% diseased pixels
- **Minor:** 1-10%
- **Moderate:** 10-30%
- **Severe:** > 30%

## 🔗 Related Files

- `unet_model.py` - Training script
- `run_unet.py` - Basic prediction (fixed threshold)
- `run_unet_enhanced.py` - Enhanced prediction with visualization
- `diagnose_unet.py` - Diagnostic tool
- `labelme_to_dataset.py` - Mask generation from annotations

---
*Last Updated: 2025-11-19*
