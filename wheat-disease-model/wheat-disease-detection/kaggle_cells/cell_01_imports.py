# Cell 1: Import Libraries and Setup
"""
Import all necessary libraries for the wheat disease detection model.
Run this cell first to load all dependencies.
"""

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

print("✅ All libraries imported successfully!")
print(f"TensorFlow version: {__import__('tensorflow').__version__}")
