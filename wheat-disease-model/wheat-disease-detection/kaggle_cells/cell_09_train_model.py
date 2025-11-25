# Cell 9: Train Model
"""
Train the model on the wheat disease dataset.
This will take several minutes (or hours depending on dataset size).
The training will use GPU acceleration if available on Kaggle.
"""

print("🚀 Starting model training...")
print("="*70)

# Display training configuration
print(f"📊 Training Configuration:")
print(f"   Epochs:       {EPOCHS}")
print(f"   Batch Size:   {BATCH_SIZE}")
print(f"   Optimizer:    Adam (lr=0.001)")
print(f"   Loss:         Categorical Crossentropy")
print(f"   Metrics:      Accuracy")
print(f"   GPU:          {'✅ Available' if len(tf.config.list_physical_devices('GPU')) > 0 else '❌ Not Available'}")
print()

# Record training start time
import time
start_time = time.time()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Calculate training duration
training_time = time.time() - start_time
hours = int(training_time // 3600)
minutes = int((training_time % 3600) // 60)
seconds = int(training_time % 60)

print("="*70)
print(f"✅ Training completed!")
print(f"⏱️  Training time: {hours}h {minutes}m {seconds}s")
print(f"📈 Best validation loss: {min(history.history['val_loss']):.4f}")
print(f"📈 Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"💾 Best model saved to: {MODEL_PATH}")
