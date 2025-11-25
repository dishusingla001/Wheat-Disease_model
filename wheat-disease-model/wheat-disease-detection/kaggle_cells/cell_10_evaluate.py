# Cell 10: Evaluate Model
"""
Evaluate the trained model on the test set.
The test set contains images the model has never seen during training.
This gives us an unbiased estimate of model performance.
"""

print("📊 Evaluating model on test set...")
print("="*70)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)

print(f"📈 Test Results:")
print(f"   Test Loss:     {test_loss:.4f}")
print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Make predictions on test set
print("\n🔮 Making predictions on test set...")
test_predictions = model.predict(test_images, verbose=0)
test_pred_classes = np.argmax(test_predictions, axis=1)
test_true_classes = np.argmax(test_labels, axis=1)

# Calculate classification report
from sklearn.metrics import classification_report, confusion_matrix

print("\n📋 Classification Report:")
print(classification_report(
    test_true_classes, 
    test_pred_classes, 
    target_names=label_binarizer.classes_
))

# Display confusion matrix
print("📊 Confusion Matrix:")
cm = confusion_matrix(test_true_classes, test_pred_classes)
print(cm)
print()
print("   Rows: True labels")
print("   Columns: Predicted labels")
print(f"   Classes: {', '.join(label_binarizer.classes_)}")

# Calculate per-class accuracy
print("\n📈 Per-Class Accuracy:")
for i, class_name in enumerate(label_binarizer.classes_):
    class_mask = (test_true_classes == i)
    if np.sum(class_mask) > 0:
        class_accuracy = np.mean(test_pred_classes[class_mask] == test_true_classes[class_mask])
        print(f"   {class_name:20s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")

print("="*70)
print("✅ Evaluation completed!")
