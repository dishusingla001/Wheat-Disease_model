# Cell 6: Split Dataset
"""
Split dataset into training, validation, and test sets.
- 70% for training (learn patterns)
- 15% for validation (tune during training)
- 15% for testing (final evaluation)
"""

print("✂️  Splitting dataset into train/validation/test sets...")
print("="*70)

# First split: separate test set (15%)
train_val_images, test_images, train_val_labels, test_labels = train_test_split(
    images, labels, 
    test_size=0.15,           # 15% for testing
    random_state=42,          # For reproducibility
    stratify=labels           # Keep class distribution balanced
)

# Second split: separate validation from training (15% of remaining = ~12.75% of total)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_val_images, train_val_labels,
    test_size=0.176,          # ~15% of remaining 85% ≈ 12.75% of total
    random_state=42,
    stratify=train_val_labels
)

# Display split statistics
print(f"📊 Dataset Split:")
print(f"   Training:   {len(train_images):,} images ({len(train_images)/len(images)*100:.1f}%)")
print(f"   Validation: {len(val_images):,} images ({len(val_images)/len(images)*100:.1f}%)")
print(f"   Test:       {len(test_images):,} images ({len(test_images)/len(images)*100:.1f}%)")
print(f"   Total:      {len(images):,} images")

# Show class distribution in each set
print("\n📈 Class Distribution:")
for i, class_name in enumerate(label_binarizer.classes_):
    train_count = np.sum(train_labels[:, i])
    val_count = np.sum(val_labels[:, i])
    test_count = np.sum(test_labels[:, i])
    print(f"   {class_name:20s}: Train={train_count:4.0f}, Val={val_count:4.0f}, Test={test_count:4.0f}")

print("="*70)
print("✅ Dataset split completed!")
