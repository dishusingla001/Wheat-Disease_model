# Cell 11: Visualize Training History
"""
Create plots to visualize the training process:
- Training and validation accuracy over epochs
- Training and validation loss over epochs
These plots help identify overfitting, underfitting, and convergence.
"""

print("📊 Creating training history visualizations...")
print("="*70)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Accuracy
axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Add best accuracy markers
best_train_acc_idx = np.argmax(history.history['accuracy'])
best_val_acc_idx = np.argmax(history.history['val_accuracy'])
axes[0].plot(best_train_acc_idx, history.history['accuracy'][best_train_acc_idx], 
             'ro', markersize=8, label=f'Best Train: {history.history["accuracy"][best_train_acc_idx]:.4f}')
axes[0].plot(best_val_acc_idx, history.history['val_accuracy'][best_val_acc_idx], 
             'go', markersize=8, label=f'Best Val: {history.history["val_accuracy"][best_val_acc_idx]:.4f}')
axes[0].legend(loc='lower right')

# Plot 2: Loss
axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# Add best loss markers
best_train_loss_idx = np.argmin(history.history['loss'])
best_val_loss_idx = np.argmin(history.history['val_loss'])
axes[1].plot(best_train_loss_idx, history.history['loss'][best_train_loss_idx], 
             'ro', markersize=8, label=f'Best Train: {history.history["loss"][best_train_loss_idx]:.4f}')
axes[1].plot(best_val_loss_idx, history.history['val_loss'][best_val_loss_idx], 
             'go', markersize=8, label=f'Best Val: {history.history["val_loss"][best_val_loss_idx]:.4f}')
axes[1].legend(loc='upper right')

# Adjust layout and save
plt.tight_layout()
plt.savefig(HISTORY_PATH, dpi=300, bbox_inches='tight')
print(f"✅ Training history plot saved to: {HISTORY_PATH}")

# Display the plot
plt.show()

# Print summary statistics
print("\n📈 Training Summary:")
print(f"   Epochs completed:        {len(history.history['loss'])}")
print(f"   Best training accuracy:  {max(history.history['accuracy']):.4f} (epoch {best_train_acc_idx+1})")
print(f"   Best validation accuracy: {max(history.history['val_accuracy']):.4f} (epoch {best_val_acc_idx+1})")
print(f"   Best training loss:      {min(history.history['loss']):.4f} (epoch {best_train_loss_idx+1})")
print(f"   Best validation loss:    {min(history.history['val_loss']):.4f} (epoch {best_val_loss_idx+1})")

# Check for overfitting
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
acc_gap = final_train_acc - final_val_acc

print(f"\n🔍 Overfitting Analysis:")
print(f"   Final train-val accuracy gap: {acc_gap:.4f}")
if acc_gap > 0.10:
    print("   ⚠️  Warning: Large gap suggests overfitting")
elif acc_gap < 0.02:
    print("   ✅ Excellent: Model generalizes well")
else:
    print("   ✅ Good: Acceptable generalization")

print("="*70)
print("✅ Visualization completed!")
