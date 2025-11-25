import os
import shutil

# Path to Wheat_healthy folder
BASE_DIR = os.path.dirname(__file__)
HEALTHY_DIR = os.path.join(BASE_DIR, 'cropDiseaseDataset', 'Wheat_healthy')

# Valid image extensions
VALID_EXTS = ('.png', '.jpg', '.jpeg', '.jfif', '.bmp')

print("=" * 70)
print("🔄 Renaming Wheat_healthy images to consistent format")
print("=" * 70)

if not os.path.exists(HEALTHY_DIR):
    print(f"❌ Directory not found: {HEALTHY_DIR}")
    exit(1)

# Get all image files
image_files = []
for f in os.listdir(HEALTHY_DIR):
    if f.lower().endswith(VALID_EXTS):
        image_files.append(f)

if not image_files:
    print("❌ No image files found in directory")
    exit(1)

# Sort files to ensure consistent ordering
image_files.sort()

print(f"\n📁 Found {len(image_files)} images in {HEALTHY_DIR}")
print(f"📝 Will rename to format: healthy_00001.jpg, healthy_00002.jpg, etc.\n")

# Ask for confirmation
response = input("Proceed with renaming? (yes/no): ").strip().lower()
if response not in ['yes', 'y']:
    print("❌ Operation cancelled")
    exit(0)

# Create a backup list (optional - just for logging)
rename_log = []

# Rename files
success_count = 0
error_count = 0

# Use a temporary naming scheme first to avoid conflicts
temp_files = []
for idx, old_name in enumerate(image_files, start=1):
    old_path = os.path.join(HEALTHY_DIR, old_name)
    
    # Get extension (keep original if jpg/jpeg, convert jfif to jpg)
    _, ext = os.path.splitext(old_name)
    ext = ext.lower()
    if ext in ['.jfif', '.jpeg']:
        ext = '.jpg'
    
    # Temporary name to avoid conflicts during renaming
    temp_name = f"_temp_{idx:05d}{ext}"
    temp_path = os.path.join(HEALTHY_DIR, temp_name)
    
    try:
        os.rename(old_path, temp_path)
        temp_files.append((temp_name, idx, ext))
        rename_log.append(f"{old_name} → temp_{idx:05d}{ext}")
    except Exception as e:
        print(f"⚠️  Error renaming {old_name}: {e}")
        error_count += 1

# Now rename from temp to final format
for temp_name, idx, ext in temp_files:
    temp_path = os.path.join(HEALTHY_DIR, temp_name)
    new_name = f"healthy_{idx:05d}{ext}"
    new_path = os.path.join(HEALTHY_DIR, new_name)
    
    try:
        os.rename(temp_path, new_path)
        success_count += 1
        if success_count <= 10 or success_count % 50 == 0:
            print(f"✅ {temp_name.replace('_temp_', '')} → {new_name}")
    except Exception as e:
        print(f"⚠️  Error in final rename {temp_name}: {e}")
        error_count += 1

print("\n" + "=" * 70)
print(f"✅ Renaming complete!")
print(f"   • Successfully renamed: {success_count} images")
if error_count > 0:
    print(f"   • Errors: {error_count}")
print(f"   • New format: healthy_00001{ext} to healthy_{success_count:05d}{ext}")
print("=" * 70)

# Save rename log
log_path = os.path.join(BASE_DIR, 'rename_log.txt')
with open(log_path, 'w', encoding='utf-8') as f:
    f.write(f"Renamed {success_count} images in Wheat_healthy folder\n")
    f.write("=" * 70 + "\n\n")
    for line in rename_log[:100]:  # Save first 100 entries
        f.write(line + "\n")
    if len(rename_log) > 100:
        f.write(f"\n... and {len(rename_log) - 100} more files\n")

print(f"\n📝 Rename log saved to: {log_path}")
