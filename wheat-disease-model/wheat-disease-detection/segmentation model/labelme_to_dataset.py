import os
import json
import numpy as np
import cv2
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

def convert_labelme_to_masks(json_folder, output_masks_folder, output_images_folder=None):
    """
    Convert LabelMe JSON files to binary masks
    
    Args:
        json_folder: Path to folder containing JSON files
        output_masks_folder: Path to save generated masks
        output_images_folder: Path to save original images (optional)
    """
    
    # Create output directories
    os.makedirs(output_masks_folder, exist_ok=True)
    if output_images_folder:
        os.makedirs(output_images_folder, exist_ok=True)
    
    json_files = list(Path(json_folder).glob("*.json"))
    print(f"Found {len(json_files)} JSON files to convert")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get image data
            if 'imageData' in data and data['imageData']:
                # Decode base64 image data
                image_data = base64.b64decode(data['imageData'])
                img = Image.open(BytesIO(image_data))
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            else:
                # Try to load image from imagePath
                img_path = json_file.parent / data['imagePath']
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                else:
                    print(f"Warning: Image not found for {json_file.name}")
                    continue
            
            h, w = img.shape[:2]
            
            # Create binary mask
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Process all shapes (polygons)
            shapes_found = 0
            for shape in data.get('shapes', []):
                if shape['shape_type'] == 'polygon':
                    points = np.array(shape['points'], dtype=np.int32)
                    cv2.fillPoly(mask, [points], 255)  # White for diseased area
                    shapes_found += 1
            
            # Save mask
            mask_filename = json_file.stem + '.png'
            mask_path = Path(output_masks_folder) / mask_filename
            cv2.imwrite(str(mask_path), mask)
            
            # Save original image if output folder specified
            if output_images_folder:
                img_filename = json_file.stem + '.jpg'
                img_path = Path(output_images_folder) / img_filename
                cv2.imwrite(str(img_path), img)
            
            print(f"✅ Converted: {json_file.name} -> {mask_filename} ({shapes_found} shapes)")
            
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {str(e)}")
    
    print(f"\n🎉 Conversion completed! Masks saved to: {output_masks_folder}")

def convert_all_disease_folders():
    """Convert all disease type folders"""
    
    BASE_DIR = os.path.dirname(__file__)
    base_path = os.path.join(BASE_DIR,'Labelme_mask')
    output_base = os.path.join(BASE_DIR,'mask_folder')
    
    # Disease folders
    disease_folders = [
        "Wheat_crown_root_rot",
        "Wheat_leaf_rust", 
        "Wheat_loose_smut",
        "Wheat_healthy"
    ]
    
    all_masks_base = Path(output_base) / "generated_masks"
    all_images_base = Path(output_base) / "generated_images"
    
    total_converted = 0
    
    for disease_folder in disease_folders:
        json_folder = Path(base_path) / disease_folder
        
        if json_folder.exists():
            print(f"\n📁 Processing {disease_folder}...")
            
            # Create disease-specific subfolders for masks and images
            disease_masks_folder = all_masks_base / disease_folder
            disease_images_folder = all_images_base / disease_folder
            
            os.makedirs(disease_masks_folder, exist_ok=True)
            os.makedirs(disease_images_folder, exist_ok=True)
            
            # Convert JSON files from this disease folder
            convert_labelme_to_masks(
                json_folder=str(json_folder),
                output_masks_folder=str(disease_masks_folder),
                output_images_folder=str(disease_images_folder)
            )
            
            # Count converted files
            json_count = len(list(json_folder.glob("*.json")))
            total_converted += json_count
            
        else:
            print(f"⚠️  Folder not found: {json_folder}")
    
    print(f"\n🏁 Total files processed: {total_converted}")
    print(f"📂 Masks saved to: {all_masks_base} (organized by disease subfolders)")
    print(f"📂 Images saved to: {all_images_base} (organized by disease subfolders)")

if __name__ == "__main__":
    print("🚀 Starting LabelMe JSON to Mask conversion...")
    convert_all_disease_folders()