import os
import cv2
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------- Step 1: Dataset Prep ----------------
def prepare_dataset(input_folder="raw_images", output_folder="dataset"):
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "masks"), exist_ok=True)

    image_paths = glob(os.path.join(input_folder, "*.jpg")) + glob(os.path.join(input_folder, "*.png"))
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        img = cv2.resize(img, (256, 256))
        cv2.imwrite(f"{output_folder}/images/{i}.jpg", img)

        # Dummy mask (replace later with real annotated masks)
        mask = np.ones((256, 256), dtype=np.uint8) * 255
        cv2.imwrite(f"{output_folder}/masks/{i}.png", mask)

    print(f"✅ Dataset prepared with {len(image_paths)} images.")

# ---------------- Step 2: Dataset Loader ----------------
class WheatDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=True):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=20, p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], 0)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        augmented = self.transform(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]

        mask = (mask > 127).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0)
        return img, mask

# ---------------- Step 3: Model ----------------
def get_model():
    model = smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    return model

# ---------------- Step 4: Train + Validation ----------------
def train_model(model, train_loader, val_loader, device, epochs=10):
    criterion = smp.losses.DiceLoss("binary") + nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    model.to(device)
    best_iou = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        iou_scores, dice_scores = [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = torch.sigmoid(model(imgs))
                preds = (outputs > 0.5).float()

                intersection = (preds * masks).sum()
                union = preds.sum() + masks.sum()
                dice = (2 * intersection) / (union + 1e-7)
                iou = intersection / (union - intersection + 1e-7)

                dice_scores.append(dice.item())
                iou_scores.append(iou.item())

        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)
        scheduler.step(avg_val_loss := (1 - avg_dice))

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Dice={avg_dice:.4f}, IoU={avg_iou:.4f}")

        if avg_iou > best_iou:
            torch.save(model.state_dict(), "unetpp_best.pth")
            best_iou = avg_iou
            print("✅ Best model saved!")

# ---------------- Step 5: Inference (Disease %) ----------------
def predict_image(model, image_path, device):
    model.eval()
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (256, 256))

    transform = A.Compose([A.Normalize(), ToTensorV2()])
    img_tensor = transform(image=img_resized)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = torch.sigmoid(model(img_tensor))
        pred = (output > 0.5).float().cpu().numpy()[0, 0]

    diseased_area = np.sum(pred)
    total_area = pred.shape[0] * pred.shape[1]
    percentage = (diseased_area / total_area) * 100

    print(f"🌾 Diseased Area = {percentage:.2f}%")
    return pred

# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to raw wheat images")
    args = parser.parse_args()

    prepare_dataset(args.data, "dataset")

    dataset = WheatDataset("dataset/images", "dataset/masks", augment=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model()

    train_model(model, train_loader, val_loader, device, epochs=20)

    # Test prediction on 1 image
    model.load_state_dict(torch.load("unetpp_best.pth", map_location=device))
    predict_image(model, glob("dataset/images/*.jpg")[0], device)
