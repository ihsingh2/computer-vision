import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.transforms as T
import wandb
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.segmentation import MeanIoU

DATA_DIR = "../data/q2/dataset_256"
CLASS_NAMES = [
    "Unlabeled",
    "Building",
    "Fence",
    "Other",
    "Pedestrian",
    "Pole",
    "Roadline",
    "Road",
    "Sidewalk",
    "Vegetation",
    "Car",
    "Wall",
    "Traffic sign"
]
PALETTE = np.array([
    [0, 0, 0],        #  0 Unlabeled
    [70, 70, 70],     #  1 Building
    [100, 40, 40],    #  2 Fence
    [55, 90, 80],     #  3 Other
    [220, 20, 60],    #  4 Pedestrian
    [153, 153, 153],  #  5 Pole
    [157, 234, 50],   #  6 Roadline
    [128, 64, 128],   #  7 Road
    [244, 35, 232],   #  8 Sidewalk
    [107, 142, 35],   #  9 Vegetation
    [0, 0, 142],      # 10 Car
    [102, 102, 156],  # 11 Wall
    [220, 220, 0]     # 12 Traffic sign
], dtype=np.uint8)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_WORKERS = 8

class SegmentationDataset(Dataset):
    def __init__(self, split):
        assert split in ["train", "test"], "Unrecognized split of dataset"
        self.split = split
        self.images_dir = os.path.join(DATA_DIR, split, "images")
        self.masks_dir = os.path.join(DATA_DIR, split, "labels")
        self.images = sorted(os.listdir(self.images_dir))
        self.masks = sorted(os.listdir(self.masks_dir))
        if split == "train":
            self.transform_img = T.Compose([
                T.ToTensor(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform_img = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        self.transform_mask = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = self.transform_img(image)
        mask = self.transform_mask(mask) * 255
        if self.split == "train" and random.random() < 0.5:
            image = T.functional.hflip(image)
            mask = T.functional.hflip(mask)
        mask = mask.squeeze(0).long()
        return image, mask

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNet(nn.Module):
    def __init__(self, num_classes, base_channels=64):
        super().__init__()
        self.enc1 = conv_block(3, base_channels)
        self.enc2 = conv_block(base_channels, base_channels*2)
        self.enc3 = conv_block(base_channels*2, base_channels*4)
        self.enc4 = conv_block(base_channels*4, base_channels*8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base_channels*8, base_channels*16)

        self.up4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.dec4 = conv_block(base_channels*16, base_channels*8)

        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_channels*8, base_channels*4)

        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_channels*4, base_channels*2)

        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_channels*2, base_channels)

        self.conv_last = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.conv_last(d1)
        return out

class UNetNoSkip(nn.Module):
    def __init__(self, num_classes, base_channels=64):
        super().__init__()
        self.enc1 = conv_block(3, base_channels)
        self.enc2 = conv_block(base_channels, base_channels*2)
        self.enc3 = conv_block(base_channels*2, base_channels*4)
        self.enc4 = conv_block(base_channels*4, base_channels*8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base_channels*8, base_channels*16)

        self.up4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.dec4 = conv_block(base_channels*8, base_channels*8)

        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_channels*4, base_channels*4)

        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_channels*2, base_channels*2)

        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_channels, base_channels)

        self.conv_last = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool(x)
        x = self.enc2(x)
        x = self.pool(x)
        x = self.enc3(x)
        x = self.pool(x)
        x = self.enc4(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        x = self.up4(x)
        x = self.dec4(x)
        x = self.up3(x)
        x = self.dec3(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.up1(x)
        x = self.dec1(x)
        out = self.conv_last(x)
        return out

class ResidualUNet(nn.Module):
    def __init__(self, num_classes, base_channels=64):
        super().__init__()
        self.enc1 = ResidualBlock(3, base_channels)
        self.enc2 = ResidualBlock(base_channels, base_channels*2)
        self.enc3 = ResidualBlock(base_channels*2, base_channels*4)
        self.enc4 = ResidualBlock(base_channels*4, base_channels*8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(base_channels*8, base_channels*16)

        self.up4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(base_channels*16, base_channels*8)

        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_channels*8, base_channels*4)

        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_channels*4, base_channels*2)

        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_channels*2, base_channels)

        self.conv_last = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.conv_last(d1)
        return out

class AttentionUNet(nn.Module):
    def __init__(self, num_classes, base_channels=64):
        super().__init__()
        self.enc1 = conv_block(3, base_channels)
        self.enc2 = conv_block(base_channels, base_channels*2)
        self.enc3 = conv_block(base_channels*2, base_channels*4)
        self.enc4 = conv_block(base_channels*4, base_channels*8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base_channels*8, base_channels*16)

        self.up4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=base_channels*8, F_l=base_channels*8, F_int=base_channels*4)
        self.dec4 = conv_block(base_channels*16, base_channels*8)

        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=base_channels*4, F_l=base_channels*4, F_int=base_channels*2)
        self.dec3 = conv_block(base_channels*8, base_channels*4)

        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=base_channels*2, F_l=base_channels*2, F_int=base_channels)
        self.dec2 = conv_block(base_channels*4, base_channels*2)

        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=base_channels, F_l=base_channels, F_int=base_channels//2)
        self.dec1 = conv_block(base_channels*2, base_channels)

        self.conv_last = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        e4_att = self.att4(e4, d4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        e3_att = self.att3(e3, d3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_att = self.att2(e2, d2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_att = self.att1(e1, d1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)

        out = self.conv_last(d1)
        return out

def train_model(model, train_loader, val_loader, num_epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        miou_metric = MeanIoU(num_classes=len(CLASS_NAMES), input_format='index').to(DEVICE)
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            miou_metric.update(preds, masks)
        train_loss = running_loss / len(train_loader)
        train_miou = miou_metric.compute().item()

        model.eval()
        running_val_loss = 0.0
        miou_metric_val = MeanIoU(num_classes=len(CLASS_NAMES), input_format='index').to(DEVICE)
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                running_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                miou_metric_val.update(preds, masks)
        val_loss = running_val_loss / len(val_loader)
        val_miou = miou_metric_val.compute().item()

        wandb.log({
            "train_loss": train_loss,
            "train_mIoU": train_miou,
            "val_loss": val_loss,
            "val_mIoU": val_miou
        })
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | " \
            f"Train mIoU: {train_miou:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")

def evaluate_model(model, data_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    miou_metric = MeanIoU(num_classes=len(CLASS_NAMES), input_format='index').to(DEVICE)
    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            preds = torch.argmax(outputs, dim=1)
            miou_metric.update(preds, masks)
    test_miou = miou_metric.compute().item()

    wandb.log({
        "test_mIoU": test_miou,
    })
    print(f"Test mIoU: {test_miou:.4f}")

def visualize_prediction(model, test_loader, file_name, num_samples=5):
    model.eval()
    images, masks = next(iter(test_loader))
    images = images.to(DEVICE)
    with torch.no_grad():
        outputs = model(images)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        img = images[i].transpose(1, 2, 0)
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        axes[i][0].imshow(img)
        axes[i][0].set_title("Original Image")
        axes[i][0].axis("off")

        axes[i][1].imshow(PALETTE[masks[i]])
        axes[i][1].set_title("Ground Truth Mask")
        axes[i][1].axis("off")

        axes[i][2].imshow(PALETTE[preds[i]])
        axes[i][2].set_title("Predicted Mask")
        axes[i][2].axis("off")

    plt.tight_layout()
    plt.savefig(f'predictions/{file_name}.png', bbox_inches='tight')
    plt.close()
    plt.clf()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

full_dataset = SegmentationDataset(split="train")
val_size = int(0.25 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
test_dataset = SegmentationDataset(split="test")

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

models = {
    'unet': UNet(num_classes=len(CLASS_NAMES)),
    "unet-noskip": UNetNoSkip(num_classes=len(CLASS_NAMES)),
    "unet-residual": ResidualUNet(num_classes=len(CLASS_NAMES)),
    "unet-gated-attention": AttentionUNet(num_classes=len(CLASS_NAMES))
}

for run_name, model in models.items():
    wandb.init(project="cv-s25-a4-unet", name=run_name, reinit=True)
    model = model.to(DEVICE)
    train_model(model, train_loader, val_loader)
    evaluate_model(model, test_loader)
    visualize_prediction(model, test_loader, run_name)
    torch.save(model.state_dict(), os.path.join('models', run_name + '.pt'))
    model = model.cpu()
    wandb.finish()
