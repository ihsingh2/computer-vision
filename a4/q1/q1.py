import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import wandb
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.segmentation import MeanIoU

DATA_DIR = "../data/q1/dataset_224"
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
BATCH_SIZE = 32
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

class FCN(nn.Module):
    def __init__(self, num_classes, variant, freeze_backbone):
        super().__init__()
        self.num_classes = num_classes
        self.variant = variant
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.score_fr = nn.Conv2d(512, num_classes, kernel_size=1)

        if variant == "32s":
            self.upscore32 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32,
                                                 padding=16, bias=False)
        elif variant == "16s":
            self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2,
                                               padding=1, bias=False)
            self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
            self.upscore16 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16,
                                                padding=8, bias=False)
        elif variant == "8s":
            self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2,
                                              padding=1, bias=False)
            self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
            self.upscore2_2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2,
                                                padding=1, bias=False)
            self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
            self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8,
                                              padding=4, bias=False)
        else:
            raise ValueError("Unknown FCN variant. Use '32s', '16s', or '8s'.")

    def forward(self, x):
        pool3 = pool4 = pool5 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 16:
                pool3 = x
            elif i == 23:
                pool4 = x
            elif i == 30:
                pool5 = x

        score = self.score_fr(pool5)
        if self.variant == "32s":
            out = self.upscore32(score)
        elif self.variant == "16s":
            upscore2 = self.upscore2(score)
            score_pool4 = self.score_pool4(pool4)
            fuse = upscore2 + score_pool4
            out = self.upscore16(fuse)
        elif self.variant == "8s":
            upscore2 = self.upscore2(score)
            score_pool4 = self.score_pool4(pool4)
            fuse_pool4 = upscore2 + score_pool4
            upscore2_2 = self.upscore2_2(fuse_pool4)
            score_pool3 = self.score_pool3(pool3)
            fuse_pool3 = upscore2_2 + score_pool3
            out = self.upscore8(fuse_pool3)
        return out

def visualize_class_masks(num_samples=5):
    masks = os.listdir(os.path.join(DATA_DIR, "train/labels"))
    random.shuffle(masks)
    masks = masks[:num_samples]
    for mask_idx, mask in enumerate(masks):
        original_path = os.path.join(DATA_DIR, "train/images", mask)
        original_img = np.array(Image.open(original_path))
        mask_path = os.path.join(DATA_DIR, "train/labels", mask)
        mask_img = np.array(Image.open(mask_path).convert("L"))
        plt.figure(figsize=(15, 10))
        plt.subplot(3, 5, 1)
        plt.imshow(original_img)
        plt.title("Original Image")
        plt.axis("off")
        for class_id, name in enumerate(CLASS_NAMES):
            binary_mask = (mask_img == class_id).astype(np.uint8)
            plt.subplot(3, 5, class_id + 2)
            plt.imshow(binary_mask, cmap="gray")
            plt.title(name)
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(f'binary_masks/{mask}', bbox_inches='tight')
        plt.close()
        plt.clf()

def train_model(model, train_loader, val_loader, num_epochs=25):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

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

# visualize_class_masks()

full_dataset = SegmentationDataset(split="train")
val_size = int(0.25 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
test_dataset = SegmentationDataset(split="test")

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

for freeze in [True, False]:
    for variant in ["32s", "16s", "8s"]:
        run_name = f"{variant}-frozen" if freeze else variant
        wandb.init(project="cv-s25-a4-fcn", name=run_name, reinit=True)
        model = FCN(num_classes=len(CLASS_NAMES), variant=variant, freeze_backbone=freeze).to(DEVICE)
        train_model(model, train_loader, val_loader)
        evaluate_model(model, test_loader)
        visualize_prediction(model, test_loader, run_name)
        torch.save(model.state_dict(), os.path.join('models', run_name + '.pt'))
        wandb.finish()
