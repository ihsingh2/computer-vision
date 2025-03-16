import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import wandb
from PIL import Image
from torchvision import transforms
from torchvision.models.resnet import resnet34, ResNet34_Weights
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from tqdm import tqdm

class FruitDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, train=False):
        self.images = images
        self.masks = masks
        self.train = True
        if train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image_np = np.array(image)
        mask_np = np.array(mask)
        boxes = masks_to_boxes(mask_np)
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.ones((len(boxes),), dtype=torch.int64)
        image = self.transform(image)
        if self.train and random.random() < 0.5:
            image = torchvision.transforms.functional.hflip(image)
            target["boxes"] = self.hflip_box(target["boxes"], image.shape[-1])
        return image, target

    def hflip_box(self, boxes, image_width):
        x1 = boxes[:, 0].detach().clone()
        x2 = boxes[:, 2].detach().clone()
        boxes[:, 0] = image_width - x2
        boxes[:, 2] = image_width - x1
        return boxes

class ResNet34FPNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = IntermediateLayerGetter(resnet34(weights=ResNet34_Weights.DEFAULT), return_layers={"layer3": "0"})
        self.fpn = FeaturePyramidNetwork([256], 256)
        self.out_channels = 256
    def forward(self, x):
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        return fpn_features

def masks_to_boxes(mask):
    """ Convert a grayscale segmentation mask to bounding boxes. """

    boxes = []
    intensities = np.unique(mask)

    for intensity in intensities[1:]:
        pos = np.where(mask == intensity)
        xmin = np.min(pos[1])
        ymin = np.min(pos[0])
        xmax = np.max(pos[1])
        ymax = np.max(pos[0])
        if xmax - xmin >= 1 and ymax - ymin >= 1:
            boxes.append([xmin, ymin, xmax, ymax])

    return np.array(boxes)

def visualize_bounding_box_original_image(image_paths, mask_paths, n=5):
    """ Visualize the bounding box overlaid on the original image for n starting images. """

    fig, axes = plt.subplots(1, 5, figsize=(n * 3, 5))
    fig.suptitle('Bounding Box on Original Image')

    for i in range(n):
        image = np.array(Image.open(image_paths[i]).convert('RGB'))
        mask = np.array(Image.open(mask_paths[i]).convert('L'))
        boxes = masks_to_boxes(mask)
        axes[i].imshow(image)
        for box in boxes:
            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                 fill=False, edgecolor='red', linewidth=1)
            axes[i].add_patch(rect)
        axes[i].set_title(f'Image {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('visualization/bounding_box_original_image.png', bbox_inches='tight')
    plt.close()
    plt.clf()

def visualize_bounding_box_masked_image(image_paths, mask_paths, n=5):
    """ Visualize the bounding box overlaid on the masked image for n starting images. """

    fig, axes = plt.subplots(1, 5, figsize=(n * 3, 5))
    fig.suptitle('Bounding Box on Masked Image')

    for i in range(n):
        image = np.array(Image.open(image_paths[i]).convert('RGB'))
        mask = np.array(Image.open(mask_paths[i]).convert('L'))
        boxes = masks_to_boxes(mask)
        axes[i].imshow(mask, cmap='gray')
        for box in boxes:
            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                 fill=False, edgecolor='green', linewidth=1)
            axes[i].add_patch(rect)
        axes[i].set_title(f'Image {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('visualization/bounding_box_masked_image.png', bbox_inches='tight')
    plt.close()
    plt.clf()

def visualize_bounding_box_mask(image_paths, mask_paths, n=5):
    """ Visualize the bounding box and mask for n starting images. """

    fig, axes = plt.subplots(5, 2, figsize=(8, n * 9))
    fig.suptitle('Comparison of Bounding Box and Mask')

    for i in range(n):
        image = np.array(Image.open(image_paths[i]).convert('RGB'))
        mask = np.array(Image.open(mask_paths[i]).convert('L'))
        boxes = masks_to_boxes(mask)
        boxes_image = np.zeros_like(image)
        for box in boxes:
            cv2.rectangle(boxes_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        axes[i, 0].imshow(boxes_image, cmap='gray')
        axes[i, 0].set_title(f'Bounding Box: Image {i + 1}')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'Mask: Image {i + 1}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('visualization/bounding_box_mask_comparison.png', bbox_inches='tight')
    plt.close()
    plt.clf()

def compute_iou(box1, box2):
    """ Compute Intersection over Union (IoU) of two boxes. """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def evaluate_model(model, dataloader, device, iou_thresholds=[0.5, 0.75]):
    """ Evaluate the AP, precision-recall curve, false positives and negatives. """

    model.eval()

    idx = 0
    all_gt = {}
    all_predictions = []
    with torch.no_grad():
        for image, target in tqdm(dataloader):
            image = list(img.to(device) for img in image)
            outputs = model(image)[0]
            boxes_pred = outputs['boxes'].cpu().numpy()
            scores = outputs['scores'].cpu().numpy()
            gt_boxes = target[0]['boxes'].cpu().numpy()
            all_gt[idx] = {'boxes': gt_boxes, 'detected': [False] * len(gt_boxes)}
            for box, score in zip(boxes_pred, scores):
                all_predictions.append({'image_id': idx, 'score': score, 'box': box})
            idx += 1

    metrics = {}
    all_predictions.sort(key=lambda x: x['score'], reverse=True)
    for iou_thresh in iou_thresholds:
        TP = np.zeros(len(all_predictions))
        FP = np.zeros(len(all_predictions))
        total_gt = sum([len(v['boxes']) for v in all_gt.values()])
        for key in all_gt.keys():
            all_gt[key]['detected'] = [False] * len(all_gt[key]['boxes'])
        for i, pred in enumerate(all_predictions):
            image_id = pred['image_id']
            pred_box = pred['box']
            gt_info = all_gt[image_id]
            gt_boxes = gt_info['boxes']
            best_iou = 0
            best_idx = -1
            for j, gt_box in enumerate(gt_boxes):
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou >= iou_thresh:
                if not gt_info['detected'][best_idx]:
                    TP[i] = 1
                    gt_info['detected'][best_idx] = True
                else:
                    FP[i] = 1
            else:
                FP[i] = 1
        cum_TP = np.cumsum(TP)
        cum_FP = np.cumsum(FP)
        precisions = cum_TP / (cum_TP + cum_FP + 1e-6)
        recalls = cum_TP / (total_gt + 1e-6)
        ap = np.trapezoid(precisions, recalls)
        metrics[iou_thresh] = {
            'mAP': ap,
            'precision': precisions,
            'recall': recalls,
            'FP': int(cum_FP[-1]) if len(cum_FP) > 0 else 0,
            'FN': int(total_gt - cum_TP[-1]) if len(cum_TP) > 0 else total_gt
        }

    return metrics

wandb_log = False
run_name = ''
model_name = ''

# Log
if wandb_log:
    wandb.init(project='cv-s25-a3-fruit-detection', name=run_name, reinit=True)

# Seed
seed = 1122
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Directories
train_images_dir = "../data/Q2/detection/train/images"
train_masks_dir = "../data/Q2/detection/train/masks"
test_images_dir = "../data/Q2/detection/test/images"

# Image Paths
train_image_paths = sorted(glob.glob(os.path.join(train_images_dir, "*.png")))
train_mask_paths = sorted(glob.glob(os.path.join(train_masks_dir, "*.png")))

# # Visualization
# visualize_bounding_box_original_image(train_image_paths, train_mask_paths, n=5)
# visualize_bounding_box_masked_image(train_image_paths, train_mask_paths, n=5)
# visualize_bounding_box_mask(train_image_paths, train_mask_paths, n=5)

# Image Paths
train_image_paths = np.array(sorted(glob.glob(os.path.join(train_images_dir, "*.png"))), dtype=object)
train_mask_paths = np.array(sorted(glob.glob(os.path.join(train_masks_dir, "*.png"))), dtype=object)

# Train Validation Split
indices = np.arange(len(train_image_paths))
np.random.shuffle(indices)
train_end = int(0.8*len(train_image_paths))
train_indices, val_indices = indices[:train_end], indices[train_end:]
val_image_paths, val_mask_paths = list(train_image_paths[val_indices]), list(train_mask_paths[val_indices])
train_image_paths, train_mask_paths = list(train_image_paths[train_indices]), list(train_mask_paths[train_indices])

# Dataset, Dataloader
train_dataset = FruitDetectionDataset(train_image_paths, train_mask_paths, train=True)
val_dataset = FruitDetectionDataset(val_image_paths, val_mask_paths, train=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Model
backbone = ResNet34FPNBackbone()
anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
    sizes=((32, 64, 128),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)
model = torchvision.models.detection.FasterRCNN(
    backbone,
    num_classes=2,
    rpn_anchor_generator=anchor_generator
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training Loop
model.train()
for epoch in range(10):

    loss_epoch = 0
    for images, targets in tqdm(train_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_epoch += losses
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    loss_epoch /= len(train_loader)

    # Log
    print(f"Epoch {epoch+1}: Loss {loss_epoch}")
    if wandb_log:
        wandb.log({
            "Loss": loss_epoch
        })

# Save model
if wandb_log:
    torch.save(model.state_dict(), model_name)

# Evaluation
metrics = evaluate_model(model, val_loader, device, iou_thresholds=np.arange(0.5, 1.0, 0.1))
for iou, met in metrics.items():
    print(f"\nmAP @ {iou:.2f}: {met['mAP']:.4f} | FP: {met['FP']} | FN: {met['FN']}")
    if wandb_log:
        wandb.log({
            f"mAP@{iou:.2f}": met['mAP'],
            f"FP@{iou:.2f}": met['FP'],
            f"FN@{iou:.2f}": met['FN']
        })

# Log
if wandb_log:
    wandb.finish()
