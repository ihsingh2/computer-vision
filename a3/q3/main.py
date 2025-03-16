import json
import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import transforms
from torchvision.models.resnet import resnet34, ResNet34_Weights
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from tqdm import tqdm

class HumanPartsDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, train=False):
        self.data_list = data_list
        self.train = train
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

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img = Image.open(item['image_path']).convert("RGB")
        boxes = []
        labels = []
        for ann in item['annotations']:
            boxes.append(ann['bbox'])
            labels.append(ann['label'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, target

    def __len__(self):
        return len(self.data_list)

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

def collate_fn(batch):
    return tuple(zip(*batch))

def mask_to_bboxes(mask, color_to_label, scale_x, scale_y):
    """ Convert segmentation mask to bounding box annotations, scaled to the image size. """

    annotations = []
    for color, label in color_to_label.items():
        if label == 0:
            continue
        binary_mask = np.all(mask == np.array(color), axis=-1).astype(np.uint8)
        if binary_mask.sum() == 0:
            continue
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            scaled_bbox = [x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y]
            annotations.append({'label': label, 'bbox': scaled_bbox})
    return annotations

def visualize_bounding_boxes(dataset):
    """Visualize bounding boxes for a dataset."""

    fig, axes = plt.subplots(1, len(dataset), figsize=(4 * len(dataset), 9))
    fig.suptitle('Bounding Boxes overlaid on Original Images')

    for i, data in enumerate(dataset):
        image = np.array(Image.open(data['image_path']).convert('RGB'))
        axes[i].imshow(image)
        axes[i].set_title(f'Image {i + 1}')
        for ann in data['annotations']:
            x1, y1, x2, y2 = ann['bbox']
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            axes[i].add_patch(rect)
            axes[i].text(x1 + 2, y1 + 7, str(ann['label']), color='yellow', fontsize=12)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('visualization/bounding_boxes.png')
    plt.show()

def compute_iou(boxA, boxB):
    """ Compute the Intersection over Union of two boxes. """

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
    return iou

def compute_ap(preds, gts, iou_threshold):
    """ Compute Average Precision (AP) for a single class.

    Args:
        preds: list of (score, box) tuples, sorted by descending score.
        gts: list of ground truth boxes.
        iou_threshold: Threshold for IoU.
    """

    matched = [False] * len(gts)
    tp = []
    fp = []
    for score, pred_box in preds:
        best_iou = 0
        best_idx = -1
        for idx, gt_box in enumerate(gts):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_threshold and best_idx != -1 and not matched[best_idx]:
            tp.append(1)
            fp.append(0)
            matched[best_idx] = True
        else:
            tp.append(0)
            fp.append(1)
    tp = np.array(tp)
    fp = np.array(fp)
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / (len(gts) + 1e-6)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
    ap = 0
    prev_recall = 0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_recall)
        prev_recall = r
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    if len(f1_scores) > 0:
        best_f1_idx = np.argmax(f1_scores)
        best_precision = precisions[best_f1_idx]
        best_recall = recalls[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]
    else:
        best_precision = 0
        best_recall = 0
        best_f1 = 0
    return ap, best_precision, best_recall

def evaluate_model(model, dataloader, iou_threshold=0.5):
    """ Compute the mAP, per-class AP, precision, and recall. """

    model.eval()
    all_preds = {}
    all_gts = {}
    with torch.no_grad():
        for imgs, targets in tqdm(dataloader):
            imgs = list(img.to(device) for img in imgs)
            outputs = model(imgs)
            for i, output in enumerate(outputs):
                gt = targets[i]
                for j, label in enumerate(gt['labels'].tolist()):
                    if label not in all_gts:
                        all_gts[label] = []
                    all_gts[label].append(gt['boxes'][j].cpu().numpy())
                for j, label in enumerate(output['labels'].tolist()):
                    score = output['scores'][j].cpu().item()
                    box = output['boxes'][j].cpu().numpy()
                    if label not in all_preds:
                        all_preds[label] = []
                    all_preds[label].append((score, box))
    aps, pr, rec = {}, {}, {}
    classes = set(list(all_gts.keys()) + list(all_preds.keys()))
    for cls in classes:
        preds = all_preds.get(cls, [])
        gts = all_gts.get(cls, [])
        preds = sorted(preds, key=lambda x: x[0], reverse=True)
        ap, precision, recall = compute_ap(preds, gts, iou_threshold)
        aps[cls] = ap
        pr[cls] = precision
        rec[cls] = recall
    mAP = np.mean(list(aps.values())) if aps else 0
    return mAP, aps, pr, rec

data_dir = '../data/Q3/data'
color_to_label = {
    (  0,   0, 143): 0,     # Background
    (  0,  32, 255): 1,     # Hair
    (  0, 191, 255): 2,     # Face
    ( 96, 255, 159): 3,     # Torso
    (255,  80,   0): 4,     # Hands
    (255, 255,   0): 5,     # Legs
    (143,   0,   0): 6      # Feet
}

# # Create annotations
# dataset = []
# for subdir in os.listdir(data_dir):
#     subdir_path = os.path.join(data_dir, subdir)
#     if not os.path.isdir(subdir_path):
#         continue
#     for file in os.listdir(subdir_path):
#         if file.endswith('.jpg'):
#             img_path = os.path.join(subdir_path, file)
#             mask_file = file.replace('.jpg', '_m.png')
#             mask_path = os.path.join(subdir_path, mask_file)
#             if not os.path.exists(mask_path):
#                 continue
#             image = np.array(Image.open(img_path).convert('RGB'))
#             mask = np.array(Image.open(mask_path).convert('RGB'))
#             img_h, img_w, _ = image.shape
#             mask_h, mask_w, _ = mask.shape
#             scale_x = img_w / mask_w
#             scale_y = img_h / mask_h
#             ann = mask_to_bboxes(mask, color_to_label, scale_x, scale_y)
#             dataset.append({'image_path': img_path, 'mask_path': mask_path, 'annotations': ann})
# with open('dataset.json', 'w') as f:
#     json.dump(dataset, f)

# # Distribution of parts
# part_counts = [0 for idx in range(6)]
# areas = [[] for idx in range(6)]
# aspect_ratios = [[] for idx in range(6)]
# center_x = [[] for idx in range(6)]
# center_y = [[] for idx in range(6)]
# for item in dataset:
#     for ann in item['annotations']:
#         x1, y1, x2, y2 = ann['bbox']
#         part_counts[ann['label']] += 1
#         areas[ann['label']].append(abs(x2 - x1))
#         areas[ann['label']].append(abs(y2 - y1))
#         aspect_ratios[ann['label']].append(abs(x2 - x1) / abs(y2 - y1))
#         center_x[ann['label']].append((x1 + x2) / 2)
#         center_y[ann['label']].append((y1 + y2) / 2)
# for idx in range(6):
#     print(f'Part {idx}')
#     print(f'Count: {part_counts[idx]}')
#     print(f'Length: [{np.min(areas[idx]):.1f}, {np.max(areas[idx]):.1f}], {np.mean(areas[idx]):.1f}, {np.std(areas[idx]):.1f}')
#     print(f'Aspect: [{np.min(aspect_ratios[idx]):.1f}, {np.max(aspect_ratios[idx]):.1f}], {np.mean(aspect_ratios[idx]):.1f}, {np.std(aspect_ratios[idx]):.1f}')
#     print(f'Center: [({np.min(center_x[idx]):.1f}, {np.min(center_y[idx]):.1f}), ({np.max(center_x[idx]):.1f}, {np.max(center_y[idx]):.1f})], ({np.mean(center_x[idx]):.1f}, {np.mean(center_y[idx]):.1f}), ({np.std(center_x[idx]):.1f}, {np.std(center_y[idx]):.1f})')
#     print()

# # Visualization
# visualize_bounding_boxes([dataset[idx] for idx in range(5)])

# Read annotations
with open('dataset.json', 'r') as f:
    dataset = json.load(f)

# Seed
seed = 1122
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Dataset
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=seed)
train_dataset = HumanPartsDataset(train_data, train=True)
val_dataset = HumanPartsDataset(val_data, train=False)

# Dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Model
backbone = ResNet34FPNBackbone()
anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
    sizes=((32, 64, 128),),
    aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),)
)
model = torchvision.models.detection.FasterRCNN(
    backbone,
    num_classes=7,
    rpn_anchor_generator=anchor_generator
)

# Device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training Loop
for epoch in range(10):
    model.train()
    epoch_loss = 0.0
    for imgs, targets in tqdm(train_loader):
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
    print(f"Epoch {epoch+1}: Loss {epoch_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), 'model.pth')

# Evaluation
for iou in np.arange(0.5, 1.0, 0.2):
    mAP, aps, pr, rec = evaluate_model(model, val_loader, iou_threshold=iou)
    print(f"mAP@{iou:.2f}: {mAP:.4f}")
    for cls in sorted(aps.keys()):
        print(f"Class {cls}: AP = {aps[cls]:.4f}, Precision = {pr[cls]:.4f}, Recall = {rec[cls]:.4f}")
    print()
