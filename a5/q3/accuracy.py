import clip
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm


class ImageNetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.synset_to_idx = {}
        self.idx_to_labels = []
        self.image_paths = []
        self.image_classes = []
        self.class_sample_counts = np.zeros(1000, dtype=int)

        with open(os.path.join(data_dir, 'synset_mapping.txt'), 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split(' ', 1)
                synset = parts[0]
                labels_str = parts[1] if len(parts) > 1 else ''
                labels = [label.strip() for label in labels_str.split(',')]
                self.synset_to_idx[synset] = i
                self.idx_to_labels.append(labels)

        val_dir = os.path.join(data_dir, 'val')
        for synset in os.listdir(val_dir):
            synset_dir = os.path.join(val_dir, synset)
            if os.path.isdir(synset_dir) and synset in self.synset_to_idx:
                class_idx = self.synset_to_idx[synset]
                for img_file in os.listdir(synset_dir):
                    if img_file.endswith('.JPEG'):
                        self.image_paths.append(os.path.join(synset_dir, img_file))
                        self.image_classes.append(class_idx)
                        self.class_sample_counts[class_idx] += 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.image_classes[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def compute_resnet_accuracy(model, dataloader, k=5, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    num_classes = 1000
    misclassified_counts = torch.zeros(num_classes, dtype=torch.int64).to(device)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, topk_preds = outputs.topk(k, 1, True, True)

            expanded_labels = labels.view(-1, 1).expand_as(topk_preds)
            matches = torch.eq(topk_preds, expanded_labels)
            is_correct = matches.any(dim=1)
            correct += torch.sum(is_correct).item()
            total += labels.size(0)

            misclassified_mask = ~is_correct
            misclassified_labels = labels[misclassified_mask]
            if misclassified_labels.numel() > 0:
                misclassified_in_batch = torch.bincount(
                    misclassified_labels, minlength=num_classes
                ).to(device)
                misclassified_counts += misclassified_in_batch

    accuracy = correct / total if total > 0 else 0
    return accuracy, misclassified_counts


def compute_clip_accuracy(clip_model, dataloader, class_embeddings, k=5, device="cpu"):
    clip_model.eval()
    correct = 0
    total = 0
    num_classes = 1000
    misclassified_counts = torch.zeros(num_classes, dtype=torch.int64).to(device)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating CLIP"):
            images = images.to(device)
            labels = labels.to(device)
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ class_embeddings.T)
            _, topk_preds = similarity.topk(k, 1, True, True)

            expanded_labels = labels.view(-1, 1).expand_as(topk_preds)
            matches = torch.eq(topk_preds, expanded_labels)
            is_correct = matches.any(dim=1)
            correct += torch.sum(is_correct).item()
            total += labels.size(0)

            misclassified_mask = ~is_correct
            misclassified_labels = labels[misclassified_mask]
            if misclassified_labels.numel() > 0:
                misclassified_in_batch = torch.bincount(
                    misclassified_labels, minlength=num_classes
                ).to(device)
                misclassified_counts += misclassified_in_batch

    accuracy = correct / total if total > 0 else 0
    return accuracy, misclassified_counts


def compute_clip_class_embeddings(clip_model, dataset, device="cpu"):
    class_labels_list = dataset.idx_to_labels
    clip_class_embeddings = []
    with torch.no_grad():
        for labels in tqdm(class_labels_list, desc="Processing text embeddings"):
            tokens = clip.tokenize([ "A photo of a " + noun for noun in labels ]).to(device)
            text_features = clip_model.encode_text(tokens)
            avg_text_feature = text_features.mean(dim=0)
            avg_text_feature /= avg_text_feature.norm()
            clip_class_embeddings.append(avg_text_feature)
    clip_class_embeddings = torch.stack(clip_class_embeddings).to(device)
    return clip_class_embeddings


def compute_clip_probabilities(clip_model, images, class_embeddings, k=5):
    image_features = clip_model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    similarity = (100 * image_features @ class_embeddings.T)
    _, topk_preds = similarity.topk(k, 1, True, True)
    probabilities = torch.nn.functional.softmax(similarity, dim=1)
    return topk_preds, probabilities


def analyze_results(accuracy, misclassified_counts, dataset, name, top_n=10):
    print(f"[{name}] Top-5 Accuracy: {accuracy:.4f}")

    class_counts = dataset.class_sample_counts
    error_rates = misclassified_counts.cpu().numpy() / np.maximum(class_counts, 1)
    top_indices = np.argsort(error_rates)[-top_n:][::-1]

    print(f"\nTop {top_n} Most Misclassified Classes:")
    for idx in top_indices:
        class_labels = dataset.idx_to_labels[idx]
        error_rate = error_rates[idx]
        misclassified = misclassified_counts[idx].item()
        total_samples = class_counts[idx]
        print(f"Class {idx} ({class_labels[0]}): {misclassified}/{total_samples} misclassified ({error_rate:.2f})")
    print()


def main():

    data_dir = "../data/imagenet-mini"
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ImageNet-ResNet50
    model_resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    resnet_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    resnet_dataset = ImageNetDataset(data_dir, transform=resnet_transform)
    resnet_dataloader = DataLoader(resnet_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    resnet_acc, resnet_misclass = compute_resnet_accuracy(
        model_resnet50, resnet_dataloader, k=5, device=device
    )
    analyze_results(resnet_acc, resnet_misclass, resnet_dataset, "ImageNet-ResNet50")

    # CLIP-ResNet50
    clip_model, clip_preprocess = clip.load("RN50", device=device)
    clip_dataset = ImageNetDataset(data_dir, transform=clip_preprocess)
    clip_dataloader = DataLoader(clip_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    clip_class_embeddings = compute_clip_class_embeddings(clip_model, clip_dataset, device=device)

    clip_acc, clip_misclass = compute_clip_accuracy(
        clip_model, clip_dataloader, clip_class_embeddings, k=5, device=device
    )
    analyze_results(clip_acc, clip_misclass, clip_dataset, "CLIP-ResNet50")

    # Comparision
    better_classes_resnet = [ clip_dataset.idx_to_labels[idx] for idx in (clip_misclass - resnet_misclass).topk(3, 0, True, True)[1] ]
    better_classes_clip = [ clip_dataset.idx_to_labels[idx] for idx in (resnet_misclass - clip_misclass).topk(3, 0, True, True)[1] ]
    print(f"Classes where ImageNet-Resnet50 performs better: {better_classes_resnet}")
    print(f"Classes where CLIP-Resnet50 performs better: {better_classes_clip}")


if __name__ == "__main__":
    main()
