import clip
import numpy as np
import os
import socket
import time
import torch
from datetime import datetime
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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
    image_features = clip_model(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    if class_embeddings.dtype != image_features.dtype:
        class_embeddings = class_embeddings.to(image_features.dtype)
    similarity = (100 * image_features @ class_embeddings.T)
    _, topk_preds = similarity.topk(k, 1, True, True)
    probabilities = torch.nn.functional.softmax(similarity, dim=1)
    return topk_preds, probabilities


def record_memory_snapshot(model, images, max_entries=100000, TIME_FORMAT_STR="%b_%d_%H_%M_%S"):

    # FP32
    model_fp32 = model.float()
    images_fp32 = images.float()
    _ = model_fp32(images_fp32)
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    for _ in range(5):
        _ = model_fp32(images_fp32)
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"fp32_{host_name}_{timestamp}"
    try:
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
    torch.cuda.memory._record_memory_history(enabled=None)

    # FP16
    model_fp16 = model.half()
    images_fp16 = images.half()
    _ = model_fp16(images_fp16)
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    for _ in range(5):
        _ = model_fp16(images_fp16)
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"fp16_{host_name}_{timestamp}"
    try:
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
    torch.cuda.memory._record_memory_history(enabled=None)


def test_encoding_time(model, images, runs=100):

    # FP32
    model_fp32 = model.float()
    images_fp32 = images.float()
    _ = model_fp32(images_fp32)
    times_fp32 = []
    for _ in range(runs):
        start = time.time()
        _ = model_fp32(images)
        times_fp32.append(time.time() - start)
    fp32_mean = np.mean(times_fp32)
    fp32_std = np.std(times_fp32)

    # FP16
    model_fp16 = model.half()
    images_fp16 = images.half()
    _ = model_fp16(images_fp16)
    times_fp16 = []
    for _ in range(runs):
        start = time.time()
        _ = model_fp16(images_fp16)
        times_fp16.append(time.time() - start)
    fp16_mean = np.mean(times_fp16)
    fp16_std = np.std(times_fp16)

    return fp32_mean, fp32_std, fp16_mean, fp16_std


def main():

    data_dir = "../data/imagenet-mini"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, clip_preprocess = clip.load("RN50", device=device)
    clip_dataset = ImageNetDataset(data_dir, transform=clip_preprocess)

    with torch.no_grad():

        clip_dataloader = DataLoader(clip_dataset, batch_size=1, shuffle=False, num_workers=4)
        images = next(iter(clip_dataloader))[0].to(device)
        fp32_mean, fp32_std, fp16_mean, fp16_std = test_encoding_time(clip_model.visual, images)
        print(f"[FP32] Execution time: {fp32_mean} +- {fp32_std}")
        print(f"[FP16] Execution time: {fp16_mean} +- {fp16_std}")

        clip_class_embeddings = compute_clip_class_embeddings(clip_model, clip_dataset, device=device)
        clip_dataloader = DataLoader(clip_dataset, batch_size=5, shuffle=True, num_workers=4)
        images = next(iter(clip_dataloader))[0].to(device)
        topk_fp32, probabilities_fp32 = compute_clip_probabilities(clip_model.visual.float(), images, clip_class_embeddings)
        topk_fp16, probabilities_fp16 = compute_clip_probabilities(clip_model.visual.half(), images, clip_class_embeddings)
        topk_prob_fp32 = torch.stack([probabilities_fp32[i, topk_fp32[i]] for i in range(probabilities_fp32.size(0))])
        topk_prob_fp16 = torch.stack([probabilities_fp16[i, topk_fp32[i]] for i in range(probabilities_fp16.size(0))])
        print(f"[FP32] Top Probabilities: {topk_prob_fp32}")
        print(f"[FP16] Top Probabilities: {topk_prob_fp16}")

        clip_dataloader = DataLoader(clip_dataset, batch_size=32, shuffle=True, num_workers=4)
        images = next(iter(clip_dataloader))[0].to(device)
        record_memory_snapshot(clip_model.visual, images)

if __name__ == "__main__":
    main()
