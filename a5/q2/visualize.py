import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from diff_vit import DifferentialVisionTransformer


def visualize_last_layer_attention(model, dataloader, device):
    model.eval()
    images, _ = next(iter(dataloader))
    images = images.to(device)

    _, attentions = model(images)
    last_attn = attentions[-1].detach().cpu()
    num_heads, num_tokens = last_attn[0][:, 0, 1:].shape
    grid_size = int(np.sqrt(num_tokens))

    num_images = min(images.shape[0], 4)
    num_heads = min(num_heads, 8)
    fig, axs = plt.subplots(num_images, num_heads + 2, figsize=(2 * (num_heads + 2), 2 * num_images))
    fig.suptitle("Last Layer [CLS] Token Attentions")

    for i in range(num_images):
        image = images[i].cpu().permute(1, 2, 0).numpy()
        image = np.array([0.229, 0.224, 0.225]) * image + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)

        cls_attn = last_attn[0][:, 0, 1:]
        avg_attn = cls_attn.mean(dim=0).reshape(grid_size, grid_size).cpu().numpy()
        resized_avg_attn = cv2.resize(avg_attn, images.shape[2:], interpolation=cv2.INTER_LINEAR)
        normalized_avg_attn = (resized_avg_attn - np.min(resized_avg_attn)) / (np.max(resized_avg_attn) - np.min(resized_avg_attn) + 1e-8)
        blended_avg_attn = image * np.expand_dims(normalized_avg_attn, axis=-1)

        axs[i, 0].imshow(image)
        axs[i, 0].set_title("Input Image")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(blended_avg_attn)
        axs[i, 1].set_title("Average")
        axs[i, 1].axis('off')

        for j in range(num_heads):
            head_attn = cls_attn[j].reshape(grid_size, grid_size).cpu().numpy()
            resized_head_attn = cv2.resize(head_attn, images.shape[2:], interpolation=cv2.INTER_LINEAR)
            normalized_head_attn = (resized_head_attn - np.min(resized_head_attn)) / (np.max(resized_head_attn) - np.min(resized_head_attn) + 1e-8)
            blended_head_attn = image * np.expand_dims(normalized_head_attn, axis=-1)

            axs[i, j + 2].imshow(blended_head_attn)
            axs[i, j + 2].set_title(f"Head {j + 1}")
            axs[i, j + 2].axis('off')

    plt.savefig(os.path.join('visualization', 'last_layer_attention.png'), bbox_inches='tight')
    plt.close()
    plt.clf()


def visualize_all_layer_attention(model, dataloader, device):
    model.eval()
    images, _ = next(iter(dataloader))
    images = images.to(device)

    _, attentions = model(images)
    num_layers = len(attentions)
    num_images = min(images.shape[0], 4)
    num_tokens = len(attentions[0][0, 0, 1:])
    grid_size = int(np.sqrt(num_tokens))

    fig, axs = plt.subplots(num_images, num_layers + 1, figsize=(2 * (num_layers + 1), 2 * num_images))
    fig.suptitle("Layerwise Average [CLS] Token Attentions")

    for i in range(num_images):
        image = images[i].cpu().permute(1, 2, 0).numpy()
        image = np.array([0.229, 0.224, 0.225]) * image + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)

        axs[i, 0].imshow(image)
        axs[i, 0].set_title("Input Image")
        axs[i, 0].axis('off')

        for layer_idx, attn in enumerate(attentions):
            attn_layer = attn[i].detach().cpu()
            avg_attn = attn_layer.mean(dim=0)[0, 1:].reshape(grid_size, grid_size).cpu().numpy()
            avg_attn = cv2.resize(avg_attn, images.shape[2:], interpolation=cv2.INTER_LINEAR)
            normalized_avg_attn = (avg_attn - np.min(avg_attn)) / (np.max(avg_attn) - np.min(avg_attn) + 1e-8)
            blended_avg_attn = image * np.expand_dims(normalized_avg_attn, axis=-1)

            axs[i, layer_idx + 1].imshow(blended_avg_attn)
            axs[i, layer_idx + 1].set_title(f"Layer {layer_idx + 1}")
            axs[i, layer_idx + 1].axis('off')

    plt.savefig(os.path.join('visualization', 'all_layer_attention.png'), bbox_inches='tight')
    plt.close()
    plt.clf()


def compute_attention_rollout(attentions):
    device = attentions[0].device
    num_images, _, num_tokens, _ = attentions[0].shape
    rollout = [ torch.eye(num_tokens, device=device) for _ in range(num_images) ]

    for image in range(num_images):
        for attn in attentions:
            attn_mean = attn[image].mean(dim=0)
            attn_aug = attn_mean + torch.eye(num_tokens, device=device)
            attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
            rollout[image] = attn_aug @ rollout[image]

    return rollout


def visualize_attention_rollout(model, dataloader, device):
    model.eval()
    images, _ = next(iter(dataloader))
    images = images.to(device)

    _, attentions = model(images)
    rollout = compute_attention_rollout(attentions)
    rollout_patch = [ rollout[image][0, 1:].detach().cpu().numpy() for image in range(len(images)) ]
    grid_size = int(np.sqrt(rollout_patch[0].shape[0]))
    rollout_map = [ rollout_patch[image].reshape(grid_size, grid_size) for image in range(len(images)) ]

    num_images = images.shape[0]
    fig = plt.figure(figsize=(3 * num_images, 5))
    fig.suptitle('Input Images and their Attention Rollouts')

    for i in range(num_images):

        image = images[i].cpu().permute(1, 2, 0).numpy()
        image = np.array([0.229, 0.224, 0.225]) * image + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)

        resized_attention = cv2.resize(rollout_map[i], images.shape[2:], interpolation=cv2.INTER_LINEAR)
        normalized_attention = (resized_attention - np.min(resized_attention)) / (np.max(resized_attention) - np.min(resized_attention) + 1e-8)
        blended_image = image * np.expand_dims(normalized_attention, axis=-1)

        plt.subplot(2, num_images, i + 1)
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(blended_image)
        plt.axis('off')

    plt.savefig(os.path.join('visualization', 'attention_rollout.png'), bbox_inches='tight')
    plt.close()
    plt.clf()


def visualize_positional_embedding(model):
    pos_embed = model.pos_embed.detach().cpu().squeeze(0)[1:]
    sim_matrix = (pos_embed @ pos_embed.T).numpy()

    fig = plt.figure(figsize=(8,8))
    im = plt.imshow(sim_matrix, cmap='plasma')
    plt.title("Similarity between Learned 1D Positional Embeddings")
    plt.ylabel("Patch Index")
    plt.xlabel("Patch Index")
    cbar = fig.colorbar(im)
    cbar.set_label('Dot Product')
    plt.savefig(os.path.join('visualization', 'positional_embeddings.png'), bbox_inches='tight')
    plt.close()
    plt.clf()


def main(args):
    os.makedirs('visualization', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=2)

    model = DifferentialVisionTransformer(
        image_size=32,
        in_channels=3,
        num_classes=10,
        emb_dim=args.emb_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_hidden_dim=args.mlp_hidden_dim,
        patch_size=args.patch_size,
        pos_embed_type=args.pos_embed_type,
        dropout=args.dropout
    ).to(device)
    model.load_state_dict(torch.load(os.path.join('models', args.run_name + '.pt'), map_location=device))

    visualize_last_layer_attention(model, test_loader, device)
    visualize_all_layer_attention(model, test_loader, device)
    visualize_attention_rollout(model, test_loader, device)
    # visualize_positional_embedding(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Diff-ViT on CIFAR10")
    parser.add_argument("--run_name", type=str, help="Run name for WandB logging", required=True)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--emb_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=6, help="Number of transformer encoder layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--mlp_hidden_dim", type=int, default=512, help="Hidden dimension in MLP")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size")
    parser.add_argument("--pos_embed_type", type=str, default="1d",
                        choices=["none", "1d", "2d", "sinusoidal"],
                        help="Type of positional embedding")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    args = parser.parse_args()

    main(args)
