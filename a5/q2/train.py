import argparse
import math
import os
import torch
import torch.nn.functional as F
import torchvision
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from diff_vit import DifferentialVisionTransformer


def train(model, optimizer, dataloader, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def main(args):
    wandb.init(project="cv-s25-a5-diff-vit", name=args.run_name, reinit=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_train = transforms.Compose([
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
        train_loss = train(model, optimizer, train_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        wandb.log({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": test_acc
        })
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} | "
              f"Test Loss {test_loss:.4f} | Test Acc {test_acc*100:.2f}%")

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('models', args.run_name + '.pt'))
    wandb.finish()


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
