import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from shared.vgg_models import VGG5, VGG8
import numpy as np
import os

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            correct += (preds.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)
    return correct / total

def main(protocol=None, model_name=None, file_path=None):
    # Select model
    model_class = VGG5 if model_name == 'vgg5' else VGG8
    model = model_class()

    # Load weights
    if file_path:
        try:
            weights = np.load(file_path, allow_pickle=True)
            state_dict = {k: torch.tensor(v) for k, v in weights.items()}
            model.load_state_dict(state_dict, strict=False)
            print(f"📂 Loaded weights from {file_path}")
        except Exception as e:
            print(f"❌ Failed to load model from {file_path}: {e}")
            return
    else:
        model_path = f"server/models/global_model_{model_name}_{protocol}.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"📂 Loaded global model from {model_path}")

    # Load CIFAR-10 test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(testset, batch_size=64, shuffle=False)

    acc = evaluate(model, loader, device='cpu')
    print(f"✅ Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", help="Protocol used (for global model)")
    parser.add_argument("--model", choices=["vgg5", "vgg8"], required=True, help="Model architecture")
    parser.add_argument("--file", help="Optional .npz model update to evaluate")
    args = parser.parse_args()

    main(protocol=args.protocol, model_name=args.model, file_path=args.file)
