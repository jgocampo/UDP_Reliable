import torch
import torch.nn as nn
import argparse
import os

class VGG5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (3, 32, 32) -> (32, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> (32, 16, 16)
            nn.Conv2d(32, 64, 3, padding=1),             # -> (64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> (64, 8, 8)
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


class VGG8(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # -> (32, 16, 16)
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # -> (64, 8, 8)
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=str, choices=["vgg5", "vgg8"], required=True, help="Modelo a inicializar")
    args = parser.parse_args()

    if args.init == "vgg5":
        model = VGG5()
    elif args.init == "vgg8":
        model = VGG8()

    os.makedirs("global_models", exist_ok=True)
    protocol = input("Protocol (tcp / udp_raw / udp_reliable): ").strip()
    path = f"server/models/global_model_{args.init}_{protocol}.pt"
    torch.save(model.state_dict(), path)
    print(f"Global model initialized and saved {path}")
