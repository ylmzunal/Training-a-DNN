import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt

# Load dataset
with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f) # Ensure this function is correctly implemented

# Extract data correctly
img_before, action, pos_after, img_after = zip(*dataset)  # Unpack dataset

# Convert to tensors
img_before = torch.stack(img_before).float()
img_after = torch.stack(img_after).float()

# Ensure correct channel format (batch, 3, height, width)
if img_before.shape[1] != 3:
    img_before = img_before.permute(0, 3, 1, 2)

if img_after.shape[1] != 3:
    img_after = img_after.permute(0, 3, 1, 2)

# Normalize to range [0,1]
img_before /= 255.0
img_after /= 255.0

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

class ReconstructionCNN(nn.Module):
    def __init__(self):
        super(ReconstructionCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid()  # Normalize output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

train_losses = []

def train():
    model = ReconstructionCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(img_before.to(device))
        loss = criterion(outputs, img_after.to(device))
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

    # Save trained model
    torch.save(model.state_dict(), "hw1_3.pt")
    print("Training complete. Model saved as 'hw1_3.pt'.")

import matplotlib.pyplot as plt

def test():
    model = ReconstructionCNN()
    model.load_state_dict(torch.load("hw1_3.pt", weights_only=True))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        predicted_images = model(img_before.to(device))

    for i in range(5):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(img_before[i].permute(1, 2, 0).cpu().numpy())
        plt.title("Before")

        plt.subplot(1, 3, 2)
        plt.imshow(img_after[i].permute(1, 2, 0).cpu().numpy())
        plt.title("Actual")

        plt.subplot(1, 3, 3)
        plt.imshow(predicted_images[i].permute(1, 2, 0).cpu().numpy())
        plt.title("Predicted")

        plt.savefig(f"image/output_{i}.png")
        plt.close()

    print("Predicted images saved.")

def plot_loss():
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", marker='o', linestyle="-", color='b')
    # plt.plot(epochs, test_losses * len(train_losses), label="Test Loss", marker='s', linestyle="--", color='r')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("image/loss_plot_3.png")

    # Command-line argument handling
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train()
        elif sys.argv[1] == "test":
            test()
        else:
            print("Invalid argument. Use 'train' or 'test'.")
    else:
        train()
        test()
        plot_loss