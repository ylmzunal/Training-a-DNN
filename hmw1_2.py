# Read the contents of hmw1_CNN.py to modify it according to hmw1_1.py's structure


import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from Prep_funct import preprocess_data  # Import preprocessing function
import matplotlib.pyplot as plt

# Load dataset from pickle file
with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# Preprocess dataset
img_before, X, y = preprocess_data(dataset)  # Extract image data as well

# Convert data to tensors
img_before = (img_before if isinstance(img_before, torch.Tensor) 
             else torch.from_numpy(img_before)).float()
# Print shape for debugging
print("Image shape before permute:", img_before.shape)
# Ensure correct shape: (batch_size, channels, height, width)
if len(img_before.shape) == 4 and img_before.shape[-1] == 3:  # If shape is (batch, height, width, channels)
    img_before = img_before.permute(0, 3, 1, 2)
print("Image shape after permute:", img_before.shape)

X = X.clone().detach() if isinstance(X, torch.Tensor) else torch.from_numpy(X).float()
y = y.clone().detach() if isinstance(y, torch.Tensor) else torch.from_numpy(y).float()

# Split dataset into training and testing sets (80% training, 20% testing)
split_idx = int(0.8 * len(X))
img_train, img_test = img_before[:split_idx], img_before[split_idx:]
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First get input channels from data
        in_channels = img_before.shape[1]  # Should be 3 for RGB
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Calculate the flattened size based on input dimensions
        self.flatten_size = 32 * img_before.shape[2] * img_before.shape[3]
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, y.shape[1])

    def forward(self, img):
        x = torch.relu(self.conv1(img))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = CNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
train_losses = []

def train():
    print("Starting CNN model training...")
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(img_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "hw1_2.pt")
    print("Training complete. Model saved as 'hw1_2.pt'.")

# Testing function
def test():
    print("Starting CNN model evaluation...")
    model.load_state_dict(torch.load("hw1_2.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        predictions = model(img_test)
        loss = criterion(predictions, y_test)
        print(f"Test Loss: {loss.item():.4f}")

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
    plt.savefig("image/hmw1_1/loss_plot_2.png")

# Run training
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
        plot_loss()