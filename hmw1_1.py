"""""
# This script trains a Multi-Layer Perceptron (MLP) model to predict positional changes of an object 
# based on actions taken in a robotic environment. 

# - Loads dataset from `dataset.pkl` if available.
# - Preprocesses data using the `preprocess_data` function.
# - Splits dataset into training (80%) and testing (20%) sets.
# - Defines and initializes an MLP model with fully connected layers.
# - Trains the model using Mean Squared Error (MSE) loss and Adam optimizer.
# - Saves the trained model as `hw1_1.pt`.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import sys
from Prep_funct import preprocess_data  # Import preprocessing function
import matplotlib.pyplot as plt

# Load dataset from pickle file
with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# Preprocess dataset
_, X, y = preprocess_data(dataset)

# Convert data to tensors (only if they're not already tensors)
X = X if isinstance(X, torch.Tensor) else torch.from_numpy(X).float()
y = y if isinstance(y, torch.Tensor) else torch.from_numpy(y).float()

# Split dataset into training and testing sets (80% training, 20% testing)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(X.shape[1], 64),  # Input layer
            nn.ReLU(),
            nn.Linear(64, 128),  # Hidden layer 1
            nn.ReLU(),
            nn.Linear(128, 64),  # Hidden layer 2
            nn.ReLU(),
            nn.Linear(64, y.shape[1])  # Output layer
        )

    def forward(self, x):
        return self.layers(x)

# Initialize model, loss function, and optimizer
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
train_losses = []

def train():
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "hw1_1.pt")
    print("Training complete. Model saved as 'hw1_1.pt'.")

# Testing function
def test():
    model.load_state_dict(torch.load("hw1_1.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
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
    plt.savefig("image/hmw1_1/loss_plot_1.png")

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