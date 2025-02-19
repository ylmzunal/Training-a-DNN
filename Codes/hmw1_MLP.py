"""
Yilmaz Ünal Student ID: 2023719108
This script loads a pre-collected dataset, preprocesses it, and trains an MLP (Multi-Layer Perceptron) model 
for predicting positional changes based on actions taken in an environment.

- Loads dataset from `dataset.pkl` if available, otherwise prompts the user to run data collection.
- Uses the `preprocess_data` function to convert raw data into tensors.
- Splits the dataset into training (80%) and testing (20%) sets.
- Defines and initializes an MLP model with:
  - Input: One-hot encoded action tensor (4-dimensional).
  - Hidden layers: Two fully connected layers with ReLU activation.
  - Output: 2-dimensional positional prediction.
- Trains the MLP model for 100 epochs using Mean Squared Error (MSE) loss and Adam optimizer.
- Evaluates the trained model on the test dataset and prints the final loss.

"""

import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from Prep_funct import preprocess_data


# Check if the dataset already exists
dataset_path = "dataset.pkl" 

if os.path.exists(dataset_path):
    print(f"📂 Loading dataset from {dataset_path}...")
    
    # Load the dataset from the file
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    
    print("✅ Dataset loaded successfully!")
else:
    print("⚠️ Dataset not found! Run the data collection code first.")

# Convert dataset to tensors
img_before_tensors, action_tensors, pos_after_tensors = preprocess_data(data)

# Split dataset into training and testing
train_size = int(0.8 * len(action_tensors))
test_size = len(action_tensors) - train_size

# Separate datasets for MLP
train_dataset_mlp = torch.utils.data.TensorDataset(action_tensors[:train_size], pos_after_tensors[:train_size])
test_dataset_mlp = torch.utils.data.TensorDataset(action_tensors[train_size:], pos_after_tensors[train_size:])
train_loader_mlp = torch.utils.data.DataLoader(train_dataset_mlp, batch_size=8, shuffle=True)
test_loader_mlp = torch.utils.data.DataLoader(test_dataset_mlp, batch_size=8, shuffle=False)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(x)

mlp_model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)

# Training loop for MLP
for epoch in range(100):
    optimizer.zero_grad()
    predictions = mlp_model(action_tensors[:train_size])
    loss = criterion(predictions, pos_after_tensors[:train_size])
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, MLP Loss: {loss.item()}")

# Evaluate MLP model on test data
mlp_model.eval()
test_loss_mlp = 0.0
with torch.no_grad():
    for actions, targets in test_loader_mlp:
        outputs = mlp_model(actions)
        loss = criterion(outputs, targets)
        test_loss_mlp += loss.item()

test_loss_mlp /= len(test_loader_mlp)
print(f"Test Loss (MLP): {test_loss_mlp}")
