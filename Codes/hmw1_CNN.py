"""
Yilmaz Ünal Student ID: 2023719108
This script loads a pre-collected dataset, preprocesses it, and trains a Convolutional Neural Network (CNN) 
to predict positional changes based on image inputs from an environment.
- Loads dataset from `dataset.pkl` if available; otherwise, prompts the user to collect data.
- Uses the `preprocess_data` function to convert raw image data into tensors.
- Splits the dataset into training (80%) and testing (20%) sets.
- Defines and initializes a CNN model with:
  - Convolutional layers for feature extraction.
  - Fully connected layers for positional prediction.
- Dynamically determines the correct flattened size for the fully connected layers.
- Trains the CNN model for 100 epochs using Mean Squared Error (MSE) loss and Adam optimizer.
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

train_dataset_cnn = torch.utils.data.TensorDataset(img_before_tensors[:train_size], pos_after_tensors[:train_size])
test_dataset_cnn = torch.utils.data.TensorDataset(img_before_tensors[train_size:], pos_after_tensors[train_size:])

train_loader_cnn = torch.utils.data.DataLoader(train_dataset_cnn, batch_size=8, shuffle=True)
test_loader_cnn = torch.utils.data.DataLoader(test_dataset_cnn, batch_size=8, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Dynamically determine the correct flattened size
        self._to_linear = None
        self._get_conv_output_size()

        self.fc_layers = nn.Sequential(
            nn.Linear(self._to_linear, 128), 
            nn.Linear(128, 2)
        )

    def _get_conv_output_size(self):
        """Pass a dummy input through conv layers to determine the flattened size."""
        with torch.no_grad():
            sample_input = torch.zeros(1, 3, 128, 128)  
            output = self.conv_layers(sample_input)
            self._to_linear = output.view(1, -1).shape[1]  # Compute dynamically

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically
        return self.fc_layers(x)

cnn_model = CNN()
cnn_criterion = nn.MSELoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.01)

# Training loop for CNN
for epoch in range(10):
    total_loss = 0.0
    for imgs, targets in train_loader_cnn:  # Use the DataLoader for batching
        cnn_optimizer.zero_grad()
        
        # Ensure correct shape without flattening
        imgs = imgs.view(imgs.shape[0], 3, 128, 128)  # Keep correct 4D shape 
        outputs = cnn_model(imgs)
        loss = cnn_criterion(outputs, targets)
        loss.backward()
        cnn_optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader_cnn)
    print(f"Epoch {epoch+1}, CNN Loss: {avg_loss}")

# Evaluate CNN model on test data        
cnn_model.eval()
test_loss_cnn = 0.0
with torch.no_grad():
    for imgs, targets in test_loader_cnn:
        outputs = cnn_model(imgs)
        loss = cnn_criterion(outputs, targets)
        test_loss_cnn += loss.item()

test_loss_cnn /= len(test_loader_cnn)
print(f"Test Loss (CNN): {test_loss_cnn}")