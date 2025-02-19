"""
Yilmaz Ünal Student ID: 2023719108
This script defines a preprocessing function for preparing image-based dataset for a deep learning model.
- It applies transformations to images:
  - Resizes them to (64, 64) for consistency.
  - Converts them to tensors.
  - Normalizes pixel values to the range [-1, 1].
- The `preprocess_data` function converts raw data into tensors:
  - Transforms `img_before` into a tensor.
  - Converts `action_id` into a one-hot encoded tensor.
  - Converts `pos_after` into a float tensor.
- Returns stacked tensors for use in training a neural network.
"""
import torch
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Ensure correct size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Preprocessing function
def preprocess_data(data):
    img_before_tensors = []
    action_tensors = []
    pos_after_tensors = []

    for img_before, action_id, pos_after, _ in data:
        img_before_tensor = img_before.float() if isinstance(img_before, torch.Tensor) else transform(img_before)
        action_tensor = torch.zeros(4)
        action_tensor[action_id] = 1.0
        pos_after_tensor = torch.tensor(pos_after, dtype=torch.float32)

        img_before_tensors.append(img_before_tensor)
        action_tensors.append(action_tensor)
        pos_after_tensors.append(pos_after_tensor)

    return torch.stack(img_before_tensors), torch.stack(action_tensors), torch.stack(pos_after_tensors)