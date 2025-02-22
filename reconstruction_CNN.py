import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionCNN(nn.Module):
    def __init__(self):
        super(ReconstructionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 3 * 64 * 64)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 3, 64, 64)  # Çıkışı 64x64 renkli görüntüye dönüştür
        return x

# Modeli başlat
model = ReconstructionCNN()

# Modeli kaydet
torch.save(model.state_dict(), "hw1_3.pt")
print("Model başarıyla kaydedildi: hw1_3.pt")