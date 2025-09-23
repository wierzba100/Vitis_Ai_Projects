import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "mnist_cnn.pth"
img_path = "seven.png"

# Transformacje takie same jak przy trenowaniu
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Definicja modelu taka sama jak przy trenowaniu
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Wczytanie modelu
model = Net().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Wczytanie obrazu
img = Image.open(img_path)
img = transform(img).unsqueeze(0).to(device)

# Detekcja
start_time = time.time()
with torch.no_grad():
    output = model(img)
    pred = output.argmax(dim=1, keepdim=True)
end_time = time.time()

print("Rozpoznana cyfra:", pred.item())
print(f"Czas detekcji: {end_time - start_time:.6f} sekund")
