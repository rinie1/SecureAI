import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# Усложнённые аугментации
transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# Улучшенная модель: 784 -> 128 -> ReLU -> 10
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Тренировка
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Сохраняем только fc2 (линейный слой для сервера)
W2 = model.fc2.weight.data.cpu().numpy()  # (10, 128)
b2 = model.fc2.bias.data.cpu().numpy()    # (10,)
# Также сохраняем fc1 для использования на клиенте
W1 = model.fc1.weight.data.cpu().numpy()  # (128, 784)
b1 = model.fc1.bias.data.cpu().numpy()    # (128,)

np.savez("mnist_model_split.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("Модель обучена и веса сохранены.")
