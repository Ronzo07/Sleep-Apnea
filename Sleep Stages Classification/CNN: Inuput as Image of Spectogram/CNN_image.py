import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

directory_train = r"C:\Users\Kareem Hassani\OneDrive\Desktop\College\Year Three\Fall\EECE 490\SleepStudy_Stages\Train" 
directory_val = r"C:\Users\Kareem Hassani\OneDrive\Desktop\College\Year Three\Fall\EECE 490\SleepStudy_Stages\Validation"
directory_test = r"C:\Users\Kareem Hassani\OneDrive\Desktop\College\Year Three\Fall\EECE 490\SleepStudy_Stages\Test"

transform = transforms.Compose([transforms.Resize((224, 224)),  transforms.ToTensor()])
class MyDataSet(Dataset):
    def __init__(self, directoryOfDataset, transform=None):
        self.data = ImageFolder(directoryOfDataset, transform=transform)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class CNN_FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  
        self.conv2 = nn.Conv2d(6, 16, 5)  
        self.conv3 = nn.Conv2d(16, 32, 3)  
        self.pool = nn.MaxPool2d(2, 2)  # Reduce spatial dimensions by half
        self.bn1 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, 3)  
        self.conv5 = nn.Conv2d(64, 128, 3)  
        self.conv6 = nn.Conv2d(128, 256, 3)  
        self.bn2 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, 3)  
        self.conv8 = nn.Conv2d(512, 1024, 3)  
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(1024, 5)  # Output: 5 classes
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.bn1(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = self.bn2(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))

        x = self.adaptive_pool(x)  # Global average pooling
        x = torch.flatten(x, 1)  # Flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x

dataset_train = MyDataSet(directory_train, transform=transform)

dataloader_train = DataLoader(dataset_train, batch_size=10, shuffle=True)

dataset_val = MyDataSet(directory_val, transform=transform) 

dataloader_val = DataLoader(dataset_val, batch_size=10, shuffle=False)

dataset_test = MyDataSet(directory_test, transform=transform)

dataloader_test = DataLoader(dataset_test, batch_size=10, shuffle=False)


model = CNN_FF()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 10
for epoch in range(epochs):
    
    model.train()
    print("Hi")
    running_loss = 0

    for images, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader_train):.4f}')
    

def evaluate_model(loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total



val_accuracy = evaluate_model(dataloader_val)
test_accuracy = evaluate_model(dataloader_test)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


