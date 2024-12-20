import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np

class CNN_AutoEncoder(nn.Module):
    def __init__(self):
        super(CNN_AutoEncoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv4 = nn.Conv2d(32, 64, 5, padding=2)

        # Calculate flattened size after convolution
        self.input_size = 600  # Assuming input images are 600x600
        self.final_conv_size = self.input_size // (2 ** 3)  # Downsampled by 3 pooling layers
        self.flattened_size = 64 * (self.final_conv_size ** 2)  # Channels * final width * final height

        # Bottleneck layers
        self.fc1 = nn.Linear(self.flattened_size, 128)  
        self.fc2 = nn.Linear(128, self.flattened_size)  

        # Decoder
        self.deconv4 = nn.ConvTranspose2d(64, 32, 5, padding=2)
        self.deconv3 = nn.ConvTranspose2d(32, 16, 5, padding=2)
        self.deconv2 = nn.ConvTranspose2d(16, 6, 5, padding=2)
        self.deconv1 = nn.ConvTranspose2d(6, 3, 5, padding=2)

    def forward(self, x):
        # Encoding
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))  # Bottleneck

        # Decoding
        x = F.relu(self.fc2(x))  
        x = x.view(-1, 64, self.final_conv_size, self.final_conv_size)
        x = F.relu(self.deconv4(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.deconv3(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.deconv2(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.sigmoid(self.deconv1(x))
        return x

    def encode(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)  # Bottleneck
        return x




autoencoder = CNN_AutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.Resize((600, 600)),  
    transforms.ToTensor()
])

epochs = 3
batch_size = 16
directory_train = r"C:\Users\Kareem Hassani\OneDrive\Desktop\College\Year Three\Fall\EECE 490\SleepStudy_Stages\Train" 
directory_val = r"C:\Users\Kareem Hassani\OneDrive\Desktop\College\Year Three\Fall\EECE 490\SleepStudy_Stages\Validation"
directory_test = r"C:\Users\Kareem Hassani\OneDrive\Desktop\College\Year Three\Fall\EECE 490\SleepStudy_Stages\Test"

train_dataset = ImageFolder(directory_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print("Training AutoEncoder")
for epoch in range(epochs):
    for images, _ in train_loader:
        optimizer.zero_grad()
        outputs = autoencoder(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


latent_vectors = []
labels = []

for data, label in train_loader:
    latent_vector = autoencoder.encode(data) 
    latent_vectors.append(latent_vector.detach().numpy())
    labels.append(label.numpy())

latent_vectors = np.concatenate(latent_vectors)
labels = np.concatenate(labels)


GMM = GaussianMixture(n_components=5)
GMM.fit(latent_vectors)

GMM_clusters = GMM.predict(latent_vectors)

Colors = ["b", "g", "r", "k", "y"]  
Legends = ["N1 - Blue", "N2 - Green", "N3 - Red", "R - Black", "W - Yellow"]

plt.figure(figsize=(8, 6))
for cluster_id in range(5):
    cluster_points = latent_vectors[GMM_clusters == cluster_id]
    plt.scatter(
        cluster_points[:, 0], cluster_points[:, 1],   #
        c=Colors[cluster_id], label=Legends[cluster_id]
    )

plt.legend()
plt.title("GMM Clusters on Latent Features")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

def get_latent_vectors(dataset_loader):
    latent_vectors = []
    labels = []
    for data, label in dataset_loader:
        latent_vector = autoencoder.encode(data)
        latent_vectors.append(latent_vector.detach().numpy())
        labels.append(label.numpy())
    return np.concatenate(latent_vectors), np.concatenate(labels)


directory_val = r"C:\Users\Kareem Hassani\OneDrive\Desktop\College\Year Three\Fall\EECE 490\SleepStudy_Stages\Validation"
val_dataset = ImageFolder(directory_val, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
val_latent_vectors, val_labelsTrue = get_latent_vectors(val_loader)

directory_test = r"C:\Users\Kareem Hassani\OneDrive\Desktop\College\Year Three\Fall\EECE 490\SleepStudy_Stages\Test"
test_dataset = ImageFolder(directory_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_latent_vectors, test_labelsTrue = get_latent_vectors(test_loader)


print("Validation latent vectors shape:", val_latent_vectors.shape)
print("Validation labels shape:", val_labelsTrue.shape)




val_LabelsPredicted = GMM.predict(val_latent_vectors) 

test_LabelsPredicted = GMM.predict(test_latent_vectors)



predicted = np.array(val_LabelsPredicted)
true = np.array(val_labelsTrue)

conf_matrix = confusion_matrix(true, predicted)

row_ind, col_ind = linear_sum_assignment(-conf_matrix)  

mapping = {pred: true for pred, true in zip(col_ind, row_ind)}



mapped_predictions = np.array([mapping[label] for label in predicted])
print("Mapped PREDICTED Labels: ")
print(mapped_predictions)
print("True Labels: ")
print(true)


accuracy = np.mean(mapped_predictions == true)
print(f"Validation Accuracy: {accuracy:.2f}")

print("___________________")

predicted = np.array(test_LabelsPredicted)
true = np.array(test_labelsTrue)

conf_matrix = confusion_matrix(true, predicted)

row_ind, col_ind = linear_sum_assignment(-conf_matrix)  

mapping = {pred: true for pred, true in zip(col_ind, row_ind)}

mapped_predictions = np.array([mapping[label] for label in predicted])
print("Mapped PREDICTED Labels: ")
print(mapped_predictions)
print("True Labels: ")
print(true)


accuracy = np.mean(mapped_predictions == true)
print(f"Test Accuracy: {accuracy:.2f}")