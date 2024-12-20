from mne.io import read_raw_edf
from scipy.signal import spectrogram
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



def organize_labels(path):
    # Load the label
    raw_labels = pd.read_csv(path, usecols=[' Sleep Stage'])  # Skip header row

    # Assign W to 0, Ni to i, and R to 4
    numbered_labels = []
    for row in raw_labels.values:
        if(row[0] == ' W'):
            numbered_labels.append(0)
        
        elif(row[0] == ' N1'):
            numbered_labels.append(1)
        
        elif(row[0] == ' N2'):
            numbered_labels.append(2)
        
        elif(row[0] == ' N3'):
            numbered_labels.append(3)
        
        elif(row[0] == ' R'):
            numbered_labels.append(4)
    
    return np.array(numbered_labels)

# Process the data
def read_EEG_file_and_store_in_array(path):
    raw_data = read_raw_edf(path, preload=True, verbose=False)
    data = raw_data.get_data()
    sampling_freq = int(raw_data.info['sfreq'])
    segment_duration = 30  # seconds
    num_datapoints = sampling_freq * segment_duration
 
    # Get indices of target channels
    all_channels = raw_data.info['ch_names']
    channels_to_use = [all_channels.index(channel) for channel in TARGET_CHANNELS if channel in all_channels]
    if not channels_to_use:
        raise ValueError("None of the target channels are available in the EDF file.")
 
    spec_intervals = []
    for j in range(0, len(raw_data.times) // num_datapoints):
        start = j * num_datapoints
        end = start + num_datapoints
        temp_data = data[:, start:end]
 
        temp_channel_specs = []
        for channel in channels_to_use:
            # Compute spectrogram
            f, t, Sxx = spectrogram(
                temp_data[channel, :],
                fs=sampling_freq,
                window='hann',
                nperseg=256,
                noverlap=128,
                scaling='density',
                mode='magnitude'
            )

            Sxx = np.log1p(Sxx)
            temp_channel_specs.append(Sxx)
 
        
        spec_intervals.append(np.stack(temp_channel_specs, axis=0))
 
    return np.array(spec_intervals)




class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # First block
        self.conv1_1 = nn.Conv2d(7, 16, 3, 1, padding=1)  
        self.conv1_2 = nn.Conv2d(16, 32, 3, 1, padding=1)  
        self.conv1_3 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.pool_1 = nn.MaxPool2d(2, 2) 
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.5)
        
        # Second block
        self.conv2_1 = nn.Conv2d(32, 16, 1, 1) 
        self.conv2_2 = nn.Conv2d(16, 32, 3, 1, padding=1)  
        self.conv2_3 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.pool_2 = nn.AvgPool2d(2, 2)  
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.5)
        
        # Third block
        self.conv3_1 = nn.Conv2d(32, 20, 1, 1) 
        self.conv3_2 = nn.Conv2d(20, 64, 3, 1, padding=1)  
        self.conv3_3 = nn.Conv2d(64, 64, 3, 1, padding=1)  
        self.pool_3 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(0.5)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  
        
        # Fully connected layer
        self.fc = nn.Linear(64, 5)  

    def forward(self, x):

        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv1_3(x))
        x = self.pool_1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        block1_output = x  
        
       
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = self.pool_2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        
        if x.size() == block1_output.size():
            x = x + block1_output
        
        
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool_3(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        
        
        x = self.gap(x)
        x = torch.flatten(x, 1)  
        
        
        x = self.fc(x)
        
        return x


def evaluate_model(model, dataloader):
    model.eval()  
    correct = 0
    total = 0
    stage_map = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}

    with torch.no_grad():  
        for batch_spectrograms, batch_labels in dataloader:
            outputs = model(batch_spectrograms)
            _, predicted = torch.max(outputs, 1)  
            
            
            for i, (expected, pred) in enumerate(zip(batch_labels, predicted)):
                print(f"Interval {i+1}: Expected: {stage_map[expected.item()]}, Predicted: {stage_map[pred.item()]}")
            
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total
    return accuracy
   



# Specify target channels
TARGET_CHANNELS = ['EEG F3-REF', 'EEG F4-REF', 'EEG O1-REF',
                   'EEG O2-REF', 'EEG C3-REF', 'EEG C4-REF', 'EMG Chin-REF']


path1_edf = "./Data/PSG_raw_WPS_0075.edf"
path2_edf = "./Data/PSG_raw_WPS_0083.edf"
path3_edf = "./Data/PSG_raw_WPS_0064.edf"
path4_edf = "./Data/PSG_raw_WPS_0084.edf"


path1_csv = "./Data/PSG_sleepstages_WPS_0075.csv"
path2_csv = "./Data/PSG_sleepstages_WPS_0083.csv"
path3_csv = "./Data/PSG_sleepstages_WPS_0064.csv"
path4_csv = "./Data/PSG_sleepstages_WPS_0084.csv"


# test dataset
spectrograms4 = read_EEG_file_and_store_in_array(path4_edf)
labels4 = organize_labels(path4_csv)[:len(spectrograms4)]

# Convert test data to tensors
test_spectrograms_tensor = torch.tensor(spectrograms4, dtype=torch.float32)
test_labels_tensor = torch.tensor(labels4, dtype=torch.long)
test_dataset = TensorDataset(test_spectrograms_tensor, test_labels_tensor)


# training datasets combined
spectrograms1 = read_EEG_file_and_store_in_array(path1_edf)
labels1 = organize_labels(path1_csv)[:len(spectrograms1)]

spectrograms2 = read_EEG_file_and_store_in_array(path2_edf)
labels2 = organize_labels(path2_csv)[:len(spectrograms2)]

spectrograms3 = read_EEG_file_and_store_in_array(path3_edf)
labels3 = organize_labels(path3_csv)[:len(spectrograms3)]

spectrograms = np.concatenate([spectrograms1, spectrograms2,spectrograms3])
labels = np.concatenate([labels1, labels2,labels3])


spectrograms_tensor = torch.tensor(spectrograms, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)
dataset = TensorDataset(spectrograms_tensor, labels_tensor)


model = Net()


# Use DataLoader for batching and shuffling
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(20):
    running_loss = 0.0
    for batch_spectrograms, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_spectrograms)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader):.4f}")

print("Finished Training")
        
        
# Evaluate on test data
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
accuracy = evaluate_model(model, test_dataloader)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    