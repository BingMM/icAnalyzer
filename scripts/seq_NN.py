#%% Imports

import numpy as np
import glob
from icreader import ConductanceImage
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

#%% Paths

p_in = '/home/bing/Dropbox/work/data/conductance/'

#%% Fetch orbits

orbit_files = sorted(glob.glob(p_in + '*.nc'))
o = [int(f[-7:-3]) for f in orbit_files]

#%% Load all data

H, P, dH, dP  = [], [], [], []
for orbit in tqdm(o, total=len(o)):
    filename = p_in + f'or_{str(orbit).zfill(4)}.nc'
    cI = ConductanceImage(filename)
    H.append(cI.H)   # shape: (N, 36, 36)
    P.append(cI.P)
    dH.append(cI.dH)
    dP.append(cI.dP)

#%% Load training labels

q = np.load('/home/bing/Dropbox/work/temp_storage/seq_classification_results.npz')
labeled_orbits = q['orbits']
indices = q['indices']
labels = q['labels']

#%% Extract training sequences (10-frame segments)

#%% Extract training sequences (10-frame segments) WITH MASKS

X, y = [], []

def prepare_seq(H_, P_, dH_, dP_, idx):
    # (10, 6, 36, 36) where channels are H, P, dH, dP, mask_H, mask_P
    h = H_[idx:idx+10]
    p = P_[idx:idx+10]
    dh = dH_[idx:idx+10]
    dp = dP_[idx:idx+10]
    mask = np.isnan(h) | np.isnan(dh) | np.isnan(p) | np.isnan(dp)
    mask = (~mask).astype(np.float32)

    # Replace NaNs with zero (NOT imputation – model ignores via masks)
    h = np.nan_to_num(h, nan=0.0)
    p = np.nan_to_num(p, nan=0.0)
    dh = np.nan_to_num(dh, nan=0.0)
    dp = np.nan_to_num(dp, nan=0.0)

    seq = np.stack([h, p, dh, dp, mask], axis=1)  # (10, 5, 36, 36)
    return seq

for i, idx, lab in zip(labeled_orbits, indices, labels):
    seq = prepare_seq(H[i], P[i], dH[i], dP[i], idx)
    X.append(seq)
    y.append(lab)

X = np.stack(X)  # (N, 10, 6, 36, 36)
y = np.array(y)

# Normalize only physical channels (0:4)
phys_channels = X[:, :, :4, :, :]
mean = phys_channels.mean()
std = phys_channels.std()
X[:, :, :4, :, :] = (phys_channels - mean) / std

#%% Create PyTorch dataset

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)

train_set, val_set = random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))])
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)

#%% Define CNN with 10 timesteps × 6 channels = 60 total input channels

class MaskedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(60, 64, kernel_size=(1, 3, 3), padding=(0,1,1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0,1,1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # Input x shape: (B, 10, 6, 36, 36)
        x = x.permute(0, 1, 2, 3, 4).reshape(x.size(0), -1, 36, 36)  # (B, 60, 36, 36)
        x = x.unsqueeze(2)  # (B, 60, 1, 36, 36) to match Conv3d input shape
        return self.fc(self.conv(x))


#%% Define simple classifier

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(10, 32, kernel_size=(1, 3, 3), padding=(0,1,1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0,1,1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # Reshape to (B, Channels=10, 2, 36, 36) → merge time & channel first
        x = x.permute(0, 1, 2, 3, 4)  # (B, 10, 2, 36, 36)
        x = x.reshape(x.size(0), x.size(1), -1, 36, 36)  # treat 10 time steps as separate channels
        x = self.conv(x)
        return self.fc(x)

#%% Train the model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.3f}")

#%% Predict on all orbits and collect good segments

predicted_good = []

model.eval()
with torch.no_grad():
    for i in tqdm(range(len(o))):
        if H[i].shape[0] < 10:
            continue
        for start in range(len(H[i]) - 9):
            seq = np.stack([H[i][start:start+10], P[i][start:start+10]], axis=1)
            seq = np.nan_to_num(seq, nan=0.0)
            seq = (seq - mean) / std
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(seq_tensor)
            label = pred.argmax(dim=1).item()
            if label == 1:  # predicted "good"
                predicted_good.append((i, start))

#%% Plot a few examples

for i, (orb_idx, start) in enumerate(predicted_good[:3]):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for t in range(10):
        ax = axes[t // 5, t % 5]
        img = H[orb_idx][start + t]
        ax.imshow(np.nan_to_num(img), cmap='inferno', origin='lower')
        ax.set_title(f'Time {t}')
        ax.axis('off')
    fig.suptitle(f'Orbit {o[orb_idx]}, Predicted GOOD segment starting at index {start}')
    plt.tight_layout()
    plt.show()



