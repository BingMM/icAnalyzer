#%%

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

#%%
# 1. Load your data
H = np.load('/home/bing/Dropbox/work/temp_storage/H.npy')  # shape (N, 36, 36)
P = np.load('/home/bing/Dropbox/work/temp_storage/P.npy')  # shape (N, 36, 36)
labels_file = np.load('/home/bing/Dropbox/work/temp_storage/classification_results.npz')
indices = labels_file['indices']
labels = labels_file['labels']

#%%

# 2. Balance the dataset (equal good/bad)
good_idxs = indices[labels == 1]
bad_idxs = indices[labels == 0]

num_good = len(good_idxs)
balanced_bad_idxs = np.random.choice(bad_idxs, size=num_good, replace=False)

balanced_indices = np.concatenate([good_idxs, balanced_bad_idxs])
balanced_labels = np.concatenate([np.ones(num_good), np.zeros(num_good)])

# Shuffle
perm = np.random.permutation(len(balanced_indices))
balanced_indices = balanced_indices[perm]
balanced_labels = balanced_labels[perm]

# 3. Stack data and build NaN mask
X1 = H[balanced_indices]
X2 = P[balanced_indices]
X = np.stack([X1, X2], axis=1)  # shape (N, 2, 36, 36)
mask = ~np.isnan(X)
X = np.nan_to_num(X, nan=0.0)

# 4. Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
mask_tensor = torch.tensor(mask.astype(np.float32))
y_tensor = torch.tensor(balanced_labels, dtype=torch.float32)

X_combined = torch.cat([X_tensor, mask_tensor], dim=1)  # shape (N, 4, 36, 36)

# 5. Train/val split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_combined, y_tensor, test_size=0.2, random_state=42)

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

# 6. Define model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # shape (B, 32, 1, 1)
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 7. Train the model
for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb).squeeze()
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb).squeeze()
            val_loss += loss_fn(pred, yb).item()
    print(f"Epoch {epoch+1}, val loss: {val_loss:.4f}")

#%%
# 8. Evaluate performance
model.eval()
with torch.no_grad():
    y_true = y_val.numpy()
    y_pred_probs = model(X_val).squeeze().numpy()
    y_pred = (y_pred_probs > 0.5).astype(int)

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=['Bad', 'Good']))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(4, 4))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xticks([0, 1], ['Bad', 'Good'])
plt.yticks([0, 1], ['Bad', 'Good'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

#%%
# 9. Visualize example predictions
def show_examples(X, y_true, y_pred, title, n=5):
    idxs = np.where(y_true == y_pred)[0] if title == "Correct" else np.where(y_true != y_pred)[0]
    chosen = np.random.choice(idxs, size=min(n, len(idxs)), replace=False)
    
    for i, idx in enumerate(chosen):
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(X[idx, 0], cmap='viridis')
        axs[0].set_title('Array 1')
        axs[1].imshow(X[idx, 1], cmap='viridis')
        axs[1].set_title('Array 2')
        fig.suptitle(f"{title} prediction - True: {int(y_true[idx])}, Pred: {int(y_pred[idx])}")
        plt.tight_layout()
        plt.show()

show_examples(X_val.numpy(), y_true, y_pred, title="Correct")
show_examples(X_val.numpy(), y_true, y_pred, title="Incorrect")

#%%

import random

def show_real_data_examples_batched_random(arr1, arr2, model, n_show=10, batch_size=512):
    model.eval()
    
    N = arr1.shape[0]
    good_idxs = []
    bad_idxs = []

    # Go through the data in batches and store all indices
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        X1 = np.nan_to_num(arr1[start:end], nan=0.0).astype(np.float32)
        X2 = np.nan_to_num(arr2[start:end], nan=0.0).astype(np.float32)
        M1 = ~np.isnan(arr1[start:end])
        M2 = ~np.isnan(arr2[start:end])
        
        X = np.stack([X1, X2, M1, M2], axis=1)
        X_tensor = torch.tensor(X)

        with torch.no_grad():
            preds = model(X_tensor).squeeze().numpy()

        preds_binary = (preds > 0.5).astype(int)

        for i, cls in enumerate(preds_binary):
            global_idx = start + i
            if cls == 0:
                bad_idxs.append(global_idx)
            elif cls == 1:
                good_idxs.append(global_idx)

    # Randomly sample from full list of classified indices
    bad_sample = random.sample(bad_idxs, min(n_show, len(bad_idxs)))
    good_sample = random.sample(good_idxs, min(n_show, len(good_idxs)))

    def plot_examples(indices, title):
        fig, axs = plt.subplots(2, len(indices), figsize=(len(indices) * 2, 4))
        fig.suptitle(title, fontsize=14)
        for i, idx in enumerate(indices):
            axs[0, i].imshow(arr1[idx], cmap='viridis')
            axs[0, i].axis('off')
            axs[0, i].set_title("Array 1")

            axs[1, i].imshow(arr2[idx], cmap='viridis')
            axs[1, i].axis('off')
            axs[1, i].set_title("Array 2")
        plt.tight_layout()
        plt.show()

    plot_examples(bad_sample, "Random Predicted Bad Examples")
    plot_examples(good_sample, "Random Predicted Good Examples")

show_real_data_examples_batched_random(H, P, model, n_show=10)

