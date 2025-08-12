import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

df = pd.read_csv('data/DS4.csv')

# Replace Inf/-Inf with NaN, then drop all rows with any NaN
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

# Target and features
target_col = 'delta_V'
exclude_cols = [
    'Unnamed: 0',
    'framework',
    'charged_id',
    'charged_formula',
    'discharged_id',
    'discharged_formula',
    'active_metals',
    'composition',
    target_col
]

# Select features (drop non-features)
X = df.drop(columns=exclude_cols)
y = df[target_col].values.reshape(-1, 1)

# Convert to numpy float32
X = X.astype(np.float32)
y = y.astype(np.float32)

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features with StandardScaler (fit on train only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to torch tensors
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test)

# Dataset and dataloader
train_ds = TensorDataset(X_train_t, y_train_t)
test_ds = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# Define the model
class VoltageRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 360),
            nn.ReLU(),
            nn.Linear(360, 90),
            nn.ReLU(),
            nn.Linear(90, 30),
            nn.ReLU(),
            nn.Linear(30, 1)
        )
    def forward(self, x):
        return self.model(x)

# Initialize model and send to device
input_dim = X_train.shape[1]
model = VoltageRegressor(input_dim).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
mae_criterion = nn.L1Loss()  # MAE for monitoring
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

epochs = 100
patience = 20  # early stopping patience
best_val_loss = float('inf')
epochs_no_improve = 0

train_losses = []
test_losses = []
train_maes = []
test_maes = []

for epoch in tqdm(range(epochs), desc="Epochs"):
    model.train()
    running_loss = 0
    running_mae = 0

    for xb, yb in tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False):
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        mae = mae_criterion(preds, yb)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        running_mae += mae.item() * xb.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_mae = running_mae / len(train_loader.dataset)

    model.eval()
    val_loss_accum = 0
    val_mae_accum = 0

    with torch.no_grad():
        for xb_test, yb_test in tqdm(test_loader, desc="Validation", leave=False):
            xb_test = xb_test.to(device)
            yb_test = yb_test.to(device)
            val_preds = model(xb_test)
            val_loss_accum += criterion(val_preds, yb_test).item() * xb_test.size(0)
            val_mae_accum += mae_criterion(val_preds, yb_test).item() * xb_test.size(0)

    val_loss = val_loss_accum / len(test_loader.dataset)
    val_mae = val_mae_accum / len(test_loader.dataset)

    train_losses.append(epoch_loss)
    test_losses.append(val_loss)
    train_maes.append(epoch_mae)
    test_maes.append(val_mae)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'models/DNN1.pth')  # save best model
    else:
        epochs_no_improve += 1

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:03d}: "
            f"Train Loss = {epoch_loss:.4f}, Test Loss = {val_loss:.4f}, "
            f"Train MAE = {epoch_mae:.4f}, Test MAE = {val_mae:.4f}"
        )

    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Save final training history for visualization
history_df = pd.DataFrame({
    'epoch': list(range(1, len(train_losses) + 1)),
    'train_loss': train_losses,
    'test_loss': test_losses,
    'train_mae': train_maes,
    'test_mae': test_maes
})
history_df.to_csv('models/training_history.csv', index=False)

print("Training complete. Best model saved as 'models/DNN1.pth'.")
