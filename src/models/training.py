import zarr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class ZarrDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[1]

    def __getitem__(self, idx):
        x = self.features[:, idx]
        t = self.targets[idx]
        return x, t


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


def load_features(zarr_path, resolution):
    features_zarr = zarr.open(zarr_path, mode='r')
    features = features_zarr[f'Features/{resolution}']
    scan_mask = features_zarr[f'ScanMask/{resolution}']
    valid_positions = scan_mask[:] == 1
    valid_features = features[:, valid_positions]
    return valid_features


def train_step(trn_dl, model, criterion, optimizer):
    model.train()
    total_loss = 0
    total_samples = 0
    for i, (x, t) in enumerate(trn_dl):
        optimizer.zero_grad()
        x = x.cuda()
        t = t.cuda()
        y_hat = model(x)
        y_hat = torch.squeeze(y_hat)
        loss = criterion(y_hat, t.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_samples += x.size(0)
    return total_loss / total_samples


def validation_step(val_dl, model, criterion):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for i, (x, t) in enumerate(val_dl):
            x = x.cuda()
            t = t.cuda()
            y_hat = model(x)
            loss = criterion(y_hat, t.float())
            total_loss += loss.item()
            total_samples += x.size(0)
    return total_loss / total_samples


def training_loop(
    num_epochs: int,
    model,
    trn_dl,
    val_dl,
    criterion,
    optimizer,
    threshold: float = 1e-4
):
    prior_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_step(trn_dl, model, criterion, optimizer)
        val_loss = validation_step(val_dl, model, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        if abs(prior_loss - val_loss) < threshold:
            print("Early stopping criterion met")
            break
        prior_loss = val_loss


if __name__ == "__main__":
    zarr_path = 'path_to_feature_zarr'
    resolution = '1'
    features = load_features(zarr_path, resolution)
    targets = ...  # Load or generate corresponding age scores for these features

    dataset = ZarrDataset(features, targets)
    trn_dl = DataLoader(dataset, batch_size=32, shuffle=True)
    val_dl = DataLoader(dataset, batch_size=32)

    input_dim = features.shape[0]  # C-dimension
    model = MLP(input_dim=input_dim, hidden_dim=512).cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    training_loop(num_epochs=50, model=model, trn_dl=trn_dl, val_dl=val_dl, criterion=criterion, optimizer=optimizer)
