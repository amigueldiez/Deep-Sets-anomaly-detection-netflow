import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time


# ----------------------------
# Config
# ----------------------------

# Flows per bag (fixed size for Deep Sets)
FLOWS_PER_BAG = 1000

# Deep Sets parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42  # Seed for dataset creation (fixed)
BAG_SIZE = 1000
BATCH_SIZE = 128
EPOCHS = 94
LR = 1e-3
PATIENCE = 7
TOPK = 20
NUM_RUNS = 10  # Number of training runs with different seeds


# dataset  = pd.read_csv('NF-UNSW-NB15-v3.csv')
print("Loading dataset...")
time_init = time.time()
dataset = pd.read_csv("NF-CICIDS2018-v3.csv")
print(f"Dataset loaded in {time.time() - time_init:.2f} seconds")
print("Dataset loaded with shape:", dataset.shape)


# Delete null and infinite values from the original dataset
print("Cleaning dataset...")
print("Deleting null and infinite values...")
dataset = dataset.replace([float("inf"), float("-inf")], pd.NA).dropna()


# Create bags
dataset["FLOW_START_MILLISECONDS"] = pd.to_datetime(
    dataset["FLOW_START_MILLISECONDS"], unit="ms"
)
dataset["FLOW_END_MILLISECONDS"] = pd.to_datetime(
    dataset["FLOW_END_MILLISECONDS"], unit="ms"
)

dataset["Hour"] = dataset["FLOW_START_MILLISECONDS"].dt.hour
dataset["Day"] = dataset["FLOW_START_MILLISECONDS"].dt.day
dataset["Minute"] = dataset["FLOW_START_MILLISECONDS"].dt.minute
bags = {}


# Sort dataset by time to maintain temporal order
dataset_sorted = dataset.sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)

# Create bags with fixed number of flows
num_bags = len(dataset_sorted) // FLOWS_PER_BAG

for i in range(num_bags):
    start_idx = i * FLOWS_PER_BAG
    end_idx = start_idx + FLOWS_PER_BAG
    bag_data = dataset_sorted.iloc[start_idx:end_idx]

    if not bag_data.empty:
        bag_key = f"Bag_{i:06d}"
        bags[bag_key] = bag_data


print(f"Total bags created: {len(bags)}")


# Filtrar solo las features que existen
available_features = set(dataset.columns)
# Use this features
selected_features = [
    "TCP_WIN_MAX_IN",
    "L4_DST_PORT",
    "TCP_WIN_MAX_OUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "L7_PROTO",
    "TCP_FLAGS",
    "NUM_PKTS_UP_TO_128_BYTES",
    "MAX_IP_PKT_LEN",
    "IN_BYTES",
    "IN_PKTS",
    "LONGEST_FLOW_PKT",
    "SERVER_TCP_FLAGS",
    "OUT_PKTS",
    "MAX_TTL",
    "OUT_BYTES",
]
# # Drop MAX_TTL and MIN_TTL because they have a strong correlation with Label
# selected_features = [f for f in selected_features if f not in ['MAX_TTL', 'MIN_TTL']]
selected_features.append("Label")


print(
    f"\nFeatures seleccionadas finales: {len(selected_features)-1} (sin contar Label)"
)


# Use only selected features plus the label in the bags
for bag_key in bags:
    bags[bag_key] = bags[bag_key][selected_features]


# %% [markdown]
# ## Deep Sets


feature_cols = [c for c in selected_features if c != "Label"]
D = len(feature_cols)

# Set seed for dataset creation only (this will remain fixed)
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# %%
# ----------------------------
# 1) dict -> tensor (filtrando tamaño fijo)
# ----------------------------

X_list, y_list = [], []
for k, df in bags.items():
    if len(df) != BAG_SIZE:
        continue
    X_list.append(df[feature_cols].to_numpy(dtype=np.float32))
    y_list.append(int(df["Label"].max()))

X = np.stack(X_list, axis=0)  # (BAGS, 1000, D)
y = np.array(y_list, dtype=np.float32)  # float para BCE

print("X:", X.shape, "y:", y.shape, "pos:", int(y.sum()))

# %%
# Print number of benign and malicious samples
print("Benign samples:", int((y == 0).sum()))
print("Malicious samples:", int((y == 1).sum()))

# %%
# ----------------------------
# 2) split train/val/test por bag (estratificado)
# ----------------------------

idx = np.arange(len(y))
idx_tr, idx_te = train_test_split(
    idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
idx_tr, idx_va = train_test_split(
    idx_tr, test_size=0.20, random_state=RANDOM_STATE, stratify=y[idx_tr]
)  # 64/16/20

Xtr, Xva, Xte = X[idx_tr], X[idx_va], X[idx_te]
ytr, yva, yte = y[idx_tr], y[idx_va], y[idx_te]


train_counts = [len(ytr) - int(ytr.sum()), int(ytr.sum())]
val_counts = [len(yva) - int(yva.sum()), int(yva.sum())]
test_counts = [len(yte) - int(yte.sum()), int(yte.sum())]

# Print numbers Train, Validation, Test
print(
    f"Train samples: {len(ytr)} (Benign: {train_counts[0]}, Malicious: {train_counts[1]})"
)
print(
    f"Validation samples: {len(yva)} (Benign: {val_counts[0]}, Malicious: {val_counts[1]})"
)
print(
    f"Test samples: {len(yte)} (Benign: {test_counts[0]}, Malicious: {test_counts[1]})"
)

# %%
# ----------------------------
# 3) normalización por feature (fit solo en train)
# ----------------------------
mu = Xtr.reshape(-1, D).mean(axis=0)
sigma = Xtr.reshape(-1, D).std(axis=0) + 1e-8

Xtr = (Xtr - mu) / sigma
Xva = (Xva - mu) / sigma
Xte = (Xte - mu) / sigma


# %%
# ----------------------------
# 4) Dataset
# ----------------------------
class BagDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# %% [markdown]
# ### DeepSets with top-k pooling

# %%
# ----------------------------
# 5) DeepSets con top-k pooling
# ----------------------------


class DeepSetsTopK(nn.Module):
    def __init__(self, d_in, d_hidden=128, d_latent=64, dropout=0.15, topk=10):
        super().__init__()
        self.topk = topk
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent),
            nn.ReLU(),
        )
        # usamos mean + topk_mean (2*latent)
        self.rho = nn.Sequential(
            nn.Linear(2 * d_latent, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),  # logit binario
        )

    def forward(self, x):
        # x: (B, N, D)
        z = self.phi(x)  # (B, N, L)
        z_mean = z.mean(dim=1)  # (B, L)

        k = min(self.topk, z.size(1))
        # top-k por dimension latente (MIL-friendly)
        z_topk, _ = torch.topk(z, k=k, dim=1, largest=True, sorted=False)  # (B, k, L)
        z_topk_mean = z_topk.mean(dim=1)  # (B, L)

        z_bag = torch.cat([z_mean, z_topk_mean], dim=1)  # (B, 2L)
        logit = self.rho(z_bag).squeeze(1)  # (B,)
        return logit


class DeepSets(nn.Module):
    def __init__(self, d_in, d_hidden=128, d_latent=64, dropout=0.15):
        super().__init__()

        self.phi = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent),
            nn.ReLU(),
        )

        # solo usamos la media → dimensión d_latent
        self.rho = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),  # logit binario
        )

    def forward(self, x):
        # x: (B, N, D)
        z = self.phi(x)  # (B, N, L)
        z_mean = z.mean(dim=1)  # (B, L)

        logit = self.rho(z_mean).squeeze(1)  # (B,)
        return logit


# %%
# ----------------------------
# 6) Multiple runs with different training seeds
# ----------------------------

# Store results from all runs
all_precisions_class0 = []
all_recalls_class0 = []
all_f1_scores_class0 = []
all_supports_class0 = []

all_precisions_class1 = []
all_recalls_class1 = []
all_f1_scores_class1 = []
all_supports_class1 = []

all_accuracies = []
all_confusion_matrices = []
all_training_times = []
all_prediction_times = []

print(f"\n{'='*60}")
print(f"Starting {NUM_RUNS} training runs with different seeds")
print(f"{'='*60}\n")

for run in range(1, NUM_RUNS + 1):
    print(f"\n{'='*60}")
    print(f"RUN {run}/{NUM_RUNS} - Training with seed {RANDOM_STATE + run}")
    print(f"{'='*60}\n")

    # Set different seed for this training run
    train_seed = RANDOM_STATE + run
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)

    # Create fresh data loaders with the new seed
    train_loader = DataLoader(
        BagDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        BagDataset(Xva, yva), batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        BagDataset(Xte, yte), batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    # Create new model
    model = DeepSetsTopK(d_in=D, d_hidden=128, d_latent=64, dropout=0.15, topk=TOPK).to(
        DEVICE
    )
    # model = DeepSets(d_in=D, d_hidden=128, d_latent=64, dropout=0.15).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Evaluation function
    def eval_loader(loader):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                logit = model(xb)
                prob = torch.sigmoid(logit).cpu().numpy()
                ys.append(yb.numpy())
                ps.append(prob)
        y_true = np.concatenate(ys).astype(int)
        prob = np.concatenate(ps)
        y_pred = (prob >= 0.5).astype(int)
        return y_true, y_pred, prob

    # Training loop
    training_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logit = model(xb)
            loss = criterion(logit, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        yv_true, yv_pred, _ = eval_loader(val_loader)
        f1v = f1_score(yv_true, yv_pred)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | val F1 {f1v:.4f}")

    training_end_time = time.time()
    training_time = training_end_time - training_start_time

    # Test evaluation
    prediction_start_time = time.time()
    yt_true, yt_pred, yt_prob = eval_loader(test_loader)
    prediction_end_time = time.time()
    prediction_time = prediction_end_time - prediction_start_time

    # Calculate metrics per class
    precision_per_class, recall_per_class, f1_per_class, support_per_class = (
        precision_recall_fscore_support(yt_true, yt_pred, average=None, zero_division=0)
    )
    accuracy = (yt_true == yt_pred).mean()
    cm = confusion_matrix(yt_true, yt_pred)

    # Store results
    all_precisions_class0.append(precision_per_class[0])
    all_recalls_class0.append(recall_per_class[0])
    all_f1_scores_class0.append(f1_per_class[0])
    all_supports_class0.append(support_per_class[0])

    all_precisions_class1.append(precision_per_class[1])
    all_recalls_class1.append(recall_per_class[1])
    all_f1_scores_class1.append(f1_per_class[1])
    all_supports_class1.append(support_per_class[1])

    all_accuracies.append(accuracy)
    all_confusion_matrices.append(cm)
    all_training_times.append(training_time)
    all_prediction_times.append(prediction_time)

    print(f"\nRun {run} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(
        f"  Class 0 - Precision: {precision_per_class[0]:.4f}, Recall: {recall_per_class[0]:.4f}, F1: {f1_per_class[0]:.4f}"
    )
    print(
        f"  Class 1 - Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}"
    )
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Prediction time: {prediction_time:.4f} seconds")
    print(f"  Confusion Matrix:\n{cm}")


# %%
# ----------------------------
# 7) Average results across all runs
# ----------------------------
print(f"\n\n{'='*60}")
print(f"AVERAGE RESULTS OVER {NUM_RUNS} RUNS")
print(f"{'='*60}\n")

# Calculate means and stds for each class
mean_precision_class0 = np.mean(all_precisions_class0)
std_precision_class0 = np.std(all_precisions_class0)
mean_recall_class0 = np.mean(all_recalls_class0)
std_recall_class0 = np.std(all_recalls_class0)
mean_f1_class0 = np.mean(all_f1_scores_class0)
std_f1_class0 = np.std(all_f1_scores_class0)
mean_support_class0 = np.mean(all_supports_class0)

mean_precision_class1 = np.mean(all_precisions_class1)
std_precision_class1 = np.std(all_precisions_class1)
mean_recall_class1 = np.mean(all_recalls_class1)
std_recall_class1 = np.std(all_recalls_class1)
mean_f1_class1 = np.mean(all_f1_scores_class1)
std_f1_class1 = np.std(all_f1_scores_class1)
mean_support_class1 = np.mean(all_supports_class1)

mean_accuracy = np.mean(all_accuracies)
std_accuracy = np.std(all_accuracies)

# Calculate macro averages
mean_precision_macro = (mean_precision_class0 + mean_precision_class1) / 2
mean_recall_macro = (mean_recall_class0 + mean_recall_class1) / 2
mean_f1_macro = (mean_f1_class0 + mean_f1_class1) / 2

# Calculate weighted averages
total_support = mean_support_class0 + mean_support_class1
mean_precision_weighted = (
    mean_precision_class0 * mean_support_class0
    + mean_precision_class1 * mean_support_class1
) / total_support
mean_recall_weighted = (
    mean_recall_class0 * mean_support_class0 + mean_recall_class1 * mean_support_class1
) / total_support
mean_f1_weighted = (
    mean_f1_class0 * mean_support_class0 + mean_f1_class1 * mean_support_class1
) / total_support

# Print classification report style table
print("Classification Report (Average over all runs):\n")
print(
    f"{'Class':<15} {'Precision':<20} {'Recall':<20} {'F1-Score':<20} {'Support':<10}"
)
print("-" * 85)
print(
    f"{'0':<15} {mean_precision_class0:.4f} ± {std_precision_class0:.4f}    {mean_recall_class0:.4f} ± {std_recall_class0:.4f}    {mean_f1_class0:.4f} ± {std_f1_class0:.4f}    {mean_support_class0:.0f}"
)
print(
    f"{'1':<15} {mean_precision_class1:.4f} ± {std_precision_class1:.4f}    {mean_recall_class1:.4f} ± {std_recall_class1:.4f}    {mean_f1_class1:.4f} ± {std_f1_class1:.4f}    {mean_support_class1:.0f}"
)
print("-" * 85)
print(
    f"{'Accuracy':<15} {'':<20} {'':<20} {mean_accuracy:.4f} ± {std_accuracy:.4f}    {total_support:.0f}"
)
print(
    f"{'Macro avg':<15} {mean_precision_macro:.4f}             {mean_recall_macro:.4f}             {mean_f1_macro:.4f}             {total_support:.0f}"
)
print(
    f"{'Weighted avg':<15} {mean_precision_weighted:.4f}             {mean_recall_weighted:.4f}             {mean_f1_weighted:.4f}             {total_support:.0f}"
)

mean_training_time = np.mean(all_training_times)
std_training_time = np.std(all_training_times)
mean_prediction_time = np.mean(all_prediction_times)
std_prediction_time = np.std(all_prediction_times)

print(f"\nTraining time:   {mean_training_time:.2f} ± {std_training_time:.2f} seconds")
print(
    f"Prediction time: {mean_prediction_time:.4f} ± {std_prediction_time:.4f} seconds"
)

# Average confusion matrix
mean_cm = np.mean(all_confusion_matrices, axis=0)
print(f"\nAverage Confusion Matrix:")
print(mean_cm)
print(f"\nConfusion Matrix (rounded):")
print(np.round(mean_cm).astype(int))

print(f"\n{'='*60}\n")
