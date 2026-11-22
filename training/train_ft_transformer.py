
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("       StrokeAI - FT TRANSFORMER TRAINING")
print("=" * 60)

print("Using device:", device)

# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLoading preprocessed data...")

X_train = np.load(os.path.join(MODEL_DIR, "X_train.npy")).astype(np.float32)
X_val = np.load(os.path.join(MODEL_DIR, "X_val.npy")).astype(np.float32)
X_test = np.load(os.path.join(MODEL_DIR, "X_test.npy")).astype(np.float32)

y_train = np.load(os.path.join(MODEL_DIR, "y_train.npy")).ravel().astype(np.float32)
y_val = np.load(os.path.join(MODEL_DIR, "y_val.npy")).ravel().astype(np.float32)
y_test = np.load(os.path.join(MODEL_DIR, "y_test.npy")).ravel().astype(np.float32)

print("Data loaded successfully!")
print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Testing shape:", X_test.shape)

print("\nTraining class distribution:")
print("No Stroke:", np.sum(y_train == 0))
print("Stroke:", np.sum(y_train == 1))

if len(np.unique(y_train)) != 2:
    raise ValueError("Training data must contain both classes.")

if len(np.unique(y_val)) != 2 or len(np.unique(y_test)) != 2:
    raise ValueError("Validation and test data must contain both classes.")

# ==========================================
# 3. FT TRANSFORMER ARCHITECTURE
# ==========================================

class FTTransformer(nn.Module):

    def __init__(self, num_features, d_model=32, num_heads=4):

        super().__init__()

        self.feature_weight = nn.Parameter(
            torch.randn(num_features, d_model) * 0.02
        )

        self.feature_bias = nn.Parameter(
            torch.zeros(num_features, d_model)
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=64,
            dropout=0.2,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        tokens = (
            x.unsqueeze(-1) * self.feature_weight
            + self.feature_bias
        )

        cls = self.cls_token.expand(
            x.size(0), -1, -1
        )

        tokens = torch.cat(
            [cls, tokens], dim=1
        )

        encoded = self.transformer(tokens)

        cls_output = self.norm(encoded[:, 0, :])

        return self.classifier(
            cls_output
        ).squeeze(1)


# ==========================================
# 4. CREATE DATA LOADERS
# ==========================================

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)

val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32)
)

test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256
)

# ==========================================
# 5. CREATE MODEL
# ==========================================

model = FTTransformer(
    num_features=X_train.shape[1],
    d_model=32,
    num_heads=4
).to(device)

# Class imbalance handling
positive_count = float(np.sum(y_train == 1))
negative_count = float(np.sum(y_train == 0))

pos_weight = torch.tensor(
    [negative_count / positive_count],
    dtype=torch.float32
).to(device)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

print("\nPositive class weight:", pos_weight.item())

# ==========================================
# 6. VALIDATION FUNCTION
# ==========================================

def get_probabilities(loader):

    model.eval()

    probabilities = []
    targets = []

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(device)

            outputs = model(features)

            probs = torch.sigmoid(outputs)

            probabilities.extend(
                probs.cpu().numpy()
            )

            targets.extend(
                labels.numpy()
            )

    return np.array(probabilities), np.array(targets)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\nStarting FT-Transformer training...")

best_auc = -1
best_weights = None

patience = 8
wait = 0
epochs = 50

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for features, labels in train_loader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

    val_probs, val_targets = get_probabilities(
        val_loader
    )

    val_auc = roc_auc_score(
        val_targets,
        val_probs
    )

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {avg_loss:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )

    if val_auc > best_auc:

        best_auc = val_auc

        best_weights = copy.deepcopy(
            model.state_dict()
        )

        wait = 0

    else:

        wait += 1

    if wait >= patience:

        print("Early stopping triggered!")
        break

if best_weights is None:
    raise RuntimeError("Model training failed.")

model.load_state_dict(best_weights)

# ==========================================
# 8. SELECT THRESHOLD ON VALIDATION DATA
# ==========================================

print("\nSelecting classification threshold...")

val_probs, val_targets = get_probabilities(
    val_loader
)

thresholds = np.arange(0.10, 0.91, 0.01)

best_threshold = 0.50
best_f1 = -1

for threshold in thresholds:

    predictions = (
        val_probs >= threshold
    ).astype(int)

    current_recall = recall_score(
        val_targets,
        predictions,
        zero_division=0
    )

    current_f1 = f1_score(
        val_targets,
        predictions,
        zero_division=0
    )

    if current_recall >= 0.80 and current_f1 > best_f1:

        best_f1 = current_f1
        best_threshold = float(threshold)

if best_f1 < 0:

    print("80% validation recall not achieved.")
    print("Selecting best validation F1 threshold.")

    for threshold in thresholds:

        predictions = (
            val_probs >= threshold
        ).astype(int)

        current_f1 = f1_score(
            val_targets,
            predictions,
            zero_division=0
        )

        if current_f1 > best_f1:

            best_f1 = current_f1
            best_threshold = float(threshold)

print("Selected threshold:", round(best_threshold, 2))
print("Validation F1:", round(best_f1, 4))

# ==========================================
# 9. TEST EVALUATION
# ==========================================

print("\nEvaluating FT-Transformer...")

test_probs, test_targets = get_probabilities(
    test_loader
)

test_predictions = (
    test_probs >= best_threshold
).astype(int)

accuracy = accuracy_score(
    test_targets,
    test_predictions
)

precision = precision_score(
    test_targets,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    test_targets,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    test_targets,
    test_predictions,
    zero_division=0
)

auc = roc_auc_score(
    test_targets,
    test_probs
)

print("\n========== FT-TRANSFORMER PERFORMANCE ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(test_targets, test_predictions))

print("\nClassification Report:")

print(
    classification_report(
        test_targets,
        test_predictions,
        target_names=["No Stroke", "Stroke"],
        zero_division=0
    )
)

# ==========================================
# 10. SAVE MODEL
# ==========================================

save_path = os.path.join(
    MODEL_DIR,
    "ft_transformer_model.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "num_features": X_train.shape[1],
        "d_model": 32,
        "num_heads": 4
    },
    save_path
)

# Save selected threshold
threshold_path = os.path.join(
    MODEL_DIR,
    "ft_transformer_threshold.npy"
)

np.save(
    threshold_path,
    np.array([best_threshold])
)

print("\n" + "=" * 60)
print("FT-TRANSFORMER TRAINING COMPLETED!")
print("=" * 60)

print("Model saved:", save_path)
print("Threshold saved:", threshold_path)
