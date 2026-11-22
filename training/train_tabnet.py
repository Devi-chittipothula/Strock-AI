
import os
import numpy as np
import torch

from pytorch_tabnet.tab_model import TabNetClassifier

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

MODEL_PATH = os.path.join(MODEL_DIR, "tabnet_model")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "tabnet_threshold.npy")

print("=" * 60)
print("             STROKE AI - TABNET TRAINING")
print("=" * 60)

# ==========================================
# 2. LOAD PREPROCESSED DATA
# ==========================================

print("\nLoading preprocessed data...")

X_train = np.load(
    os.path.join(MODEL_DIR, "X_train.npy")
).astype(np.float32)

X_val = np.load(
    os.path.join(MODEL_DIR, "X_val.npy")
).astype(np.float32)

X_test = np.load(
    os.path.join(MODEL_DIR, "X_test.npy")
).astype(np.float32)

y_train = np.load(
    os.path.join(MODEL_DIR, "y_train.npy")
).ravel().astype(np.int64)

y_val = np.load(
    os.path.join(MODEL_DIR, "y_val.npy")
).ravel().astype(np.int64)

y_test = np.load(
    os.path.join(MODEL_DIR, "y_test.npy")
).ravel().astype(np.int64)

print("Data loaded successfully!")

print("\nDataset Shapes:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nTraining Class Distribution:")
print("No Stroke:", np.sum(y_train == 0))
print("Stroke:", np.sum(y_train == 1))

# Check both classes exist
if len(np.unique(y_train)) < 2:
    raise ValueError("Training data must contain both classes (0 and 1).")

if len(np.unique(y_val)) < 2:
    raise ValueError("Validation data must contain both classes.")

if len(np.unique(y_test)) < 2:
    raise ValueError("Test data must contain both classes.")

# ==========================================
# 3. CREATE TABNET MODEL
# ==========================================

print("\nCreating TabNet model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training device:", device)

model = TabNetClassifier(
    n_d=16,
    n_a=16,
    n_steps=3,
    gamma=1.5,
    lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=0.02),
    seed=42,
    verbose=1,
    device_name=device
)

# ==========================================
# 4. TRAIN TABNET MODEL
# ==========================================

print("\nStarting TabNet training...")
print("Class balancing enabled.")

model.fit(
    X_train=X_train,
    y_train=y_train,

    eval_set=[
        (X_train, y_train),
        (X_val, y_val)
    ],

    eval_name=[
        "training",
        "validation"
    ],

    eval_metric=["auc"],

    max_epochs=50,
    patience=10,

    batch_size=256,
    virtual_batch_size=128,

    num_workers=0,
    drop_last=False,

    # Automatic class balancing
    weights=1
)

print("\nTabNet training completed!")

# ==========================================
# 5. FIND BEST THRESHOLD USING VALIDATION DATA
# ==========================================

print("\nSelecting best classification threshold...")

y_val_prob = model.predict_proba(X_val)[:, 1]

best_threshold = 0.50
best_f1 = -1
best_recall = 0

# Prefer recall >= 80%, then maximize F1
for threshold in np.arange(0.10, 0.91, 0.01):

    val_pred = (y_val_prob >= threshold).astype(int)

    val_precision = precision_score(
        y_val,
        val_pred,
        zero_division=0
    )

    val_recall = recall_score(
        y_val,
        val_pred,
        zero_division=0
    )

    val_f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    if val_recall >= 0.80 and val_f1 > best_f1:
        best_f1 = val_f1
        best_threshold = threshold
        best_recall = val_recall

# If no threshold reaches 80% recall, choose highest F1
if best_f1 < 0:

    print("80% validation recall not achieved.")
    print("Selecting threshold with best validation F1.")

    for threshold in np.arange(0.10, 0.91, 0.01):

        val_pred = (y_val_prob >= threshold).astype(int)

        val_f1 = f1_score(
            y_val,
            val_pred,
            zero_division=0
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_threshold = threshold

print(f"\nSelected Threshold: {best_threshold:.2f}")
print(f"Validation F1 Score: {best_f1:.4f}")

# Save threshold for future predictions
np.save(THRESHOLD_PATH, np.array(best_threshold))

print("Threshold saved:", THRESHOLD_PATH)

# ==========================================
# 6. EVALUATE ON TEST DATA
# ==========================================

print("\nEvaluating TabNet model on test data...")

y_test_prob = model.predict_proba(X_test)[:, 1]

# Use validation-selected threshold
y_test_pred = (
    y_test_prob >= best_threshold
).astype(int)

accuracy = accuracy_score(y_test, y_test_pred)

precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(y_test, y_test_prob)

# ==========================================
# 7. DISPLAY PERFORMANCE
# ==========================================

print("\n" + "=" * 60)
print("             TABNET TEST PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=["No Stroke", "Stroke"],
        zero_division=0
    )
)

# ==========================================
# 8. SAVE TRAINED MODEL
# ==========================================

print("\nSaving TabNet model...")

model.save_model(MODEL_PATH)

print("\n" + "=" * 60)
print("       TABNET MODEL TRAINED SUCCESSFULLY!")
print("=" * 60)

print("Model saved at:", MODEL_PATH + ".zip")
print("Threshold saved at:", THRESHOLD_PATH)

print("\nNext step: Compare all trained models.")
