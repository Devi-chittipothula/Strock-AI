
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
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

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 55)
print("          StrokeAI - DNN MODEL TRAINING")
print("=" * 55)

# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLoading preprocessed data...")

X_train = np.load(os.path.join(MODEL_DIR, "X_train.npy")).astype(np.float32)
X_val = np.load(os.path.join(MODEL_DIR, "X_val.npy")).astype(np.float32)
X_test = np.load(os.path.join(MODEL_DIR, "X_test.npy")).astype(np.float32)

y_train = np.load(os.path.join(MODEL_DIR, "y_train.npy")).ravel().astype(np.int32)
y_val = np.load(os.path.join(MODEL_DIR, "y_val.npy")).ravel().astype(np.int32)
y_test = np.load(os.path.join(MODEL_DIR, "y_test.npy")).ravel().astype(np.int32)

print("Data loaded successfully!")

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Testing shape:", X_test.shape)

# Check dataset classes
print("\nTraining class distribution:")
print("No Stroke:", np.sum(y_train == 0))
print("Stroke:", np.sum(y_train == 1))

# ==========================================
# 3. CLASS WEIGHT BALANCING
# ==========================================

print("\nCalculating class weights...")

classes = np.unique(y_train)

if not np.array_equal(classes, np.array([0, 1])):
    raise ValueError(
        "Training data must contain both classes: 0 and 1."
    )

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(c): float(w)
    for c, w in zip(classes, weights)
}

print("Class weights:", class_weights)

# ==========================================
# 4. BUILD DNN MODEL
# ==========================================

print("\nBuilding DNN architecture...")

model = Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc")
    ]
)

model.summary()

# ==========================================
# 5. CALLBACKS
# ==========================================

early_stop = EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# ==========================================
# 6. TRAIN DNN
# ==========================================

print("\nStarting DNN training...")

history = model.fit(
    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=50,
    batch_size=32,

    class_weight=class_weights,

    callbacks=[
        early_stop,
        reduce_lr
    ],

    verbose=1
)

# ==========================================
# 7. VALIDATION THRESHOLD SELECTION
# ==========================================

print("\nSelecting threshold using validation data...")

val_probabilities = model.predict(
    X_val,
    verbose=0
).ravel()

thresholds = np.arange(0.10, 0.91, 0.01)

best_threshold = 0.50
best_f1 = -1

for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    current_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    current_recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    # Prefer high recall while improving F1.
    if current_recall >= 0.80 and current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = float(threshold)

if best_f1 < 0:
    print(
        "No threshold achieved 80% validation recall. "
        "Using the threshold with the best validation F1."
    )

    best_f1 = -1

    for threshold in thresholds:

        val_predictions = (
            val_probabilities >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_val,
            val_predictions,
            zero_division=0
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)

print("Selected threshold:", round(best_threshold, 2))
print("Validation F1:", round(best_f1, 4))

# ==========================================
# 8. TEST EVALUATION
# ==========================================

print("\nEvaluating DNN on test data...")

test_probabilities = model.predict(
    X_test,
    verbose=0
).ravel()

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    test_probabilities
)

print("\n========== DNN PERFORMANCE ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["No Stroke", "Stroke"],
        zero_division=0
    )
)

# ==========================================
# 9. SAVE MODEL AND THRESHOLD
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "dnn_model.keras"
)

model.save(model_path)

threshold_path = os.path.join(
    MODEL_DIR,
    "dnn_threshold.npy"
)

np.save(
    threshold_path,
    np.array([best_threshold])
)

# ==========================================
# 10. SAVE TRAINING HISTORY
# ==========================================

history_path = os.path.join(
    MODEL_DIR,
    "dnn_training_history.csv"
)

import pandas as pd

pd.DataFrame(history.history).to_csv(
    history_path,
    index=False
)

print("\n" + "=" * 55)
print("DNN TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 55)

print("Model saved:", model_path)
print("Threshold saved:", threshold_path)
print("Training history saved:", history_path)
