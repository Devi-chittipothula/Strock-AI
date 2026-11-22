
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
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

print("=" * 65)
print("             StrokeAI - FINAL MODEL COMPARISON")
print("=" * 65)

# ==========================================
# 2. LOAD TEST DATA
# ==========================================

print("\nLoading test data...")

X_test = np.load(
    os.path.join(MODEL_DIR, "X_test.npy")
).astype(np.float32)

y_test = np.load(
    os.path.join(MODEL_DIR, "y_test.npy")
).ravel().astype(int)

print("Test data loaded:", X_test.shape)

print("\nTest class distribution:")
print("No Stroke:", np.sum(y_test == 0))
print("Stroke:", np.sum(y_test == 1))

# ==========================================
# 3. LOAD SAVED THRESHOLDS
# ==========================================

def load_threshold(filename):

    path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Threshold file missing: {path}"
        )

    return float(np.load(path).ravel()[0])


dnn_threshold = load_threshold(
    "dnn_threshold.npy"
)

tabnet_threshold = load_threshold(
    "tabnet_threshold.npy"
)

ft_threshold = load_threshold(
    "ft_transformer_threshold.npy"
)

print("\nSaved Validation Thresholds:")
print("DNN:", dnn_threshold)
print("TabNet:", tabnet_threshold)
print("FT-Transformer:", ft_threshold)

# ==========================================
# 4. FT TRANSFORMER ARCHITECTURE
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
# 5. EVALUATION FUNCTION
# ==========================================

results = []

def evaluate_model(name, probabilities, threshold):

    probabilities = np.asarray(
        probabilities
    ).ravel()

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test, predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    results.append({
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc,
        "True Negatives": int(cm[0, 0]),
        "False Positives": int(cm[0, 1]),
        "False Negatives": int(cm[1, 0]),
        "True Positives": int(cm[1, 1])
    })

    print("\n" + "=" * 60)
    print(name.upper())
    print("=" * 60)

    print("Threshold:", round(threshold, 3))
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Stroke", "Stroke"],
            zero_division=0
        )
    )


# ==========================================
# 6. DNN EVALUATION
# ==========================================

print("\nLoading DNN model...")

dnn_model = load_model(
    os.path.join(MODEL_DIR, "dnn_model.keras")
)

dnn_probabilities = dnn_model.predict(
    X_test,
    verbose=0
).ravel()

evaluate_model(
    "DNN",
    dnn_probabilities,
    dnn_threshold
)


# ==========================================
# 7. TABNET EVALUATION
# ==========================================

print("\nLoading TabNet model...")

tabnet_model = TabNetClassifier()

tabnet_model.load_model(
    os.path.join(MODEL_DIR, "tabnet_model.zip")
)

tabnet_probabilities = tabnet_model.predict_proba(
    X_test
)[:, 1]

evaluate_model(
    "TabNet",
    tabnet_probabilities,
    tabnet_threshold
)


# ==========================================
# 8. FT TRANSFORMER EVALUATION
# ==========================================

print("\nLoading FT-Transformer model...")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    os.path.join(
        MODEL_DIR,
        "ft_transformer_model.pth"
    ),
    map_location=device,
    weights_only=True
)

ft_model = FTTransformer(
    num_features=checkpoint["num_features"],
    d_model=checkpoint["d_model"],
    num_heads=checkpoint["num_heads"]
).to(device)

ft_model.load_state_dict(
    checkpoint["model_state_dict"]
)

ft_model.eval()

with torch.no_grad():

    inputs = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(device)

    outputs = ft_model(inputs)

    ft_probabilities = torch.sigmoid(
        outputs
    ).cpu().numpy()

evaluate_model(
    "FT-Transformer",
    ft_probabilities,
    ft_threshold
)


# ==========================================
# 9. FINAL COMPARISON
# ==========================================

print("\n" + "=" * 85)
print("                    FINAL MODEL COMPARISON")
print("=" * 85)

df = pd.DataFrame(results)

display_columns = [
    "Model",
    "Threshold",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

print(
    df[display_columns].round(4).to_string(index=False)
)

# ==========================================
# 10. SAVE COMPARISON CSV
# ==========================================

csv_path = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)

df.to_csv(
    csv_path,
    index=False
)

print("\nComparison CSV saved:", csv_path)

# ==========================================
# 11. GENERATE COMPARISON GRAPH
# ==========================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

ax = df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 7),
    ylim=(0, 1),
    width=0.75
)

plt.title(
    "StrokeAI - AI Model Performance Comparison"
)

plt.xlabel("AI Models")
plt.ylabel("Performance Score")

plt.xticks(rotation=0)

plt.legend(
    loc="lower right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

graph_path = os.path.join(
    MODEL_DIR,
    "model_comparison.png"
)

plt.savefig(
    graph_path,
    dpi=300
)

plt.close()

print("Comparison graph saved:", graph_path)

# ==========================================
# 12. PROVISIONAL MODEL RANKING
# ==========================================

# ROC-AUC is used for exploratory ranking.
# Final model selection should use independent validation.

best_model = df.loc[
    df["ROC-AUC"].idxmax()
]

best_path = os.path.join(
    MODEL_DIR,
    "best_model.txt"
)

with open(best_path, "w") as file:

    file.write(
        "Provisional Model Ranking (Test ROC-AUC)\n"
    )

    file.write(
        f"Model: {best_model['Model']}\n"
    )

    file.write(
        f"Accuracy: {best_model['Accuracy']:.4f}\n"
    )

    file.write(
        f"Precision: {best_model['Precision']:.4f}\n"
    )

    file.write(
        f"Recall: {best_model['Recall']:.4f}\n"
    )

    file.write(
        f"F1 Score: {best_model['F1 Score']:.4f}\n"
    )

    file.write(
        f"ROC-AUC: {best_model['ROC-AUC']:.4f}\n"
    )

print("\nProvisional highest test ROC-AUC model:")
print(best_model["Model"])

print("\n" + "=" * 65)
print("          MODEL COMPARISON COMPLETED!")
print("=" * 65)

print("\nGenerated files:")
print("1. model_comparison.csv")
print("2. model_comparison.png")
print("3. best_model.txt")

print("\nAll models evaluated using saved validation thresholds.")
