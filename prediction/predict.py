
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn

from pathlib import Path
from pytorch_tabnet.tab_model import TabNetClassifier


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

DEVICE = torch.device("cpu")


# =========================================================
# INPUT FEATURES
# Must match the original Stroke Dataset column names
# =========================================================

REQUIRED_FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status"
]


# =========================================================
# FT-TRANSFORMER ARCHITECTURE
# Must match training/train_ft_transformer.py
# =========================================================

class FTTransformer(nn.Module):

    def __init__(
        self,
        num_features,
        d_model=32,
        num_heads=4
    ):
        super().__init__()

        self.feature_weight = nn.Parameter(
            torch.randn(num_features, d_model) * 0.02
        )

        self.feature_bias = nn.Parameter(
            torch.zeros(num_features, d_model)
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02
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

        # Convert each numerical feature into a feature token
        tokens = (
            x.unsqueeze(-1) * self.feature_weight
            + self.feature_bias
        )

        batch_size = x.size(0)

        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )

        tokens = torch.cat(
            [cls_tokens, tokens],
            dim=1
        )

        encoded = self.transformer(tokens)

        cls_output = self.norm(encoded[:, 0])

        logits = self.classifier(cls_output)

        return logits.squeeze(-1)


# =========================================================
# STROKE PREDICTION SYSTEM
# =========================================================

class StrokePredictor:

    def __init__(self):

        print("Loading StrokeAI models...")

        # Check preprocessing file
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(
                "Preprocessor not found. "
                "Run preprocessing/prepare_data.py first."
            )

        self.preprocessor = joblib.load(
            PREPROCESSOR_PATH
        )

        # Load feature names
        self.feature_names = joblib.load(
            MODEL_DIR / "feature_names.joblib"
        )

        # Load saved thresholds
        self.dnn_threshold = self.load_threshold(
            "dnn_threshold.npy", 0.5
        )

        self.tabnet_threshold = self.load_threshold(
            "tabnet_threshold.npy", 0.5
        )

        self.ft_threshold = self.load_threshold(
            "ft_transformer_threshold.npy", 0.5
        )

        # Load all models
        self.dnn_model = self.load_dnn()

        self.tabnet_model = self.load_tabnet()

        self.ft_model = self.load_ft_transformer()

        print("All StrokeAI models loaded successfully!")

    # -----------------------------------------------------
    # LOAD THRESHOLD
    # -----------------------------------------------------

    def load_threshold(self, filename, default):

        path = MODEL_DIR / filename

        if path.exists():
            return float(np.load(path).item())

        print(
            f"Warning: {filename} not found. "
            f"Using default threshold {default}."
        )

        return default

    # -----------------------------------------------------
    # LOAD DNN
    # -----------------------------------------------------

    def load_dnn(self):

        path = MODEL_DIR / "dnn_model.keras"

        if not path.exists():
            raise FileNotFoundError(
                f"DNN model not found: {path}"
            )

        return tf.keras.models.load_model(path)

    # -----------------------------------------------------
    # LOAD TABNET
    # -----------------------------------------------------

    def load_tabnet(self):

        path = MODEL_DIR / "tabnet_model.zip"

        if not path.exists():
            raise FileNotFoundError(
                f"TabNet model not found: {path}"
            )

        model = TabNetClassifier()

        model.load_model(str(path))

        return model

    # -----------------------------------------------------
    # LOAD FT-TRANSFORMER
    # -----------------------------------------------------

    def load_ft_transformer(self):

        path = MODEL_DIR / "ft_transformer_model.pth"

        if not path.exists():
            raise FileNotFoundError(
                f"FT-Transformer model not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=True
        )

        num_features = checkpoint["num_features"]

        model = FTTransformer(
            num_features=num_features,
            d_model=checkpoint["d_model"],
            num_heads=checkpoint["num_heads"]
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.to(DEVICE)
        model.eval()

        return model

    # -----------------------------------------------------
    # PREPROCESS PATIENT INPUT
    # -----------------------------------------------------

    def preprocess_input(self, patient_data):

        # Accept dictionary or DataFrame
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame(
                [patient_data]
            )

        elif isinstance(patient_data, pd.DataFrame):
            patient_df = patient_data.copy()

        else:
            raise TypeError(
                "Input must be a dictionary or pandas DataFrame."
            )

        # Validate required columns
        missing_columns = [
            col for col in REQUIRED_FEATURES
            if col not in patient_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing patient information: {missing_columns}"
            )

        # Keep original training feature order
        patient_df = patient_df[REQUIRED_FEATURES]

        # Apply the exact training preprocessing
        processed = self.preprocessor.transform(
            patient_df
        )

        processed = np.asarray(
            processed,
            dtype=np.float32
        )

        if processed.shape[1] != len(self.feature_names):
            raise ValueError(
                "Processed feature count does not match "
                "the saved training feature names."
            )

        if not np.isfinite(processed).all():
            raise ValueError(
                "Processed patient data contains invalid values."
            )

        return processed

    # -----------------------------------------------------
    # DNN PREDICTION
    # -----------------------------------------------------

    def predict_dnn(self, X):

        probability = self.dnn_model.predict(
            X,
            verbose=0
        ).reshape(-1)[0]

        prediction = int(
            probability >= self.dnn_threshold
        )

        return {
            "probability": float(probability),
            "prediction": prediction,
            "threshold": self.dnn_threshold
        }

    # -----------------------------------------------------
    # TABNET PREDICTION
    # -----------------------------------------------------

    def predict_tabnet(self, X):

        probabilities = self.tabnet_model.predict_proba(
            X
        )

        probability = float(
            probabilities[0, 1]
        )

        prediction = int(
            probability >= self.tabnet_threshold
        )

        return {
            "probability": probability,
            "prediction": prediction,
            "threshold": self.tabnet_threshold
        }

    # -----------------------------------------------------
    # FT-TRANSFORMER PREDICTION
    # -----------------------------------------------------

    def predict_ft_transformer(self, X):

        tensor = torch.tensor(
            X,
            dtype=torch.float32,
            device=DEVICE
        )

        self.ft_model.eval()

        with torch.no_grad():

            logits = self.ft_model(tensor)

            probability = torch.sigmoid(
                logits
            ).cpu().numpy().reshape(-1)[0]

        probability = float(probability)

        prediction = int(
            probability >= self.ft_threshold
        )

        return {
            "probability": probability,
            "prediction": prediction,
            "threshold": self.ft_threshold
        }

    # -----------------------------------------------------
    # PREDICT USING ALL MODELS
    # -----------------------------------------------------

    def predict_all(self, patient_data):

        X = self.preprocess_input(
            patient_data
        )

        dnn_result = self.predict_dnn(X)

        tabnet_result = self.predict_tabnet(X)

        ft_result = self.predict_ft_transformer(X)

        results = {
            "DNN": dnn_result,
            "TabNet": tabnet_result,
            "FT-Transformer": ft_result
        }

        # Use FT-Transformer as the current provisional
        # primary model based on the comparison output.
        selected_result = ft_result

        results["selected_model"] = "FT-Transformer"

        results["stroke_probability"] = (
            selected_result["probability"]
        )

        results["stroke_prediction"] = (
            selected_result["prediction"]
        )

        results["risk_status"] = (
            "Higher predicted risk"
            if selected_result["prediction"] == 1
            else "Lower predicted risk"
        )

        return results


# =========================================================
# TEST THE PREDICTION SYSTEM
# =========================================================

if __name__ == "__main__":

    predictor = StrokePredictor()

    # Example patient information
    sample_patient = {
        "gender": "Male",
        "age": 67,
        "hypertension": 1,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly smoked"
    }

    print("\nPredicting stroke risk...\n")

    results = predictor.predict_all(
        sample_patient
    )

    print("=" * 50)
    print("             STROKE AI RESULTS")
    print("=" * 50)

    for model_name in [
        "DNN",
        "TabNet",
        "FT-Transformer"
    ]:

        result = results[model_name]

        print(f"\n{model_name}")

        print(
            "Probability:",
            round(result["probability"] * 100, 2),
            "%"
        )

        print(
            "Prediction:",
            "Stroke" if result["prediction"] == 1
            else "No Stroke"
        )

        print(
            "Threshold:",
            result["threshold"]
        )

    print("\nSelected Model:", results["selected_model"])

    print(
        "Predicted Risk:",
        results["risk_status"]
    )

    print("=" * 50)

    print(
        "\nNote: This is an experimental prediction, "
        "not a medical diagnosis."
    )
