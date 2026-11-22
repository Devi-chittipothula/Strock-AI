
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "healthcare-dataset-stroke-data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


def prepare_data():

    print("Loading Stroke Dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Original Dataset Shape:", df.shape)

    # Remove unnecessary ID column
    df.drop(columns=["id"], inplace=True, errors="ignore")

    # Separate input and target
    X = df.drop(columns=["stroke"])
    y = df["stroke"].astype(int)

    # Identify column types
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    # Split training and temporary data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Split temporary data into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # Fit only on training data to prevent data leakage
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Convert to dense arrays for model compatibility
    X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
    X_val_processed = np.asarray(X_val_processed, dtype=np.float32)
    X_test_processed = np.asarray(X_test_processed, dtype=np.float32)

    # Save processed data
    np.save(MODEL_DIR / "X_train.npy", X_train_processed)
    np.save(MODEL_DIR / "X_val.npy", X_val_processed)
    np.save(MODEL_DIR / "X_test.npy", X_test_processed)

    np.save(MODEL_DIR / "y_train.npy", y_train.to_numpy())
    np.save(MODEL_DIR / "y_val.npy", y_val.to_numpy())
    np.save(MODEL_DIR / "y_test.npy", y_test.to_numpy())

    joblib.dump(
        preprocessor,
        MODEL_DIR / "preprocessor.joblib"
    )

    # Save feature names
    feature_names = preprocessor.get_feature_names_out()

    joblib.dump(
        feature_names,
        MODEL_DIR / "feature_names.joblib"
    )

    print("\nPreprocessing Completed Successfully!")
    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples:", len(X_test))
    print("Number of processed features:", X_train_processed.shape[1])
    print("\nStroke distribution in training data:")
    print(y_train.value_counts())


if __name__ == "__main__":
    prepare_data()
