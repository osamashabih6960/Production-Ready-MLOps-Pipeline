import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("data/processed/ai4i2020.csv")
OUTPUT_DIR = Path("data/processed")


def load_data():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {INPUT_PATH}"
        )

    return pd.read_csv(INPUT_PATH)


def clean_data(df):
    # Identifier columns remove
    columns_to_drop = [
        "UDI",
        "Product ID",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ]

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    # Missing values
    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    return df


def prepare_features_target(df):
    # Target
    y = df["Machine failure"]

    # Features
    X = df.drop(
        columns=["Machine failure"]
    )

    # Convert Type:
    # L = 0, M = 1, H = 2
    X["Type"] = X["Type"].map({
        "L": 0,
        "M": 1,
        "H": 2
    })

    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns
    )

    return X_train_scaled, X_test_scaled, scaler


def save_data(
    X_train,
    X_test,
    y_train,
    y_test
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X_train.to_csv(
        OUTPUT_DIR / "X_train.csv",
        index=False
    )

    X_test.to_csv(
        OUTPUT_DIR / "X_test.csv",
        index=False
    )

    y_train.to_csv(
        OUTPUT_DIR / "y_train.csv",
        index=False
    )

    y_test.to_csv(
        OUTPUT_DIR / "y_test.csv",
        index=False
    )

    print("Processed datasets saved successfully.")


def run():
    print("Starting preprocessing...")

    # Load
    df = load_data()

    print(f"Original shape: {df.shape}")

    # Clean
    df = clean_data(df)

    print(f"After cleaning: {df.shape}")

    # Features and target
    X, y = prepare_features_target(df)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Split
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print(f"Training data: {X_train.shape}")
    print(f"Testing data: {X_test.shape}")

    # Scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test
    )

    # Save
    save_data(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    run()