import os
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = "data/processed/ai4i2020.csv"
OUTPUT_DIR = "data/processed"


def preprocess_data():

    print("Starting data preprocessing...")

    # Load data
    df = pd.read_csv(INPUT_PATH)

    print(f"Original shape: {df.shape}")

    # Remove unnecessary columns
    columns_to_drop = [
        "UDI",
        "Product ID"
    ]

    df = df.drop(columns=columns_to_drop)

    # Separate features and target
    X = df.drop("Machine failure", axis=1)
    y = df["Machine failure"]

    # Convert categorical column into numerical columns
    X = pd.get_dummies(X, columns=["Type"], drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save datasets
    X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

    print("Preprocessing completed successfully!")
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")


if __name__ == "__main__":
    preprocess_data()