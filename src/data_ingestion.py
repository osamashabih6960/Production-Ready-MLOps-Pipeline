import os
import pandas as pd


RAW_DATA_PATH = "data/raw/ai4i2020.csv"
PROCESSED_DATA_PATH = "data/processed/ai4i2020.csv"


def ingest_data():
    # Check whether raw dataset exists
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}"
        )

    # Read dataset
    df = pd.read_csv(RAW_DATA_PATH)

    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    # Basic validation
    if df.empty:
        raise ValueError("Dataset is empty.")

    # Create processed directory
    os.makedirs("data/processed", exist_ok=True)

    # Save ingested data
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"\nProcessed dataset saved at: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    ingest_data()