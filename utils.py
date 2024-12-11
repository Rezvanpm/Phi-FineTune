import pandas as pd
import os


def load_dataset(dataset_name):
    """
    Load a dataset.

    Args:
        dataset_name (str): Name of the dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    file_path = os.path.join('./datasets', dataset_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found.")
    return pd.read_csv(file_path)
