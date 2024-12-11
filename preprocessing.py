import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def preprocess_data(data, steps):
    """
    Apply preprocessing steps to the dataset.

    Args:
        data (pd.DataFrame): Input dataset.
        steps (list): List of preprocessing steps to apply.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    logical_order = [
        'remove_missing_values',
        'convert_to_datetime',
        'normalize_column',
        'standardize_column',
    ]

    for step in logical_order:
        if step in steps:
            if step == 'remove_missing_values':
                data = data.dropna()
            elif step == 'convert_to_datetime':
                for column in steps.get(step, []):
                    data[column] = pd.to_datetime(
                        data[column], errors='coerce')
            elif step == 'normalize_column':
                for column in steps.get(step, []):
                    data[column] = MinMaxScaler().fit_transform(data[[column]])
            elif step == 'standardize_column':
                for column in steps.get(step, []):
                    data[column] = StandardScaler(
                    ).fit_transform(data[[column]])

    return data
