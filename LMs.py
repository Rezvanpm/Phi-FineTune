from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import pandas as pd


def load_language_model(model_path="./local_models/phi-3.5-mini-instruct"):
    """
    Load the Phi-3.5-mini-instruct language model and tokenizer from a local directory.
    Args:
        model_path (str): The local path to the Phi model.
    Returns:
        model, tokenizer: The loaded language model and tokenizer.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please ensure the model is downloaded locally.")

    print(f"Loading Phi-3.5-mini-instruct model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", offload_folder="./offload")
    print("Phi model loaded successfully.")
    return model, tokenizer


def process_for_fine_tuning(preprocessed_data, output_dir="./prepared_data"):
    """
    Prepare the preprocessed dataset for fine-tuning.
    Args:
        preprocessed_data (pd.DataFrame): The preprocessed dataset with columns 'Title' and 'Abstract'.
        output_dir (str): Directory to save the prepared dataset.
    Returns:
        str: Path to the prepared dataset ready for fine-tuning.
    """
    if not {"Title", "Abstract"}.issubset(preprocessed_data.columns):
        raise ValueError(
            "The preprocessed dataset must contain 'Title' and 'Abstract' columns.")

    # Combine 'Title' and 'Abstract' columns into a single text column
    preprocessed_data["text"] = "Title: " + preprocessed_data["Title"] + \
        "\nAbstract: " + preprocessed_data["Abstract"]

    # Save prepared dataset to a file
    os.makedirs(output_dir, exist_ok=True)
    prepared_file_path = os.path.join(
        output_dir, "prepared_fine_tuning_data.csv")
    preprocessed_data[["text"]].to_csv(prepared_file_path, index=False)

    print(f"Prepared dataset saved at {prepared_file_path}")
    return prepared_file_path


def list_models():
    """
    List the available language models. Only the Phi model is available for use.
    Returns:
        list: A list of model names with availability status.
    """
    models = [
        {"name": "phi-3.5-mini-instruct", "available": True},
        {"name": "gpt-4", "available": False},
        {"name": "bert", "available": False},
    ]
    return models


def select_model():
    """
    Select a language model. Only Phi is available for actual loading.
    Returns:
        str: The selected model's name.
    """
    models = list_models()
    print("Available models:")
    for model in models:
        status = "Available" if model["available"] else "Display Only"
        print(f"- {model['name']} ({status})")

    # Simulate selection of Phi model
    print("Phi-3.5-mini-instruct is selected as the default model.")
    return "phi-3.5-mini-instruct"
