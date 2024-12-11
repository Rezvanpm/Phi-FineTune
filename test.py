# import torch
# print(torch.backends.mps.is_available())  # True باید باشد


from transformers import AutoTokenizer, AutoModelForCausalLM

SAVE_PATH = "./models/phi-3.5-mini-instruct"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
model = AutoModelForCausalLM.from_pretrained(
    SAVE_PATH, device_map="auto", offload_folder="./offload")

print("Model loaded successfully!")
