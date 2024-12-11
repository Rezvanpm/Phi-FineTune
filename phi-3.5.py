from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

SAVE_PATH = "./models/phi-3.5-mini-instruct"

# بارگذاری مدل و توکنایزر
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)
model = AutoModelForCausalLM.from_pretrained(
    SAVE_PATH, device_map="auto")

# شناسایی دستگاه مدل
device = next(model.parameters()).device
print(f"Model loaded on device: {device}")

# تولید پاسخ آزمایشی
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

# انتقال ورودی‌ها به دستگاه مدل
inputs = {k: v.to(device) for k, v in inputs.items()}

# تولید پاسخ
outputs = model.generate(
    inputs["input_ids"], attention_mask=inputs['attention_mask'], max_length=50
)

# نمایش پاسخ
print("Model Response:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
