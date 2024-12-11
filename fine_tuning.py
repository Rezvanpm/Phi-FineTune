from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from peft import SFTTrainer


def fine_tune_model(dataset_path, model_name="microsoft/Phi-3.5-mini-instruct"):
    """
    اجرای فرآیند Fine-Tuning بر روی مدل زبانی Phi-3.
    Args:
        dataset_path (str): مسیر دیتاست CSV.
        model_name (str): نام مدل زبانی پیش‌آموزش‌داده‌شده.
    Returns:
        dict: وضعیت فرآیند Fine-Tuning.
    """
    try:
        # بارگذاری دیتاست
        print("Loading dataset...")
        df = pd.read_csv(dataset_path)

        # بررسی ستون‌های موجود
        if "Abstract" not in df.columns or "Title" not in df.columns:
            raise ValueError(
                "Dataset must contain 'Title' and 'Abstract' columns.")

        # ترکیب ستون‌ها برای مدل
        df["text"] = "Title: " + df["Title"] + "\nAbstract: " + df["Abstract"]

        # تبدیل به فرمت HuggingFace Dataset
        dataset = Dataset.from_pandas(df[["text"]])

        # بارگذاری مدل و توکنایزر
        print("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # تنظیمات فرآیند آموزش
        print("Configuring training arguments...")
        args = TrainingArguments(
            evaluation_strategy="steps",
            per_device_train_batch_size=4,  # تعداد نمونه‌های ورودی به GPU
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            num_train_epochs=3,
            save_strategy="epoch",
            logging_steps=10,
            output_dir="./fine_tuned_model",
            fp16=True,
        )

        # ایجاد Trainer
        print("Creating trainer...")
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=tokenizer,
        )

        # شروع فرآیند آموزش
        print("Starting fine-tuning process...")
        trainer.train()

        return {"status": "Success", "message": "Fine-Tuning completed successfully.", "output_dir": "./fine_tuned_model"}

    except Exception as e:
        return {"status": "Error", "message": str(e)}
