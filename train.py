import os
import glob
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch

# Create output directory
OUTPUT_DIR = "model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_lyrics_from_local():
    """Loads all lyrics files in the 'data' folder."""
    lyrics = []
    for file_path in glob.glob("data/*.txt"):
        with open(file_path, 'r', encoding='utf-8') as file:
            lyrics.append(file.read())
    return lyrics

def main():
    print("Loading lyrics...")
    lyrics = load_lyrics_from_local()

    print(f"Loaded {len(lyrics)} lyrics files.")

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict({"text": lyrics})

    # Load GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Tokenize
    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        tokens["labels"] = tokens["input_ids"].copy()  # This is the key part
        return tokens


    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Split into train/validation sets
    datasets = tokenized_datasets.train_test_split(test_size=0.1)
    train_dataset = datasets["train"]
    eval_dataset = datasets["test"]

    # Training setup
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
