import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model and tokenizer
model_path = "model_output"  # Make sure this points to your saved folder
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set padding token (same fix you used for training)
tokenizer.pad_token = tokenizer.eos_token

def generate_lyrics(prompt, max_length=100, num_return_sequences=1):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # Enables randomness (creativity)
            top_k=50,        # Consider top 50 words at each step
            top_p=0.95       # Nucleus sampling (cutoff low-probability words)
        )

    generated_lyrics = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
    return generated_lyrics

if __name__ == "__main__":
    print("Enter a starting line (prompt) for the lyrics:")
    prompt = input("Prompt: ")

    lyrics = generate_lyrics(prompt, max_length=200, num_return_sequences=1)

    print("\nGenerated Lyrics:")
    print("-" * 40)
    print(lyrics[0])
