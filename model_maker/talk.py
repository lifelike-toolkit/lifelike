import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys

def generate_response(prompt, model, tokenizer):
    """
    This function takes a prompt and generates a response using the fine-tuned GPT-2 model.
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def chat(character_model, tokenizer):
    """
    This function starts a chat app with the chosen character.
    """
    print(f"Start chatting with the {chosen_character} model. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            break

        prompt = chosen_character + ": " + user_input
        response = generate_response(prompt, character_model, tokenizer)
        print(f"{chosen_character}: {response}")

# Load the fine-tuned GPT-2 model and tokenizer
chosen_character = "NICK"
model_path = "./gpt2_character"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
character_model = GPT2LMHeadModel.from_pretrained(model_path)

# Start the chat app
chat(character_model, tokenizer)
