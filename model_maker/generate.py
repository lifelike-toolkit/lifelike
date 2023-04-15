import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import cleaner

def train_gpt2_on_character(dialogue, character):
    # Filter the dialogue for the specific character
    character_dialogue = [line for char, line in dialogue if char == character]

    # Save the character's dialogue to a file
    with open("character_dialogue.txt", "w") as file:
        file.write("\n".join(character_dialogue))

    # Load the tokenizer and the model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

    # Prepare the dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="character_dialogue.txt",
        block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir="./gpt2_character",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model("./gpt2_character")

# Fine-tune the GPT-2 model on the chosen character's dialogue
chosen_character = "NICK"
dialogue = cleaner.extract_dialogue("Zootopia.txt")
train_gpt2_on_character(dialogue, chosen_character)
