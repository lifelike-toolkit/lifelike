"""Streamlit-powered"""
import streamlit as st
from model_maker.talk import generate_response
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned GPT-2 model and tokenizer
chosen_character = "NICK"
model_path = "./gpt2_character"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
character_model = GPT2LMHeadModel.from_pretrained(model_path)

st.title("Lifelike Toolkit Demo")
st.markdown("**Disclaimer**: Just a demo for our baseline, we're still fucking around and hoping that we'll find out. Only up from here.\
    If you want to follow our progress, ask questions, or anything in between, follow [Mustafa on Twitter](https://twitter.com/mustafa_tariqk) where he posts updates as we make them, \
    or subscribe to [Khoa's Dev Blog](https://dev.to/basicallyok/) to hear me ramble about technical details and our process.\
    **If you want to trash on us, or provide feedback**, you can do that [here](https://forms.gle/2JvmRyw7ZZj1rVP29).")

st.write("You are currently in the world of Zootopia. Here, you can converse with Nick Wilde using a language model fine tuned with our toolkit using Nick's dialogues in the movies. Feel free to talk to him: ")
st.markdown("*May take a second, see top right for status*")

# Get input
user_input = st.text_input("You to Nick: ", value="Who are you?")

# Get desireable prompt
prompt = "Me to " + chosen_character + ": " + user_input + "\n" + chosen_character + ": "

response = generate_response(prompt, character_model, tokenizer)
st.write(response)
