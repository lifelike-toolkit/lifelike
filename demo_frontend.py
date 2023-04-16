"""Streamlit-powered"""
import streamlit as st
from model_maker.talk import generate_response
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned GPT-2 model and tokenizer
chosen_character = "NICK WILDE"
model_path = "./gpt2_character"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
character_model = GPT2LMHeadModel.from_pretrained(model_path)

st.title("Lifelike Toolkit Demo")
st.write("As this is still in the demo stage, we are still testing and finding out how we can build a proper toolkit to support this.")
st.write("You are currently in the world of Zootopia. Here, you can converse with Nick Wilde. Feel free to talk to him: ")

# Get input
user_input = st.text_input("You to Nick Wilde: ", value="Who are you?")

# Get desireable prompt
prompt = "Me to " + chosen_character + ": " + user_input + "\n" + chosen_character + ": "

response = generate_response(prompt, character_model, tokenizer)
st.write(response)
