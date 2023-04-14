import openai
from fine_tune.api_secrets import API_KEY

openai.api_key = API_KEY

prompt = "Tell me about the emu war."

response = openai.Completion.create(engine="text-davinci-001", prompt=prompt, max_tokens=50)

print(response)