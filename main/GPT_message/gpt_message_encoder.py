import os
from openai import OpenAI

client = OpenAI()
import time
import pdb

def generate_chat_response(prompt, temperature = 0):
    time.sleep(0.2)
    response = client.chat.completions.create(model = os.getenv("OPENAI_MODEL"),    # choose in "gpt-3.5-turbo" or "gpt-4"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS")),
    temperature = temperature)
    return response.choices[0].message.content
