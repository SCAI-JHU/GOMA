import openai
import time
import pdb

def generate_chat_response(prompt, max_tokens = 50, temperature = 0):
    time.sleep(0.2)
    response = openai.ChatCompletion.create(
        model = "gpt-4",    # choose in "gpt-3.5-turbo" or "gpt-4"
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens = max_tokens,
        temperature = temperature,
    )
    return response["choices"][0]["message"]["content"]
