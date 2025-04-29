import os
import os.path as osp
from openai import OpenAI

client = OpenAI()
import time
import pdb

def generate_chat_response(prompt, temperature = 0):
    time.sleep(0.1)
    response = client.chat.completions.create(model = os.getenv("OPENAI_MODEL"),    # choose in "gpt-3.5-turbo" or "gpt-4"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS")),
    temperature = temperature)
    return response.choices[0].message.content

def pairing_object(item,names):
    if item in names:
        return item
    prompt=""
    # with open("/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/hao_liu/watch_talk_help_test/watch_talk_help/GPT_message/prompt_pairing.txt", "r", encoding='utf-8') as f:
    #     prompt=f.read()

    file1 = f"{osp.dirname(osp.realpath(__file__))}/prompt_pairing.txt"
    with open(file1, "r", encoding='utf-8') as f:
        prompt=f.read()


    prompt=prompt+"\""+item+"\""+"\nList: ["+", ".join(names)+"]\nOutput: "

    #print("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPprompt:", prompt)
    output=generate_chat_response(prompt)
    # print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOoutput:",output)
    #print(output)
    time.sleep(0.2)

    return output
