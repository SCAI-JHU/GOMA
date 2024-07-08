import openai
import time
import pdb

def generate_chat_response(prompt, max_tokens = 50, temperature = 0):
    time.sleep(0.1)
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
    
def pairing_object(item,names):
    if item in names:
        return item
    prompt=""
    # with open("/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/hao_liu/watch_talk_help_test/watch_talk_help/GPT_message/prompt_pairing.txt", "r", encoding='utf-8') as f:
    #     prompt=f.read()
        
    file1 = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/hao_liu/watch_talk_help_test/watch_talk_help/GPT_message/prompt_pairing.txt"
    file2 = "/scratch2/weka/tenenbaum/kunaljha/watch_talk_help/GPT_message/prompt_pairing.txt"
    try:
        with open(file1, "r", encoding='utf-8') as f:
            prompt=f.read()
    except:
        with open(file2, "r", encoding='utf-8') as f:
            prompt=f.read()
        

    prompt=prompt+"\""+item+"\""+"\nList: ["+", ".join(names)+"]\nOutput: "

    #print("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPprompt:", prompt)
    output=generate_chat_response(prompt)
    # print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOoutput:",output)
    #print(output)
    time.sleep(0.2)
    
    return output
