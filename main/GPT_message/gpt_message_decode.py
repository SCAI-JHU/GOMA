import openai
import time
import pdb

def generate_chat_response(prompt, max_tokens = 50, temperature = 0):
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
    
def message_decode(message):
    prompt=""
    file1 = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/project_lance/watch_talk_help/GPT_message/prompt_decode2.txt"
    file2 = "/scratch2/weka/tenenbaum/lanceyin/watch_talk_help/GPT_message/prompt_decode2.txt"
    try:
        with open(file1, "r", encoding='utf-8') as f:
            prompt=f.read()
    except:
        with open(file2, "r", encoding='utf-8') as f:
            prompt=f.read()
    
    #print(prompt) 
    if message is not None:
        gpt_prompt=prompt+message+"\n\nOutput:"
        output=generate_chat_response(gpt_prompt)
    else:
        return ""
    #time.sleep(0.2)
    
    # split_message = message.split(" ")

    # if split_message[:2] == ['I', 'think']:  # sharing info
    #     item1 = split_message[2][:-1].split("(")[1]
    #     command = split_message[5]
    #     item2 = split_message[6][:-2].split("(")[1]
    #     content = f"{command}_{item1}_{item2}"
    #     output = '{"message_type": "share_info",\n"content": "' + content + '"\n}'
    #     # return output
    # elif split_message[0] == 'Sorry':
    #     content = 'unknown'
    #     output = '{"message_type": "share_info",\n"content": "' + content + '"\n}'
    #     # return output

    # elif split_message[:2] == ['Have', 'you'] and split_message[2] != 'helped':  # ask info
    #     content = split_message[-2][:-1]
    #     output = '{"message_type": "ask_info",\n"content": "' + content + '"\n}'
    #     # return output

    # elif split_message[:2] == ['Help', 'me']:  # request
    #     relevant_part = split_message[-5:-1]
    #     num_items = int(relevant_part[0])
    #     item = relevant_part[1]
    #     command = relevant_part[2]
    #     target = int(relevant_part[3][:-2].split("(")[1])
    #     content = f"{command}_{item}_{target}:{num_items}"
    #     output = '{"message_type": "request",\n"content": "' + content + '"\n}'
    #     # return output
    # else:  # check progress
    #     relevant_part = split_message[-5:-1]
    #     num_items = int(relevant_part[0])
    #     item = relevant_part[1]
    #     command = relevant_part[2]
    #     target = int(relevant_part[3][:-2].split("(")[1])
    #     content = f"{command}_{item}_{target}:{num_items}"
    #     output = '{"message_type": "check_progress",\n"content": "' + content + '"\n}'
    #     # return output
    
    return output
