import random

import openai
import torch
import json
import os
import pandas as pd
import time
import backoff
from GPT_message import gpt_message_decode, gpt_pairing_object, gpt_message_encoder
import pdb
import re
import numpy as np

class LLM:
    def __init__(self,
                 source,  # 'huggingface' or 'openai'
                 lm_id,
                 prompt_template_path,
                 communication,
                 cot,
                 sampling_parameters,
                 agent_id
                 ):
        self.goal_desc = None
        self.goal_location_with_r = None
        self.agent_id = agent_id
        self.agent_name = "Alice" if agent_id == 1 else "Bob"
        self.oppo_name = "Alice" if agent_id == 2 else "Bob"
        self.oppo_pronoun = "she" if agent_id == 2 else "he"
        self.debug = sampling_parameters.debug
        self.goal_location = None
        self.goal_location_id = None
        self.roomname2id = {}
        self.rooms = []
        self.team_goal = None
        self.prompt_template_path = prompt_template_path
        self.single = 'single' in self.prompt_template_path
        df = pd.read_csv(self.prompt_template_path)
        self.prompt_template = df['prompt'][0].replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
        if communication:
            self.generator_prompt_template = df['prompt'][1].replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
        else:
            self.generator_prompt_template = None

        self.communication = communication
        self.cot = cot
        self.source = source
        self.lm_id = lm_id
        self.chat = 'gpt-3.5-turbo' in lm_id or 'gpt-4' in lm_id
        self.OPENAI_KEY = None
        self.total_cost = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.helper_belief = {}
        self.comm_list = []

        if self.source == 'openai':
            
            if self.chat:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                }
            else:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                    "logprobs": sampling_parameters.logprobs,
                    "echo": sampling_parameters.echo,
                }
        elif source == "debug":
            self.sampling_params = sampling_parameters
        else:
            raise ValueError("invalid source")

        def lm_engine(source, lm_id, device):

            @backoff.on_exception(backoff.expo, OpenAIError)
            def _generate(prompt, sampling_params):
                usage = 0
                if source == 'openai':
                    try:
                        if self.chat:
                            response = openai.ChatCompletion.create(
                                model=lm_id, messages=prompt, **sampling_params
                            )
                            time.sleep(0.2)
                            # print(json.dumps(response, indent=4))
                            if self.debug:
                                with open(f"LLM/chat_raw.json", 'a') as f:
                                    f.write(json.dumps(response, indent=4))
                                    f.write('\n')
                            generated_samples = [response['choices'][i]['message']['content'] for i in
                                                 range(sampling_params['n'])]
                            if 'gpt-4' in self.lm_id:
                                usage = response['usage']['prompt_tokens'] * 0.03 / 1000 + response['usage']['completion_tokens'] * 0.06 / 1000
                            elif 'gpt-3.5' in self.lm_id:
                                usage = response['usage']['total_tokens'] * 0.002 / 1000
                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                        #                 range(sampling_params['n'])]
                        elif "text-" in lm_id:
                            response = openai.Completion.create(model=lm_id, prompt=prompt, **sampling_params)
                            time.sleep(0.2)
                            # print(json.dumps(response, indent=4))
                            if self.debug:
                                with open(f"LLM/raw.json", 'a') as f:
                                    f.write(json.dumps(response, indent=4))
                                    f.write('\n')
                            generated_samples = [response['choices'][i]['text'] for i in range(sampling_params['n'])]
                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                        #             range(sampling_params['n'])]
                        else:
                            raise ValueError(f"{lm_id} not available!")
                    except OpenAIError as e:
                        print(e)
                        raise e
                elif source == "debug":
                    return ["navigation"]
                else:
                    raise ValueError("invalid source")
                # generated_samples = [sample.strip().lower() for sample in generated_samples]
                return generated_samples, usage

            return _generate

        self.generator = lm_engine(self.source, self.lm_id, self.device)

    def init_belief(self, graph):
            # belief = {}
            cabinets = [n["id"] for n in graph["nodes"] if n["class_name"] in  [
            "kitchencabinet",
            "cabinet",
            "fridge",
            "stove",
            "dishwasher",
            "microwave",
            ]]

            cabinets_cnt = len(cabinets)
            # cabinets = np.random.choice(cabinets, int(cabinets_cnt*0.8))
            # cabinets = cabinets[:int(cabinets_cnt)]
            print("cabinets",cabinets)

            id2node = {node["id"]: node for node in graph["nodes"]}
            for edge in graph["edges"]:
                if "GRABBABLE" in id2node[edge["from_id"]]["properties"] and id2node[edge["to_id"]]["category"]!="Rooms" and  edge["relation_type"] not in ["CLOSE", "FACING"]:
                    if np.random.choice([0, 1], p=[0.15, 0.85]):
                        self.helper_belief[edge["from_id"]] = [edge["to_id"], edge["relation_type"]]

            # print("finished initialization of belief", self.helper_belief)

    def reset(self, rooms_name, roomname2id, goal_location, goal_location_id, unsatisfied):
        self.rooms = rooms_name
        self.roomname2id = roomname2id
        self.goal_location = goal_location
        self.goal_location_id = goal_location_id
        self.comm_list = []
        self.helper_belief={}
        self.goal_desc, self.goal_location_with_r = self.goal2description(unsatisfied, None)


    def goal2description(self, goals, goal_location_room):  # {predicate: count}
        # print(goals)
        map_rel_to_pred = {
            'inside': 'into',
            'on': 'onto',
        }
        s = "Find and put "
        r = None
        for predicate, vl in goals.items():
            relation, obj1, obj2 = predicate.split('_')
            count = vl
            if count == 0:
                continue
            if relation == 'holds':
                continue
                # s += f"Alice holds a book, "
            elif relation == 'sit':
                continue
                # s += f"Alice sits in {obj2}, "
            else:
                s += f"{count} {obj1}{'s' if count > 1 else ''}, "
                r = relation
        if r is None:
            return "None."

        s = s[:-2] + f" {map_rel_to_pred[r]} the {self.goal_location}."
        # if type(goal_location_room) is not list:
        #   s += f" in the {goal_location_room}."
        # else:
        #   ss = ' or '.join([f'{room}' for room in goal_location_room])
        #   s += f", which may be in the {ss}."
        return s, f"{map_rel_to_pred[r]} the {self.goal_location}"


    # def get_obj(self, obs, text, k=1):
    #   id2node = {node['id']: node for node in obs['nodes']}
    #   cnt = 0
    #   for x, node in id2node.items():
    #       if f'({x})' in text:
    #           cnt += 1
    #           if cnt != k: continue
    #           return f"<{node['class_name']}> ({x})"
    #   print("WARNING! No object correctly parsed!!! Random choose one")
    #   x, node = random.choice(list(id2node.items()))
    #   return f"<{node['class_name']}> ({x})"
    #
    #
    # def get_action(self, obs, text):
    #   if '[open]' in text or '[close]' in text or '[grab]' in text or '[walktowards]' in text:
    #       return f"[{text.split(']')[0].split('[')[-1]}] {self.get_obj(obs, text)}"
    #   elif 'putback' in text or 'putin' in text:
    #       obj1 = self.get_obj(obs, text)
    #       obj2 = self.get_obj(obs, text, 2)
    #       return f"[{text.split(']')[0].split('[')[-1]}] {obj1} {obj2}"


    def parse_answer(self, available_actions, text):
        for i in range(len(available_actions)):
            action = available_actions[i]
            if action in text:
                return action

        for i in range(len(available_actions)):
            action = available_actions[i]
            option = chr(ord('A') + i)
            # txt = text.lower()
            if f"option {option}" in text or f"{option}." in text.split(' ') or f"{option}," in text.split(' ') or f"Option {option}" in text or f"({option})" in text:
                return action
        print("WARNING! Fuzzy match!")
        for i in range(len(available_actions)):
            action = available_actions[i]
            if self.communication and i == 0:
                continue
            act, name, id = action.split(' ')
            option = chr(ord('A') + i)
            if f"{option} " in text or act in text or name in text or id in text:
                return action
        print("WARNING! No available action parsed!!! Random choose one")
        return random.choice(available_actions)



    def progress2text(self, current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored):
        sss = {}
        print(room_explored)
        for room, objs in ungrabbed_objects.items():
            cons = unchecked_containers[room]
            extra_obj = None
            if type(goal_location_room) is not list and goal_location_room == room:
                extra_obj = self.goal_location
            if objs is None and extra_obj is None and (room_explored is None or not room_explored[room]):
                sss[room] = f"The {room} is unexplored. "
                continue
            s = ""
            s_obj = ""
            s_con = ""
            if extra_obj is not None:
                s_obj = f"{extra_obj}, "
            if objs is not None and len(objs) > 0:
                if len(objs) == 1:
                    x = objs[0]
                    s_obj += f"<{x['class_name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['class_name']}> ({x['id']})" for x in objs])
                    s_obj += ss
            elif extra_obj is not None:
                s_obj = s_obj[:-2]
            if cons is not None and len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    s_con = f"an unchecked container <{x['class_name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['class_name']}> ({x['id']})" for x in cons])
                    s_con = f"unchecked containers " + ss
            if s_obj == "" and s_con == "":
                s += 'nothing'
                if room_explored is not None and not room_explored[room]:
                    s += ' yet'
            elif s_obj != "" and s_con != "":
                s += s_obj + ', and ' + s_con
            else:
                s += s_obj + s_con
            sss[room] = s

        s = ""
        all_satisfied_items = []
        for predicate, item_list in satisfied.items():
            if len(item_list):
                class_name = predicate.split('_')[1]
                for satisfied_item in item_list:
                    id_num = int(satisfied_item.split('_')[1])
                    all_satisfied_items.append(f"<{class_name}> ({id_num})")
        if len(all_satisfied_items):
            s = f"{'I' if self.single else 'We'}'ve already found and put " + ', '.join(all_satisfied_items) + ' ' + self.goal_location_with_r + '. '
                


        # if len(satisfied) == 0:
        #     s = ""
        # else:
        #     s = f"{'I' if self.single else 'We'}'ve already found and put "
        #     try:
        #         s += ', '.join([f"<{x['class_name']}> ({x['id']})" for x in satisfied])
        #     except:
        #         pdb.set_trace()
        #     s += ' ' + self.goal_location_with_r + '. '

        if len(grabbed_objects) == 0:
            s += "I'm holding nothing. "
        else:
            s += f"I'm holding <{grabbed_objects[0]['class_name']}> ({grabbed_objects[0]['id']}). "
            if len(grabbed_objects) == 2:
                s = s[:-2] + f" and <{grabbed_objects[1]['class_name']}> ({grabbed_objects[1]['id']}). "
        s += f"I'm in the {current_room['class_name']}, where I found {sss[current_room['class_name']]}. "
        ### opponent modeling
        if not self.single:
            ss = ""
            if len(opponent_grabbed_objects) == 0:
                ss += "nothing. "
            else:
                ss += f"<{opponent_grabbed_objects[0]['class_name']}> ({opponent_grabbed_objects[0]['id']}). "
                if len(opponent_grabbed_objects) == 2:
                    ss = ss[:-2] + f" and <{opponent_grabbed_objects[1]['class_name']}> ({opponent_grabbed_objects[1]['id']}). "
            if opponent_last_room is None:
                s += f"I don't know where {self.oppo_name} is. "
            elif opponent_last_room == current_room['class_name']:
                s += f"I also see {self.oppo_name} here in the {current_room['class_name']}, {self.oppo_pronoun} is holding {ss}"
            else:
                s += f"Last time I saw {self.oppo_name} was in the {opponent_last_room}, {self.oppo_pronoun} was holding {ss}"

        for room in self.rooms:
            if room == current_room['class_name']:
                continue
            if 'unexplored' in sss[room]:
                s += sss[room]
            else:
                s += f"I found {sss[room]} in the {room}. "

        return s


    def get_available_plans(self, grabbed_objects, unchecked_containers, ungrabbed_objects, message, room_explored):
        """
        [goexplore] <room>
        [gocheck] <container>
        [gograb] <target object>
        [goput] <goal location>
        [send_message] <"">
        """
        available_plans = []
        for room in self.rooms:
            if (room_explored is None or room_explored[room]) and unchecked_containers[room] is not None:
                continue
            available_plans.append(f"[goexplore] <{room}> ({self.roomname2id[room]})")
        if len(grabbed_objects) < 2:
            for cl in unchecked_containers.values():
                if cl is None:
                    continue
                for container in cl:
                    available_plans.append(f"[gocheck] <{container['class_name']}> ({container['id']})")
            for ol in ungrabbed_objects.values():
                if ol is None:
                    continue
                for obj in ol:
                    available_plans.append(f"[gograb] <{obj['class_name']}> ({obj['id']})")
        if len(grabbed_objects) > 0:
            available_plans.append(f"[goput] {self.goal_location}")
        
        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans

            
    def run(self, current_room, grabbed_objects, satisfied, unchecked_containers, ungrabbed_objects, goal_location_room, action_history, dialogue_history, opponent_grabbed_objects, opponent_last_room, room_explored, helper_obs, human_obs, graph):
        info = {}

        prompt_message = """
        I'm a helper robot assistant, I'm in a hurry to finish the housework with the human agent. Each of us have a partial observation of the room. Given our shared goal, dialogue history, our observations, my progress and previous actions, please help me choose a message to send to the other agent that may help each of us reach our subgoal. Note that I can hold two objects at a time and there are no costs for holding objects. All objects are denoted as name(id), such as coffeetable(712). In the message, I can only ask information about object location in the human agent's observation or share information about my observation to the human agent. The communication of the information should help the other agent achieve their goal or help me achieve mine. Please choose a message from one of the message options and rephrase to communicate it in a more colloquial way. I should only communicate when it's necessary. Reply "None" for no message.

        Please return a json output with the format: {"message": "XXX"}.

        Team Goal: $TEAM_GOAL$
        Your Goal: $GOAL$
        Your Observations: $OBS1$
        human's Observations: $OBS2$
        Progress: $PROGRESS$
        Dialogue history:
        $DIALOGUE_HISTORY$
        Previous actions: $ACTION_HISTORY$
        Available message options: $MESSAGE_OPTION$

        """

        # print(graph)

        if len(self.helper_belief) == 0:
            self.init_belief(graph)

        if self.team_goal == None:
            # with open("/data/vision/torralba//frames/data_acquisition/SyntheticStories/MultiAgent/project_lance/watch_talk_help/GPT_message/prompt_goal_inference.txt", "r", encoding='utf-8') as f:
            #     prompt=f.read()
        # prompt += f"\nHere is the mapping from id to class_name that you will sample targetID and baseItem from: \n {id2class}"
            prompt = f"You are a helpful robot assistant.\nHere is the list of requests for help the human has spoken previously: \n {dialogue_history}"
            prompt += f"""\n The team goals are the following:

            [put plate, cutleryfork, and wineglass or wineglass on table N],
            [put plate, cutleryfork, and wineglass or wineglass inside container N],
            [put apple, cupcake, pudding or salmon on table N],
            [put apple, cupcake, pudding or salmon inside container N],

    """
            prompt += f"\n Based on your own subgoal {self.goal_desc}, please indicate the most likely team goal. \nOutput: "
    # print(prompt)

            response = gpt_message_encoder.generate_chat_response(prompt, temperature=0).strip("'").strip('"')
            print(response)
            self.team_goal  = response


        # print(self.prompt_template)
        # goal_desc = self.goal2description(unsatisfied_goal, goal_location_room)
        progress_desc = self.progress2text(current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored)
        action_history_desc = ", ".join(action_history[-10:] if len(action_history) > 10 else action_history)
        dialogue_history_desc = '\n'.join(dialogue_history[-3:] if len(dialogue_history) > 3 else dialogue_history)
        prompt = self.prompt_template.replace('$TEAM_GOAL$', self.team_goal)
        prompt = prompt.replace('$GOAL$', self.goal_desc)
        prompt = prompt.replace('$PROGRESS$', progress_desc)
        prompt = prompt.replace('$ACTION_HISTORY$', action_history_desc)

        prompt_message = prompt_message.replace('$TEAM_GOAL$', self.team_goal)
        prompt_message = prompt_message.replace('$GOAL$', self.goal_desc)
        prompt_message = prompt_message.replace('$PROGRESS$', progress_desc)

        if helper_obs is not None:
            candidate_categories = ["cutleryfork", "cutleryknife", "pudding", "waterglass", "wineglass","plate", "cupcake","apple", "salmon", "bananas", "lime", "condimentbottle", "chips", "condimentshaker", "peach", "plum", "dishbowl","mug", "chocolatesyrup", "creamybuns", "breadslice", "fryingpan"]

            edgeStrings = []
            id2node = {x['id']: x for x in graph['nodes']}
            # print(helper_obs.keys())

            for edge in helper_obs["edges"]:
                if edge['from_id'] in id2node.keys() and edge['to_id'] in id2node.keys() and "GRABBABLE" in id2node[edge['from_id']]["properties"] and id2node[edge['to_id']]["category"]!="Rooms" and edge['relation_type']!="CLOSE" and edge['relation_type']!="FACING" and id2node[edge['from_id']]["class_name"] in candidate_categories:
                    self.helper_belief[edge['from_id']] = [edge["to_id"], edge['relation_type']]


            for item, loc in self.helper_belief.items():
                if id2node[item]["class_name"] in candidate_categories and item not in self.comm_list:
                    edgeStrings.append(f"[{id2node[item]['class_name']}({item}) {loc[1]} {id2node[loc[0]]['class_name']}({loc[0]})]")
            
            str_option = "[ \n"
            str_helper = "[ \n"

            for s in edgeStrings:
                str_option = str_option + f"communicate {s}\n"
                str_helper = str_helper + f"{s}\n"
            str_helper = str_helper + "]\n"

            prompt = prompt.replace('$OBS1$', str_helper)
            prompt_message = prompt_message.replace('$OBS1$', str_helper)


            edgeStrings = []
            for edge in human_obs["edges"]:
                if edge['from_id'] in id2node.keys() and edge['to_id'] in id2node.keys() and "GRABBABLE" in id2node[edge['from_id']]["properties"] and id2node[edge['to_id']]["category"]!="Rooms" and edge['relation_type']!="CLOSE" and edge['relation_type']!="FACING" and id2node[edge['from_id']]["class_name"] in candidate_categories:
                    edgeStrings.append(f"{id2node[edge['from_id']]['class_name']}({edge['from_id']})")
            
            str_main = "[ \n"
            for s in edgeStrings:
                item 
                str_option = str_option + f"ask about {s}\n"
                str_main = str_main + f"{s}\n"
            str_main = str_main + "]\n"
            str_option = str_option + "]\n"

            prompt_message = prompt_message.replace('$OBS2$', str_main)
            prompt_message = prompt_message.replace('$MESSAGE_OPTION$', str_option)

            # prompt = prompt.replace('$OBS1$', str_helper)

            # prompt = prompt.replace('$OBS1$', helper_obs)
            # prompt = prompt.replace('$OBS2$', main_obs)
        message = None

        prompt = prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)
        prompt_message = prompt_message.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)

        if self.communication:

            chat_prompt = [{"role": "user", "content": prompt_message}]
            outputs, usage = self.generator(chat_prompt if self.chat else prompt_message, self.sampling_params)
            message = outputs[0]

            
            # if not action_history[-1].startswith('[send_message]'):
            #     gen_prompt = self.generator_prompt_template.replace('$GOAL$', self.goal_desc)
            #     gen_prompt = gen_prompt.replace('$PROGRESS$', progress_desc)
            #     gen_prompt = gen_prompt.replace('$ACTION_HISTORY$', action_history_desc)
            #     gen_prompt = gen_prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)
            #     # if all_edges is not None and id2node is not None:
            #     #     gen_prompt = gen_prompt.replace('$OBS$', edgeStrings)
            #     gen_prompt = gen_prompt + f"\n{self.agent_name}:"

            #     self.total_cost += usage
                
            #     info['message_generator_prompt'] = gen_prompt
            #     info['message_generator_outputs'] = outputs
            #     info['message_generator_usage'] = usage
            #     if self.debug:
            #         print(f"message_generator_prompt:\n{gen_prompt}")
            #         print(f"message_generator_outputs:\n{message}")

        available_plans, num, available_plans_list = self.get_available_plans(grabbed_objects, unchecked_containers, ungrabbed_objects, message, room_explored)
        if num == 0 or (message is not None and num == 1):
            print("Warning! No available plans!")
            plan = None
            info.update({"num_available_actions": num,
                     "plan": None})
            return plan, info, message

        prompt = prompt.replace('$AVAILABLE_ACTIONS$', available_plans)

        print(prompt_message)

        if self.cot:
            prompt = prompt + " Let's think step by step."
            if self.debug:
                print(f"cot_prompt:\n{prompt}")
            chat_prompt = [{"role": "user", "content": prompt}]
            outputs, usage = self.generator(chat_prompt if self.chat else prompt, self.sampling_params)
            time.sleep(0.2)
            output = outputs[0]
            self.total_cost += usage
            info['cot_outputs'] = outputs
            info['cot_usage'] = usage
            if self.debug:
                print(f"cot_output:\n{output}")
            chat_prompt = [{"role": "user", "content": prompt},
                           {"role": "assistant", "content": output},
                           {"role": "user", "content": "Answer with only one best next action. So the answer is"}]
            normal_prompt = prompt + output + ' So the answer is'
            outputs, usage = self.generator(chat_prompt if self.chat else normal_prompt, self.sampling_params)
            time.sleep(0.2)
            output = outputs[0]
            self.total_cost += usage
            info['output_usage'] = usage
            if self.debug:
                print(f"base_output:\n{output}")
                print(f"total cost: {self.total_cost}")
        else:
            if self.debug:
                print(f"base_prompt:\n{prompt}")
            outputs, usage = self.generator([{"role": "user", "content": prompt}] if self.chat else prompt, self.sampling_params)
            time.sleep(0.2)
            output = outputs[0]
            info['cot_usage'] = usage
            if self.debug:
                print(f"base_output:\n{output}")

        print("output",output)
        plan = self.parse_answer(available_plans_list, output)

        print(message)

        if "None" in message:
            message = None
        time.sleep(0.2)
        if self.debug:
            print(f"plan: {plan}\n")
        info.update({"num_available_actions": num,
                     "prompts": prompt,
                     "outputs": outputs,
                     "plan": plan,
                     "total_cost": self.total_cost})

        print(message)
        return plan, info, message
