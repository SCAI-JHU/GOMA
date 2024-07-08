# from utils import utils_environment as utils
# from utils import utils_environment as utils_env2
import pickle
import sys
import os

import random

def convert_goal(task_goal, init_graph):
    new_task_goal = {}
    ids_from_class = {}

    for node in init_graph['nodes']:
        if node['class_name'] not in ids_from_class:
            ids_from_class[node['class_name']] = []
        ids_from_class[node['class_name']].append(node['id'])


    newgoals = {}
    for goal_name, count in task_goal.items():
        if type(count) == int:
            cont_id = int(goal_name.split('_')[-1])
            class_name = goal_name.split('_')[1]
            obj_grab = ids_from_class[class_name]
            newgoals[goal_name] = {
                'count': count,
                'grab_obj_ids': obj_grab,
                'container_ids': [cont_id]

            }
        else:
            newgoals[goal_name] = count
    return newgoals

# curr_dir = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(f"{curr_dir}/../../virtualhome/simulation/")
# sys.path.insert(0, f"/scratch2/weka/tenenbaum/lanceyin/virtualhome/virtualhome/simulation/")


env_task_set = pickle.load(open("/data/vision/torralba//frames/data_acquisition/SyntheticStories/agent_preferences/tshu/agent_preferences/dataset/structured_agent/train_env_task_set_20_full_task.all_apts.0,1,2,4,5.pik", "rb"))

def is_room(idx, graph):
    for node in graph["nodes"]:
        if node["id"]==idx and node["category"]=="Rooms":
            return True
    return False

def is_receptacle(idx, graph):
    for node in graph["nodes"]:
        if node["id"]==idx and node["class_name"] in [ "bathroomcabinet",
        "kitchencabinet",
        "cabinet",
        "fridge",
        "stove",
        "dishwasher",
        "microwave", "kitchentable",
        "coffeetable", "kitchencounter",
        "kitchencounterdrawer"]:
            return True
    return False


def find_rm_id(idx, graph):
    for edge in graph["edges"]:
        if edge["from_id"]==idx and is_room(edge["to_id"], graph):
            return edge["to_id"]

    return -1


def valid_placement(idx, graph):
    for edge in graph["edges"]:
        if edge["from_id"]==idx and is_receptacle(edge["to_id"], graph):
            return True

    return False

new_task_set = []

for k, env_task in enumerate(env_task_set):
    # print(env_task["task_goal"])
    task_goal = env_task["task_goal"][0]
    goal_pred = convert_goal(task_goal, env_task["init_graph"])

    remove_id = []
    obj_id = []
    all_goal_obj = []

    flag = True

    if len(goal_pred) < 2:
        continue

    # for info in goal_pred.values():
    #     count = info["count"]
    #     # if count < 2:
    #     #     flag = False

    #     for ids in info["grab_obj_ids"]:
    #         if not valid_placement(ids, env_task["init_graph"]):
    #             flag = False


    if flag:
        new_task_set.append(env_task)

for task in new_task_set:
    print(task["task_goal"])
with open("/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/project_lance/train_env_task_set_50_full_task.all_apts.0,1,2,4,5.pik", 'wb') as handle:
    pickle.dump(new_task_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

