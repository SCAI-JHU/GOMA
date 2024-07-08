import numpy as np
import traceback
import logging
import random
import time
import math
import copy
import importlib
import json
import multiprocessing as mp
from functools import partial
import ipdb
import pdb
import pickle
import eventlet
import scipy
from itertools import repeat
import os
from scipy.stats import entropy
from scipy.special import softmax
from scipy.special import logsumexp

from . import belief
from . import Mind
from . import state_belief
from envs.graph_env import VhGraphEnv

#
import pdb
from MCTS import *

import sys

sys.path.append("..")
from utils import utils_environment as utils_env
from utils import utils_exception
from utils_environment import convert_goal, check_progress, check_progress2
from GPT_message import gpt_message_decode, gpt_pairing_object, gpt_message_encoder
from arguments import get_args


def find_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, object_target
):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    id2node = {node["id"]: node for node in env_graph["nodes"]}
    containerdict = {
        edge["from_id"]: edge["to_id"]
        for edge in env_graph["edges"]
        if edge["relation_type"] == "INSIDE"
    }
    target = int(object_target.split("_")[-1])
    observation_ids = [x["id"] for x in observations["nodes"]]
    try:
        room_char = [
            edge["to_id"]
            for edge in env_graph["edges"]
            if edge["from_id"] == agent_id and edge["relation_type"] == "INSIDE"
        ][0]
    except:
        print("Error")
        # ipdb.set_trace()

    action_list = []
    cost_list = []
    # if target == 478:
    #     )ipdb.set_trace()
    while target not in observation_ids:
        try:
            container = containerdict[target]
        except:
            print(id2node[target])
            print(containerdict.keys())
            raise Exception
        # If the object is a room, we have to walk to what is insde

        if id2node[container]["category"] == "Rooms":
            action_list = [
                ("walk", (id2node[target]["class_name"], target), None)
            ] + action_list
            cost_list = [0.5] + cost_list

        elif "CLOSED" in id2node[container]["states"] or (
            "OPEN" not in id2node[container]["states"]
        ):
            action = ("open", (id2node[container]["class_name"], container), None)
            action_list = [action] + action_list
            cost_list = [0.05] + cost_list

        target = container

    ids_character = [
        x["to_id"]
        for x in observations["edges"]
        if x["from_id"] == agent_id and x["relation_type"] == "CLOSE"
    ] + [
        x["from_id"]
        for x in observations["edges"]
        if x["to_id"] == agent_id and x["relation_type"] == "CLOSE"
    ]

    if target not in ids_character:
        # If character is not next to the object, walk there
        action_list = [
            ("walk", (id2node[target]["class_name"], target), None)
        ] + action_list
        cost_list = [1] + cost_list

    return action_list, cost_list, f"find_{target}"


def touch_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, object_target
):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split("_")[-1])

    observed_ids = [node["id"] for node in observations["nodes"]]
    agent_close = [
        edge
        for edge in env_graph["edges"]
        if (
            (edge["from_id"] == agent_id and edge["to_id"] == target_id)
            or (edge["from_id"] == target_id and edge["to_id"] == agent_id)
            and edge["relation_type"] == "CLOSE"
        )
    ]

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_id][0]

    target_action = [("touch", (target_node["class_name"], target_id), None)]
    cost = [0.05]

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost, f"touch_{target_id}"
    else:
        find_actions, find_costs, _ = find_heuristic(
            agent_id, char_index, unsatisfied, env_graph, simulator, object_target
        )
        return find_actions + target_action, find_costs + cost, f"touch_{target_id}"


def grab_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, object_target
):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split("_")[-1])

    observed_ids = [node["id"] for node in observations["nodes"]]
    agent_close = [
        edge
        for edge in env_graph["edges"]
        if (
            (edge["from_id"] == agent_id and edge["to_id"] == target_id)
            or (edge["from_id"] == target_id and edge["to_id"] == agent_id)
            and edge["relation_type"] == "CLOSE"
        )
    ]
    grabbed_obj_ids = [
        edge["to_id"]
        for edge in env_graph["edges"]
        if (edge["from_id"] == agent_id and "HOLDS" in edge["relation_type"])
    ]

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_id][0]

    if target_id not in grabbed_obj_ids:
        target_action = [("grab", (target_node["class_name"], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost, f"grab_{target_id}"
    else:
        find_actions, find_costs, _ = find_heuristic(
            agent_id, char_index, unsatisfied, env_graph, simulator, object_target
        )
        return find_actions + target_action, find_costs + cost, f"grab_{target_id}"


def turnOn_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, object_target
):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split("_")[-1])

    observed_ids = [node["id"] for node in observations["nodes"]]
    agent_close = [
        edge
        for edge in env_graph["edges"]
        if (
            (edge["from_id"] == agent_id and edge["to_id"] == target_id)
            or (edge["from_id"] == target_id and edge["to_id"] == agent_id)
            and edge["relation_type"] == "CLOSE"
        )
    ]
    grabbed_obj_ids = [
        edge["to_id"]
        for edge in env_graph["edges"]
        if (edge["from_id"] == agent_id and "HOLDS" in edge["relation_type"])
    ]

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_id][0]

    if target_id not in grabbed_obj_ids:
        target_action = [("switchon", (target_node["class_name"], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost, f"turnon_{target_id}"
    else:
        find_actions, find_costs = find_heuristic(
            agent_id, char_index, unsatisfied, env_graph, simulator, object_target
        )
        return find_actions + target_action, find_costs + cost, f"turnon_{target_id}"


def sit_heuristic(
    agent_id, char_index, unsatisfied, env_graph, simulator, object_target
):
    observations = simulator.get_observations(env_graph, char_index=char_index)
    target_id = int(object_target.split("_")[-1])

    observed_ids = [node["id"] for node in observations["nodes"]]
    agent_close = [
        edge
        for edge in env_graph["edges"]
        if (
            (edge["from_id"] == agent_id and edge["to_id"] == target_id)
            or (edge["from_id"] == target_id and edge["to_id"] == agent_id)
            and edge["relation_type"] == "CLOSE"
        )
    ]
    on_ids = [
        edge["to_id"]
        for edge in env_graph["edges"]
        if (edge["from_id"] == agent_id and "ON" in edge["relation_type"])
    ]

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_id][0]

    if target_id not in on_ids:
        target_action = [("sit", (target_node["class_name"], target_id), None)]
        cost = [0.05]
    else:
        target_action = []
        cost = []

    if len(agent_close) > 0 and target_id in observed_ids:
        return target_action, cost, f"sit_{target_id}"
    else:
        find_actions, find_costs = find_heuristic(
            agent_id, char_index, unsatisfied, env_graph, simulator, object_target
        )
        return find_actions + target_action, find_costs + cost, f"sit_{target_id}"


def put_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, target):
    # Modif, now put heristic is only the immaediate after action
    observations = simulator.get_observations(env_graph, char_index=char_index)

    target_grab, target_put = [int(x) for x in target.split("_")[-2:]]

    if (
        sum(
            [
                1
                for edge in observations["edges"]
                if edge["from_id"] == target_grab
                and edge["to_id"] == target_put
                and edge["relation_type"] == "ON"
            ]
        )
        > 0
    ):
        # Object has been placed
        # ipdb.set_trace()
        return [], 0, []

    if (
        sum(
            [
                1
                for edge in observations["edges"]
                if edge["to_id"] == target_grab
                and edge["from_id"] != agent_id
                and "HOLD" in edge["relation_type"]
            ]
        )
        > 0
    ):
        # Object has been placed
        # ipdb.set_trace()
        return [], 0, []

    id2node = {node["id"]: node for node in env_graph["nodes"]}
    try:
        target_node = [node for node in env_graph["nodes"] if node["id"] == target_grab][0]
        target_node2 = [node for node in env_graph["nodes"] if node["id"] == target_put][0]
    except:
        print(id2node.keys())
        print(target_grab)
        print(target_put)
        exit(0)

    target_grabbed = (
        len(
            [
                edge
                for edge in env_graph["edges"]
                if edge["from_id"] == agent_id
                and "HOLDS" in edge["relation_type"]
                and edge["to_id"] == target_grab
            ]
        )
        > 0
    )

    object_diff_room = None
    if not target_grabbed:
        grab_obj1, cost_grab_obj1, heuristic_name = grab_heuristic(
            agent_id,
            char_index,
            unsatisfied,
            env_graph,
            simulator,
            "grab_" + str(target_node["id"]),
        )
        if len(grab_obj1) > 0:
            if grab_obj1[0][0] == "walk":
                id_room = grab_obj1[0][1][1]
                if id2node[id_room]["category"] == "Rooms":
                    object_diff_room = id_room

        return grab_obj1, cost_grab_obj1, heuristic_name
    else:
        env_graph_new = env_graph
        grab_obj1 = []
        cost_grab_obj1 = []
        find_obj2, cost_find_obj2, _ = find_heuristic(
            agent_id,
            char_index,
            unsatisfied,
            env_graph_new,
            simulator,
            "find_" + str(target_node2["id"]),
        )

    action = [
        (
            "putback",
            (target_node["class_name"], target_grab),
            (target_node2["class_name"], target_put),
        )
    ]
    cost = [0.05]
    res = grab_obj1 + find_obj2 + action
    cost_list = cost_grab_obj1 + cost_find_obj2 + cost
    # print(res, target)
    return res, cost_list, f"put_{target_grab}_{target_put}"


def putIn_heuristic(agent_id, char_index, unsatisfied, env_graph, simulator, target):
    # TODO: change this as well
    observations = simulator.get_observations(env_graph, char_index=char_index)

    target_grab, target_put = [int(x) for x in target.split("_")[-2:]]

    if (
        sum(
            [
                1
                for edge in observations["edges"]
                if edge["from_id"] == target_grab
                and edge["to_id"] == target_put
                and edge["relation_type"] == "ON"
            ]
        )
        > 0
    ):
        # Object has been placed
        return [], 0, []

    if (
        sum(
            [
                1
                for edge in observations["edges"]
                if edge["to_id"] == target_grab
                and edge["from_id"] != agent_id
                and "HOLD" in edge["relation_type"]
            ]
        )
        > 0
    ):
        # Object has been placed
        return None, 0, None

    target_node = [node for node in env_graph["nodes"] if node["id"] == target_grab][0]
    target_node2 = [node for node in env_graph["nodes"] if node["id"] == target_put][0]
    id2node = {node["id"]: node for node in env_graph["nodes"]}
    target_grabbed = (
        len(
            [
                edge
                for edge in env_graph["edges"]
                if edge["from_id"] == agent_id
                and "HOLDS" in edge["relation_type"]
                and edge["to_id"] == target_grab
            ]
        )
        > 0
    )

    object_diff_room = None
    if not target_grabbed:
        grab_obj1, cost_grab_obj1, heuristic_name = grab_heuristic(
            agent_id,
            char_index,
            unsatisfied,
            env_graph,
            simulator,
            "grab_" + str(target_node["id"]),
        )
        if len(grab_obj1) > 0:
            if grab_obj1[0][0] == "walk":
                id_room = grab_obj1[0][1][1]
                if id2node[id_room]["category"] == "Rooms":
                    object_diff_room = id_room

        return grab_obj1, cost_grab_obj1, heuristic_name

    else:
        grab_obj1 = []
        cost_grab_obj1 = []

        env_graph_new = env_graph
        grab_obj1 = []
        cost_grab_obj1 = []
        find_obj2, cost_find_obj2, _ = find_heuristic(
            agent_id,
            char_index,
            unsatisfied,
            env_graph_new,
            simulator,
            "find_" + str(target_node2["id"]),
        )
        target_put_state = target_node2["states"]
        action_open = [("open", (target_node2["class_name"], target_put))]
        action_put = [
            (
                "putin",
                (target_node["class_name"], target_grab),
                (target_node2["class_name"], target_put),
            )
        ]
        cost_open = [0.05]
        cost_put = [0.05]

        remained_to_put = 0
        for predicate, count in unsatisfied.items():
            if predicate.startswith("inside"):
                remained_to_put += count
        if remained_to_put == 1:  # or agent_id > 1:
            action_close = []
            cost_close = []
        else:
            action_close = []
            cost_close = []
            # action_close = [('close', (target_node2['class_name'], target_put))]
            # cost_close = [0.05]

        if "CLOSED" in target_put_state or "OPEN" not in target_put_state:
            res = grab_obj1 + find_obj2 + action_open + action_put + action_close
            cost_list = (
                cost_grab_obj1 + cost_find_obj2 + cost_open + cost_put + cost_close
            )
        else:
            res = grab_obj1 + find_obj2 + action_put + action_close
            cost_list = cost_grab_obj1 + cost_find_obj2 + cost_put + cost_close

        # print(res, target)
        grab_node = target_node["id"]
        place_node = target_node2["id"]
        return res, cost_list, f"putin_{grab_node}_{place_node}"



def check_object_belief(belief_dict, obj_id, p_threshold):
    belief_id_dict = belief_dict[obj_id]
    for key, value in belief_id_dict.items():
        for i, v in enumerate(value[1]):
            if i>1 and v> p_threshold:
                return False

    return True

def values_to_prob_distrib(log_p):
    if logsumexp(log_p) < np.log(1e-6):
        probs = np.array([1/len(log_p)] * len(log_p))
    else:
        log_p_normalized = log_p - logsumexp(log_p)
        probs = softmax(log_p_normalized)

    # adding small noise for numerical stability
    # Find the index of the element with value 1
    index_of_one = np.where(probs == 1)[0]
    if len(index_of_one) == 0:
        return probs
    else:
        index_of_one = [0]
    
        # Calculate the new value for the element with value 1
        new_value = 1 - (np.count_nonzero(probs == 0) * 1e-6)
        
        # Update the array with the new values
        probs[probs != 1] = 1e-6
        probs[index_of_one] = new_value
        
        return probs


def clean_graph(state, goal_spec, last_opened, backup_id2node=None):
    # TODO: document well what this is doing_
    new_graph = {}
    # get all ids
    ids_interaction = []
    nodes_missing = []
    for predicate, val_goal in goal_spec.items():
        elements = predicate.split("_")
        nodes_missing += val_goal["grab_obj_ids"]
        nodes_missing += val_goal["container_ids"]

    nodes_missing += [
        node["id"]
        for node in state["nodes"]
        if node["class_name"] == "character" or node["category"] in ["Rooms", "Doors"]
    ]

    def clean_node(curr_node):
        return {
            "id": curr_node["id"],
            "class_name": curr_node["class_name"],
            "category": curr_node["category"],
            "states": curr_node["states"],
            "properties": curr_node["properties"],
        }

    id2node = {node["id"]: clean_node(node) for node in state["nodes"]}
    backup_id2node = {nID: clean_node(node) for nID, node in backup_id2node.items()}
    # print([node for node in state['nodes'] if node['class_name'] == 'kitchentable'])
    # print(id2node[235])
    # ipdb.set_trace()
    inside = {}
    for edge in state["edges"]:
        if edge["relation_type"] == "INSIDE":
            if edge["from_id"] not in inside.keys():
                inside[edge["from_id"]] = []
            inside[edge["from_id"]].append(edge["to_id"])

    while len(nodes_missing) > 0:
        new_nodes_missing = []
        for node_missing in nodes_missing:
            if node_missing in inside:
                new_nodes_missing += [
                    node_in
                    for node_in in inside[node_missing]
                    if node_in not in ids_interaction
                ]
            ids_interaction.append(node_missing)
        nodes_missing = list(set(new_nodes_missing))

    if last_opened is not None:
        obj_id = int(last_opened[1][1:-1])
        if obj_id not in ids_interaction:
            ids_interaction.append(obj_id)

    # for clean up tasks, add places to put objects to
    augmented_class_names = []
    for key, value in goal_spec.items():
        elements = key.split("_")
        if elements[0] == "off":
            if id2node[value["containers"][0]]["class_name"] in [
                "dishwasher",
                "kitchentable",
            ]:
                augmented_class_names += [
                    "kitchencabinets",
                    "kitchencounterdrawer",
                    "kitchencounter",
                ]
                break
    for key in goal_spec:
        elements = key.split("_")
        if elements[0] == "off":
            if id2node[value["container_ids"][0]]["class_name"] in ["sofa", "chair"]:
                augmented_class_names += ["coffeetable"]
                break
    containers = [
        [node["id"], node["class_name"]]
        for node in state["nodes"]
        if node["class_name"] in augmented_class_names
    ]
    for obj_id in containers:
        if obj_id not in ids_interaction:
            ids_interaction.append(obj_id)
    try:
        new_edges = [
            edge
            for edge in state["edges"]
            if edge["from_id"] in ids_interaction and edge["to_id"] in ids_interaction
        ]
        new_nodes = []
        for id_node in ids_interaction:
            try:
                new_nodes.append(id2node[id_node])
            except:
                new_nodes.append(backup_id2node[id_node])
        new_graph = {
            "edges": new_edges,
            "nodes": new_nodes,
        }
    except:
        pdb.set_trace()

    return new_graph


def mp_run_mcts(root_node, mcts, nb_steps, last_subgoal, opponent_subgoal):
    heuristic_dict = {
        "find": find_heuristic,
        "grab": grab_heuristic,
        "put": put_heuristic,
        "putIn": putIn_heuristic,
        "sit": sit_heuristic,
        "turnOn": turnOn_heuristic,
        "touch": touch_heuristic,
    }
    # res = root_node * 2
    try:
        new_mcts = copy.deepcopy(mcts)
        res = new_mcts.run(
            root_node, nb_steps, heuristic_dict, last_subgoal, opponent_subgoal
        )
    except Exception as e:
        # print("plan fail in index", root_node.particle_id)
        # traceback.print_stack()
        # print("raising")
        # print("Exception...")
        # print(utils_exception.ExceptionWrapper(e))
        # print('---')
        return utils_exception.ExceptionWrapper(e)
    return res


def mp_run_2(
    process_id, root_node, mcts, nb_steps, last_subgoal, opponent_subgoal, res
):  
    print(f"about to call mp_run_mcts for {process_id}")
    res[process_id] = mp_run_mcts(
        root_node=root_node,
        mcts=mcts,
        nb_steps=nb_steps,
        last_subgoal=last_subgoal,
        opponent_subgoal=opponent_subgoal,
    )
    print("successful call")
    # print(f"res process is {res[process_id]}")
    #print(res[process_id])


def get_plan(
    mcts,
    particles,
    env,
    nb_steps,
    goal_spec,
    last_subgoal,
    last_action,
    opponent_subgoal=None,
    num_process=10,
    length_plan=5,
    verbose=True,
    return_score=False
):
    # if goal_spec is not None:
    #     print(goal_spec)


    root_nodes = []
    for particle_id in range(len(particles)):
        root_action = None
        root_node = Node(
            id=(root_action, [goal_spec, 0, ""]),
            particle_id=particle_id,
            plan=[],
            state=copy.deepcopy(particles[particle_id]),
            num_visited=0,
            sum_value=0,
            is_expanded=False,
        )
        root_nodes.append(root_node)

    # root_nodes = list(range(10))
    mp_run = partial(
        mp_run_mcts,
        mcts=mcts,
        nb_steps=nb_steps,
        last_subgoal=last_subgoal,
        opponent_subgoal=opponent_subgoal,
    )

    if len(root_nodes) == 0:
        print("No root nodes")
        raise Exception
    if num_process > 1:
        manager = mp.Manager()
        res = manager.dict()
        num_root_nodes = len(root_nodes)
        for start_root_id in range(0, num_root_nodes, num_process):
            end_root_id = min(start_root_id + num_process, num_root_nodes)
            jobs = []

            # node_args = [root_nodes[pID] for pID in range(start_root_id, end_root_id)]
            # process_ids = list(range(start_root_id, end_root_id))
            # with mp.Pool() as pool:
            #     p = pool.starmap(mp_run_2, zip(process_ids, node_args, repeat(mcts), repeat(nb_steps), 
            #         repeat(last_subgoal), repeat(opponent_subgoal), repeat(res)))

            for process_id in range(start_root_id, end_root_id):
                # print(process_id)
                p = mp.Process(
                    target=mp_run_2,
                    args=(
                        process_id,
                        root_nodes[process_id],
                        mcts,
                        nb_steps,
                        last_subgoal,
                        opponent_subgoal,
                        res,
                    ),
                )
                jobs.append(p)
                p.start()
            for p in jobs:
                p.join()
        info = [res[x] for x in range(len(root_nodes))]

    else:
        info = [mp_run(rn) for rn in root_nodes]

    for info_item in info:
        if isinstance(info_item, utils_exception.ExceptionWrapper):
            print("raising")
            info_item.re_raise()

    if num_process > 0:
        print("Plan Done")
    rewards_all = [inf[-1] for inf in info]
    plans_all = [inf[1] for inf in info]
    goals_all = [inf[-2] for inf in info]
    index_action = 0
    # length_plan = 5
    prev_index_particles = list(range(len(info)))

    final_actions, final_goals, action_score_list, action_reward_list = [], [], [], []
    lambd = 0.5
    # ipdb.set_trace()
    while index_action < length_plan:
        max_action = None
        max_score = None
        action_count_dict = {}
        action_reward_dict = {}
        action_goal_dict = {}
        # Which particles we select now
        index_particles = [
            p_id for p_id in prev_index_particles if len(plans_all[p_id]) > index_action
        ]
        # print(index_particles)
        if len(index_particles) == 0:
            index_action += 1
            continue
        for ind in index_particles: # for every particle
            action = plans_all[ind][index_action]
            if action is None:
                continue
            try:
                reward = rewards_all[ind][index_action]
                goal = goals_all[ind][index_action]
            except:
                ipdb.set_trace()
            if not action in action_count_dict:
                action_count_dict[action] = []
                action_goal_dict[action] = []
                action_reward_dict[action] = 0
            action_count_dict[action].append(ind)
            action_reward_dict[action] += reward
            action_goal_dict[action].append(goal)

        action_score_dict = {}
        average_reward_dict = {}
        for action in action_count_dict:
            # Average reward of this action
            average_reward = (
                action_reward_dict[action] / len(action_count_dict[action])
            )
            # Average proportion of particles
            average_visit = len(action_count_dict[action]) / len(index_particles)
            score = average_reward * lambd + average_visit
            goal = action_goal_dict[action]

            action_score_dict[action] = len(action_count_dict[action])
            average_reward_dict[action] = average_reward

            if max_score is None or max_score < score:
                max_score = score
                max_action = action
                max_goal = goal
            elif max_score == score:  # randomly select between maximizing actions
                max_score = score 
                max_action = random.choice([max_action, action])
                if max_action == action:
                    max_goal = goal


        index_action += 1
        prev_index_particles = action_count_dict[max_action]
        # print(max_action, prev_index_particles)
        final_actions.append(max_action)
        final_goals.append(max_goal)
        action_score_list.append(action_score_dict)
        action_reward_list.append(average_reward_dict)

    # ipdb.set_trace()
    # If there is no action predicted but there were goals missing...
    if len(final_actions) == 0:
        print("No actions")
        # ipdb.set_trace()

    plan = final_actions
    subgoals = final_goals

    # ipdb.set_trace()
    # subgoals = [[None, None, None], [None, None, None]]
    # next_root, plan, subgoals = mp_run_mcts(root_nodes[0])
    next_root = None

    # ipdb.set_trace()
    # print('plan', plan)
    # if 'put' in plan[0]:
    #     ipdb.set_trace()
    for pIdx, particle_plans in enumerate(plans_all):
        if len(particle_plans) > 0:
            plans_all[pIdx] = [{p:1} for p in particle_plans[:min(length_plan, len(particle_plans))]]

    if verbose:
        print("plan", plan)
        print("subgoal", subgoals)
    sample_id = None

    if sample_id is not None:
        res[sample_id] = plan
    elif return_score:
        return plan, next_root, subgoals, action_score_list, action_reward_list, plans_all
    else:
        return plan, next_root, subgoals


class MCTS_agent_particle_v2_instance_helper_message:
    """
    MCTS for a single agent
    """

    def __init__(
        self,
        agent_id,
        char_index,
        max_episode_length,
        num_simulation,
        max_rollout_steps,
        c_init,
        c_base,
        num_particles=20,
        recursive=False,
        num_samples=1,
        num_processes=1,
        comm=None,
        logging=False,
        logging_graphs=False,
        agent_params={},
        get_plan_states=False,
        seed=None,
        agent_type = "MCTS_helper_message",
        comm_type = "goal"
    ):
        self.agent_type = agent_type
        self.comm_type = comm_type
        self.verbose = False
        self.recursive = recursive
        self.goal_belief_objects = [[],[]]

        self.room_id = 0

        # self.env = unity_env.env
        if seed is None:
            seed = random.randint(0, 100)
        self.seed = seed
        self.logging = logging
        self.logging_graphs = logging_graphs

        self.last_obs = None
        self.last_plan = None

        self.agent_id = agent_id
        self.char_index = char_index

        self.sim_env = VhGraphEnv(n_chars=self.agent_id)
        self.sim_env.pomdp = True
        self.belief = None

        #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& Anything more about how to init minds?

        self.belief_params = agent_params["belief"]
        self.agent_params = agent_params
        self.max_episode_length = max_episode_length
        self.num_simulation = num_simulation
        self.max_rollout_steps = max_rollout_steps
        self.c_init = c_init
        self.c_base = c_base
        self.num_samples = num_samples
        self.num_processes = num_processes
        self.num_particles = num_particles
        self.get_plan_states = get_plan_states

        self.previous_belief_graph = None
        self.verbose = False

        # self.should_close = True
        # if self.planner_params:
        #     if 'should_close' in self.planner_params:
        #         self.should_close = self.planner_params['should_close']

        self.mcts = None

        self.particles = [None for _ in range(self.num_particles)]
        self.particles_full = [None for _ in range(self.num_particles)]
        self.human_mind_particles = [None for _ in range(self.num_particles)]

        self.message_history = []
        self.action_history = []
        # if self.mcts is None:
        #    raise Exception

        # Indicates whether there is a unity simulation
        self.comm = comm
        self.all_names=[]
        
        self.helper_goal = None
        self.human_goal = None

        self.human_belief = {}
        self.helper_belief = {}
        self.ask_item = []
        self.time_last_asked = 0

    # def locate_agent_room(self, graph):
    #     for edge in graph["edges"]:
    #         if edge["from_id"] == 1:
    #             self.

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
        # print("cabinets",cabinets)

        id2node = {node["id"]: node for node in graph["nodes"]}
        for edge in graph["edges"]:
            if "GRABBABLE" in id2node[edge["from_id"]]["properties"] and id2node[edge["to_id"]]["category"]!="Rooms" and  edge["relation_type"] not in ["CLOSE", "FACING"]:
                if np.random.choice([0, 1], p=[0.15, 0.85]):
                    self.helper_belief[edge["from_id"]] = [edge["to_id"], edge["relation_type"]]

        # print("finished initialization of belief", self.helper_belief)

    def update_obs(self, obs, output):
        # print("Adding",output)
        has_obj1=False;has_obj2=False;has_relation=False
        relation,obj1_id,obj2_id=output.split('_')
        Node1=None;Node2=None

        id2node = {node["id"]: node for node in self.init_gt_graph["nodes"]}

        # print(id2node.keys())
        if int(obj1_id) not in id2node.keys() or int(obj2_id) not in id2node.keys():
            return obs

        Node1=id2node[int(obj1_id)]
        Node2=id2node[int(obj2_id)]
        RoomNode=None

        for edge in self.init_gt_graph["edges"]:
            if edge["from_id"]==int(obj2_id):
                if id2node[edge["to_id"]]["category"]=="Rooms":
                    RoomNode=id2node[edge["to_id"]]
                    break

        for node in obs["nodes"]:
            if node["id"]==int(obj1_id):
                has_obj1=True
            if node["id"]==int(obj2_id):
                has_obj2=True
        if has_obj1 and has_obj2:
            has_relation=True

        #print(obs)
        #RoomNodeAdded=False
        if has_obj1==False:
            #print("NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNode1",Node1)
            obs["nodes"].append(Node1)
            #obs["nodes"].append(RoomNode)
            #RoomNodeAdded=True
            #obs["edges"].append({'from_id':int(obj1_id),'to_id':RoomNode["id"],'relation_type':"INSIDE"})
        if has_obj2==False:
            #print("NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNode2",Node2)
            obs["nodes"].append(Node2)
            obs["nodes"].append(RoomNode)
            obs["edges"].append({'from_id':int(obj2_id),'to_id':RoomNode["id"],'relation_type':"INSIDE"})
        if has_relation==False:
            #print("NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNEdge",relation.upper())
            if relation.upper()=="ON":
                obs["edges"].append({'from_id':int(obj1_id),'to_id':RoomNode["id"],'relation_type':"INSIDE"})
            obs["edges"].append({'from_id':int(obj1_id),'to_id':int(obj2_id),'relation_type':relation.upper()})
        return obs
                # self.belief.update_belief(obs)

    def filtering_graph(self, graph):
        new_edges = []
        edge_dict = {}
        for edge in graph["edges"]:
            key = (edge["from_id"], edge["to_id"])
            if key not in edge_dict:
                edge_dict[key] = [edge["relation_type"]]
                new_edges.append(edge)
            else:
                if edge["relation_type"] not in edge_dict[key]:
                    edge_dict[key] += [edge["relation_type"]]
                    new_edges.append(edge)

        graph["edges"] = new_edges
        return graph

    def sample_belief(self, obs_graph):
        new_graph = self.belief.update_graph_from_gt_graph(obs_graph)
        self.previous_belief_graph = self.filtering_graph(new_graph)
        return new_graph

    def get_relations_char(self, graph):
        # TODO: move this in vh_mdp
        char_id = [
            node["id"] for node in graph["nodes"] if node["class_name"] == "character"
        ][0]
        edges = [edge for edge in graph["edges"] if edge["from_id"] == char_id]
        print("Character:")
        print(edges)
        print("---")

    def new_obs(obs, ignore_ids=None):
        curr_visible_ids = [node["id"] for node in obs["nodes"]]
        relations = {"ON": 0, "INSIDE": 1}
        num_relations = len(relations)
        if set(curr_visible_ids) != set(self.last_obs["ids"]):
            new_obs = True
        else:
            state_ids = np.zeros((len(curr_visible_ids), 4))
            edge_ids = np.zeros(
                (len(curr_visible_ids), len(curr_visible_ids), num_relations)
            )
            idnode2id = []
            for idnode, node in enumerate(nodes):
                idnode2id[node["id"]] = idnode
                state_ids[idnode, 0] = "OPEN" in node["states"]
                state_ids[idnode, 1] = "CLOSED" in node["states"]
                state_ids[idnode, 0] = "ON" in node["states"]
                state_ids[idnode, 1] = "OFF" in node["states"]
            for edge in node["edges"]:
                if edge["relation_type"] in relations.keys():
                    edge_id = relations[edge["relation_type"]]
                    from_id, to_id = (
                        idnode2id[edge["from_id"]],
                        idnode2id[edge["to_id"]],
                    )
                    edge_ids[from_id, to_id, edge_id] = 1

            if ignore_ids != None:
                # We will ignore some edges, for instance if we are grabbing an object
                self.last_obs["edges"][ignore_ids, :] = edge_ids[ignore_ids, :]
                self.last_obs["edges"][:, ignore_ids] = edge_ids[:, ignore_ids]

            if (
                state_ids != self.last_obs["state"]
                or edge_ids != self.last_obs["edges"]
            ):
                new_obs = True
                self.last_obs["state"] = state_ids
                self.last_obs["edges"] = edge_ids
        return new_obs

    def infer_human_goal(self, own_goal_spec, id2node=None):
        if id2node is None:
            id2node = {node["id"]: node for node in self.init_gt_graph["nodes"]}
        goal_predicates = {}
        # if len(self.message_history) > 0:
        id2class = {k: v['class_name'] for k, v in id2node.items()}

        filepath = os.getcwd() + "/../GPT_message/prompt_goal_inference.txt"
        # def infer_goal():
        with open(filepath, "r", encoding='utf-8') as f:
            prompt=f.read()
        # prompt += f"\nHere is the mapping from id to class_name that you will sample targetID and baseItem from: \n {id2class}"
        prompt += f"\nHere is the list of requests for help the human has spoken previously: \n {self.message_history}"
        prompt += f"""\n The name of the goals is formulated as
        'setup_table_k:n' : ['on_plate_k': n, 'on_cutleryfork_k': n, 'on_wineglass_k': n, 'on_waterglass_k': n]
        'get_food_k:n' : ['on_pudding_k': n, 'on_cupcake_k': n, 'on_salmon_k': n, 'on_apple_k': n]
        'load_dishwasher_k:n' : ['inside_plate_k': n, 'inside_cutleryfork_k': n, 'inside_wineglass_k': n, 'inside_waterglass_k': n]
        'stock_fridge_k:n' : ['inside_pudding_k': n,'inside_cupcake_k': n, 'inside_salmon_k': n, 'on_apple_k': n]

        """
        prompt += f"\n Based on the human actions {self.action_history}, and your own subgoal {own_goal_spec}, please infer the most likely goal in the format of 'goal_k:n' where goal is one of [setup_table, get_food, load_dishwasher, stock_fridge] and both k and n are integers. \nOutput: "
        # print(prompt)

        response = gpt_message_encoder.generate_chat_response(prompt, temperature=0).strip("'").strip('"')

        print("goal infer", response)
            # return response
        

        inferred_message = response
        inferred_predicate = inferred_message.split(":")[0]
        count = inferred_message.split(":")[1]
        inferred_correct = True


        count = int(count)
        print(own_goal_spec, "owngoal")
        dest = inferred_predicate.split("_")[-1]
        goal = inferred_predicate.split("_")[0] + "_" + inferred_predicate.split("_")[1]

        subgoal_set = {
            'setup_table' : [f"on_plate_{dest}", f"on_cutleryfork_{dest}", f"on_waterglass_{dest}", f"on_wineglass_{dest}"],
            'get_food' : [f'on_pudding_{dest}',f'on_cupcake_{dest}', f'on_salmon_{dest}', f'on_apple_{dest}'],
            'load_dishwasher' : [f"inside_plate_{dest}", f"inside_cutleryfork_{dest}", f"inside_waterglass_{dest}", f"inside_wineglass_{dest}"],
            'stock_fridge' : [f'inside_pudding_{dest}',f'inside_cupcake_{dest}', f'inside_salmon_{dest}', f'inside_apple_{dest}'],
        }

        for subgoal in subgoal_set[goal]:
            a,b,c = tuple(subgoal.split("_"))
            if subgoal not in own_goal_spec.keys():
                if self.all_names==[]:
                    for node in self.init_gt_graph['nodes']:
                        self.all_names.append(node["class_name"])
                    self.all_names=list(set(self.all_names))

                if b in self.all_names:
                    # continue

                    is_though=False
                    while is_though==False:
                        try:
                            with eventlet.Timeout(10, True):
                                b=gpt_pairing_object.pairing_object(b, self.all_names)
                                #c=gpt_pairing_object.pairing_object(b, self.all_names)
                                is_though=True
                        except:
                            time.sleep(2)

                    goal_predicate=a+"_"+b+"_"+c
                    goal_predicates[goal_predicate] = count

        print("goal_predicates",goal_predicates)
        goal_spec = convert_goal(goal_predicates, self.init_gt_graph)
        print("inferred_human_goal_spec", goal_spec)
        for pred, task in goal_spec.items():
            self.goal_belief_objects[0].extend(task["grab_obj_ids"])

        for pred, task in own_goal_spec.items():
            self.goal_belief_objects[1].extend(task["grab_obj_ids"])
        # print(1/0)
        return goal_spec
    # return None

    def get_action(
        self, obs, messages, opponent_subgoal=None, length_plan=5, must_replan=False, prev_actions=None, return_inferred_goal=False
    ):

        id2node = {node["id"]: node for node in self.init_gt_graph["nodes"]}
        self.id2node=id2node
        #print("----------------------------------------------------------------------------------------",obs)

        #print(messages)
        if prev_actions is not None:
            self.action_history.append(prev_actions)


        if messages is not None and 0 in messages and messages[0] is not None:

            main_message = messages[0]

            is_though = False

            output=gpt_message_decode.message_decode(main_message)

            while is_though==False:
                try:
                    with eventlet.Timeout(7, True):
                        output=gpt_message_decode.message_decode(main_message)

                        is_though=True
                except:
                    time.sleep(2)

            
            output=output.replace('\n', '')
            output=json.loads(output)

            # print("output", output)

            if output["message_type"] == "share_info":

                #print("==================================================",output)
                content=output["content"]                    

                obs = self.update_obs(obs,content)
                self.belief.update_belief(obs)


                    #time.sleep(100000000)
                #print(obs)
                #time.sleep(100000000)

            
            if output["message_type"] == "request_help":
                # if message not in self.message_history:
                    # self.message_history.append(message)
                goal_predicates = {}
                #sentences = message.split(". ")
                #print("==================================================",message)

                command=output["content"]
                #for sentence in sentences:
                if len(command):
                    #if 
                    #print("==================================================Message==================================================",message)

                    goal_predicate,count=command.split(":")
                    count=int(count)
                    a,b,c = tuple(goal_predicate.split("_"))

                    if self.all_names==[]:
                        for node in self.init_gt_graph['nodes']:
                            self.all_names.append(node["class_name"])
                        self.all_names=list(set(self.all_names))
                    is_though=False
                    while is_though==False:
                        try:
                            with eventlet.Timeout(10, True):
                                b=gpt_pairing_object.pairing_object(b, self.all_names)
                                #c=gpt_pairing_object.pairing_object(b, self.all_names)
                                is_though=True
                        except:
                            time.sleep(2)
                    '''
                    if b=="fork":
                        b="cutleryfork"
                    if b=="knife":
                        b="cutleryknife"
                    '''
                    goal_predicate=a+"_"+b+"_"+c
                    
                    #if c=="kitchentable":
                        #print("==================================================Message==================================================",goal_predicate)
                        #time.sleep(10000000000)


                    #goal_predicates[]
                    goal_predicates[goal_predicate] = count
                        
                    #print("===============================",goal_predicate)
                    '''
                    for node in self.init_gt_graph['nodes']:
                        if goal_predicate.split("_")[2]==node["class_name"]:
                            temp_id=node["id"]
                    goal_predicate=goal_predicate.split("_")[0]+"_"+goal_predicate.split("_")[1]+"_"+str(temp_id)
                    '''
                #print("++++++++++++++++++++++++++++++++++++++++++++",goal_predicates)
                print("predicates",goal_predicates)
                self.goal_spec = convert_goal(goal_predicates, self.init_gt_graph)
                print("inferred goal", self.goal_spec)
                self.human_goal = self.infer_human_goal(self.goal_spec)
                #print("++++++++++++++++++++++++++++++++++++++++++++",self.goal_spec)
                # ipdb.set_trace()
        # Create the particles
        # pdb.set_trace()




        '''
        
        THIS IS WHERE WE DO GOAL INFERENCE
        
        '''

        # goal_spec = self.goal_spec

        # print("check", self.goal_spec)
        self.belief.update_belief(obs)

        # if self.human_goal is None:
            

        # additional_goals = self.human_goal
        # if additional_goals is not None:
        #     for k, v in additional_goals.items():
        #         if k not in goal_spec:
        #             goal_spec[k] = v
        

        #print("========================Belief=======================",self.belief.edge_belief)

        # TODO: maybe we will want to keep the previous belief graph to avoid replanning
        # self.sim_env.reset(self.previous_belief_graph, {0: goal_spec, 1: goal_spec})

        last_action = self.last_action
        last_subgoal = self.last_subgoal[0] if self.last_subgoal is not None else None
        subgoals = self.last_subgoal
        last_plan = self.last_plan

        # TODO: is this correct?
        nb_steps = 0
        root_action = None
        root_node = None
        verbose = self.verbose

        # If the current obs is the same as the last obs
        ignore_id = None

        should_replan = True

        goal_ids_all = []
        for goal_name, goal_val in self.goal_spec.items():
            if goal_val["count"] > 0:
                goal_ids_all += goal_val["grab_obj_ids"]

        goal_ids = [
            nodeobs["id"] for nodeobs in obs["nodes"] if nodeobs["id"] in goal_ids_all
        ]
        close_ids = [
            edge["to_id"]
            for edge in obs["edges"]
            if edge["from_id"] == self.agent_id
            and edge["relation_type"] in ["CLOSE", "INSIDE"]
        ]
        plan = []

        if last_plan is not None and len(last_plan) > 0:
            should_replan = False

            # If there is a goal object that was not there before
            next_id_interaction = []
            if len(last_plan) > 1:
                next_id_interaction.append(
                    int(last_plan[1].split("(")[1].split(")")[0])
                )

            new_observed_objects = (
                set(goal_ids)
                - set(self.last_obs["goal_objs"])
                - set(next_id_interaction)
            )
            # self.last_obs = {'goal_objs': goal_ids}
            if len(new_observed_objects) > 0:
                # New goal, need to replan
                should_replan = True
            else:
                visible_ids = {node["id"]: node for node in obs["nodes"]}
                curr_plan = last_plan

                first_action_non_walk = [
                    act for act in last_plan[1:] if "walk" not in act
                ]

                # If the first action other than walk is OPEN/CLOSE and the object is already open/closed...
                if len(first_action_non_walk):
                    first_action_non_walk = first_action_non_walk[0]
                    if "open" in first_action_non_walk:
                        obj_id = int(curr_plan[0].split("(")[1].split(")")[0])
                        if obj_id in visible_ids:
                            if "OPEN" in visible_ids[obj_id]["states"]:
                                should_replan = True
                                print("IS OPEN")
                    elif "close" in first_action_non_walk:
                        obj_id = int(curr_plan[0].split("(")[1].split(")")[0])
                        if obj_id in visible_ids:
                            if "CLOSED" in visible_ids[obj_id]["states"]:
                                should_replan = True
                                print("IS CLOSED")

                if (
                    "open" in last_plan[0]
                    or "close" in last_plan[0]
                    or "put" in last_plan[0]
                    or "grab" in last_plan[0]
                    or "touch" in last_plan[0]
                ):
                    if len(last_plan) == 1:
                        should_replan = True
                    else:
                        curr_plan = last_plan[1:]
                        subgoals = (
                            self.last_subgoal[1:]
                            if self.last_subgoal is not None
                            else None
                        )
                if (
                    "open" in curr_plan[0]
                    or "close" in curr_plan[0]
                    or "put" in curr_plan[0]
                    or "grab" in curr_plan[0]
                    or "touch" in curr_plan[0]
                ):
                    obj_id = int(curr_plan[0].split("(")[1].split(")")[0])
                    if not obj_id in close_ids or not obj_id in visible_ids:
                        should_replan = True

                next_action = not should_replan
                while next_action and "walk" in curr_plan[0]:
                    obj_id = int(curr_plan[0].split("(")[1].split(")")[0])

                    # If object is not visible, replan
                    if obj_id not in visible_ids:
                        should_replan = True
                        next_action = False
                    else:
                        if obj_id in close_ids:
                            if len(curr_plan) == 1:
                                should_replan = True
                                next_action = False
                            else:
                                curr_plan = curr_plan[1:]
                                subgoals = (
                                    subgoals[1:] if subgoals is not None else None
                                )
                        else:
                            # Keep with previous action
                            next_action = False

                if not should_replan:
                    plan = curr_plan

        self.last_obs = {"goal_objs": goal_ids}

        time1 = time.time()

        print("-------- {} --------".format("replan" if should_replan else "no replan"))
        if should_replan or must_replan:
            for particle_id, particle in enumerate(self.particles):
                # belief_states = []
                obs_ids = [node["id"] for node in obs["nodes"]]

                # if True: #particle is None:
                new_graph = self.belief.update_graph_from_gt_graph(
                    obs, resample_unseen_nodes=True, update_belief=False
                )
                # print("new_graph (helper):")
                # # print([n['id'] for n in new_graph['nodes']])
                # print(
                #     [
                #         (n["class_name"], n["id"])
                #         for n in new_graph["nodes"]
                #         if n["class_name"]
                #         in ["kitchen", "bathroom", "livingroom", "bedroom"]
                #     ]
                # )
                init_state = clean_graph(new_graph, self.goal_spec, self.mcts.last_opened, self.belief.id2node)
                satisfied, unsatisfied = utils_env.check_progress2(
                    init_state, self.goal_spec
                )
                init_vh_state = self.sim_env.get_vh_state(init_state)
                # print(colored(unsatisfied, "yellow"))
                self.particles[particle_id] = (
                    init_vh_state,
                    init_state,
                    satisfied,
                    unsatisfied,
                )

                self.particles_full[particle_id] = new_graph
            # print("-----")
            # ipdb.set_trace()
            self.opponent_subgoal = opponent_subgoal

            plan, root_node, subgoals = get_plan(
                self.mcts,
                self.particles,
                self.sim_env,
                nb_steps,
                self.goal_spec,
                last_plan,
                last_action,
                opponent_subgoal,
                length_plan=length_plan,
                verbose=verbose,
                num_process=self.num_processes,
            )

            #print("Opponent_subgoal:",opponent_subgoal)
            print("plan_len",len(plan) )
            print(colored(plan[: min(len(plan), 10)], "cyan"))
            # ipdb.set_trace()
        # else:
        #     subgoals = [[None, None, None], [None, None, None]]
        # if len(plan) == 0 and not must_replan:
        #     ipdb.set_trace()
        #     print("Plan empty")
        #     raise Exception
        if len(plan) > 0:
            action = plan[0]
            action = action.replace("[walk]", "[walktowards]")
        else:
            action = None
        if self.logging:
            info = {
                "plan": plan,
                "subgoals": subgoals,
                "belief": copy.deepcopy(self.belief.edge_belief),
                "belief_room": copy.deepcopy(self.belief.room_node),
            }
            if self.get_plan_states:
                plan_states = []
                env = self.sim_env
                env.pomdp = True
                particle_id = 0
                vh_state = self.particles[particle_id][0]
                plan_states.append(vh_state.to_dict())
                for action_item in plan:
                    try:
                        success, vh_state = env.transition(
                            vh_state, {self.char_index: action_item}
                        )
                    except:
                        pdb.set_trace()
                    plan_states.append(vh_state.to_dict())
                info["plan_states"] = plan_states
            if self.logging_graphs:
                info.update({"obs": obs["nodes"].copy()})
        else:
            info = {"plan": plan, "subgoals": subgoals}
        
        if return_inferred_goal:
            info['inferred_goal'] = self.human_goal

        self.last_action = action
        self.last_subgoal = (
            subgoals if subgoals is not None and len(subgoals) > 0 else None
        )
        self.last_plan = plan
        # print(info['subgoals'])
        # print(action)
        time2 = time.time()
        # print("Time: ", time2 - time1)
        
        args = get_args()
        if args.tp==True and action!=None:
            temp_actions=action.split(" ")
            if temp_actions[0][1:-1]=="walktowards":
                temp_actions[0]="[walk]"
                action=temp_actions[0]+" "+temp_actions[1]+" "+temp_actions[2]
        print("Replanning... ", should_replan or must_replan)
        print("Helper_Goal",self.goal_spec)

        return action, info


    def get_action_for_message(self,merged_minds,length_plan=5):
        #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& How to use belief&goal from merged_mind to return action?
        
        #goal_spec=self.helper_mind.agent_goal
 
        nb_steps = 0
        planner_particles = []
        opponent_subgoal = None
        our_goal = None
        for pID, merged_mind in enumerate(merged_minds):
            if opponent_subgoal is None:
                opponent_subgoal = merged_mind.human_goal
            if our_goal is None:
                our_goal = merged_mind.agent_goal
            working_particle = self.particles[pID][1]
            temp_id2node = {n['id']: n for n in working_particle['nodes']}
            sampled_belief = merged_mind.agent_belief.update_graph_from_gt_graph(
                working_particle, 
                resample_unseen_nodes=True, 
                update_belief=False
            )

            # print('new_graph:')
            # print([n['id'] for n in new_graph['nodes']])
            init_state = sampled_belief
            init_state = clean_graph(sampled_belief, self.goal_spec, self.mcts.last_opened, temp_id2node)
            satisfied, unsatisfied = utils_env.check_progress2(
                init_state, self.goal_spec
            )
            init_vh_state = self.sim_env.get_vh_state(init_state)
            #sampled_vh = self.sim_env.get_vh_state(sampled_belief, merged_mind.agent_belief.name_equivalence, instance_selection=True)

            #satisfied, unsatisfied = utils_env.check_progress2(sampled_belief, self.goal_spec)
            mind_particle = (
                init_vh_state, 
                init_state, 
                satisfied, unsatisfied)
            planner_particles.append(mind_particle)

        plan, root_node, subgoals, action_score_list, action_reward_list, plans_all = get_plan(
                self.mcts,
                planner_particles,
                self.sim_env,
                nb_steps,
                self.goal_spec,
                self.last_plan,
                self.last_action,
                opponent_subgoal,
                length_plan=length_plan,
                verbose=self.verbose,
                num_process=self.num_processes,
                return_score=True
            )

        # discounted futue reward
        expected_reward = 0
        discount_factor = 1
        for plan_idx, chosen_action in enumerate(plan):
            expected_reward += (discount_factor ** plan_idx) * action_reward_list[plan_idx][chosen_action]


        print(colored(plan[: min(len(plan), 10)], "cyan"))

        if len(plan) > 0:
            action = plan[0]
            action = action.replace("[walk]", "[walktowards]")
        else:
            action = None
        
        args = get_args()
        if args.tp==True and action!=None:
            temp_actions=action.split(" ")
            if temp_actions[0][1:-1]=="walktowards":
                temp_actions[0]="[walk]"
                action=temp_actions[0]+" "+temp_actions[1]+" "+temp_actions[2]

        return action, action_score_list, expected_reward, plans_all

    def locate_obj(self, obj, graph):

        id2node = {node["id"]: node for node in graph["nodes"]}
        for edge in graph["edges"]:
            if edge["from_id"]== obj:
                print(obj, edge["relation_type"], edge["to_id"], id2node[edge["to_id"]]["class_name"])

    def gen_helper_message(
        self,
        prev_actions,
        prev_message,
        obs, 
        human_obs, 
        graph

    ):



        if self.human_goal is None:
            return None
            # self.human_goal =  self.infer_human_goal(self.goal_spec)
            # self.helper_goal = self.goal_spec

        id2node = {node["id"]: node for node in self.init_gt_graph["nodes"]}

        self.time_last_asked = self.time_last_asked+1

        # self.init_belief(self.init_gt_graph)

        if len(self.helper_belief) == 0:
            self.init_belief(self.init_gt_graph)
            for item, loc in self.helper_belief.items():
                if item in self.goal_belief_objects[0]:
                    add_belief = loc[1] + "_" + str(item) + "_" + str(loc[0])
                    # edge['from_id']+"_"+edge['from_id']+"_"+edge['to_id']
                    # print(add_belief)
                    obs = self.update_obs(obs, add_belief)
                    self.belief.update_belief(obs)

        #     for i in self.goal_belief_objects[0]:
        #         self.locate_obj(i, graph)

        #     for i in self.goal_belief_objects[1]:
        #         self.locate_obj(i, graph)


            # for edge in self.helper_belief:
            #     add_belief = edge['from_id']+"_"+edge['from_id']+"_"+edge['to_id']
            #     obs = self.update_obs(add_belief)
            #     self.belief.update_belief(obs)

        candidate_categories = ["cutleryfork", "cutleryknife", "pudding", "waterglass", "wineglass","plate", "cupcake","apple", "salmon", "bananas", "lime", "condimentbottle", "chips", "condimentshaker", "peach", "plum", "dishbowl","mug", "chocolatesyrup", "creamybuns", "breadslice", "fryingpan"]


        if self.comm_type == "task-agnostic":
            for edge in obs["edges"]:
                if edge['from_id'] in id2node.keys() and edge['to_id'] in id2node.keys() and "GRABBABLE" in id2node[edge['from_id']]["properties"] and id2node[edge['to_id']]["category"]!="Rooms" and edge['relation_type']!="CLOSE" and edge['relation_type']!="FACING":
                    self.helper_belief[edge['from_id']] = [edge["to_id"], edge['relation_type']]

                    # print(edge['from_id'],self.id2node[edge['from_id']]["class_name"],self.helper_belief[edge['from_id']])

            for edge in human_obs["edges"]:
                if edge['from_id'] in id2node.keys() and edge['to_id'] in id2node.keys() and "GRABBABLE" in self.id2node[edge['from_id']]["properties"] and self.id2node[edge['to_id']]["category"]!="Rooms" and edge['relation_type']!="CLOSE" and edge['relation_type']!="FACING":
                    self.human_belief[edge['from_id']] = [edge["to_id"], edge['relation_type']]
                    # print(edge['from_id'],self.id2node[edge['from_id']]["class_name"], self.human_belief[edge['from_id']])

            comm_items_list = [item for item in list(set(self.helper_belief.keys())- set(self.human_belief.keys())) if id2node[item]["class_name"] in candidate_categories]

            # if len(comm_items_list)>0:
            # for i in range(10):

            if len(comm_items_list)> 0 and np.random.choice([0,1],p=[0.5,0.5])==1:
                item  = np.random.choice(comm_items_list)
                    # if id2node[item][] in candidate_categories:
                # print("communicate", item, self.helper_belief[item] )
                self.human_belief[item] = self.helper_belief[item]

                share_txt = " {}({}) is {} {}({})".format(id2node[item]["class_name"], item, self.helper_belief[item][1], self.id2node[self.helper_belief[item][0]]["class_name"], self.helper_belief[item][0])
                return gpt_message_encoder.generate_chat_response("You are a helpful robot assistant. Please communicate this information to the human agent in a colloquial and concise fashion: \n"+share_txt)


            for item in self.goal_belief_objects[1]:
                if item not in self.helper_belief.keys() and item not in self.ask_item and self.time_last_asked >=10 :
                    # print("Ask", id2node[item]["class_name"])
                    self.ask_item.append(item)
                    self.time_last_asked = 0

                    # self.helper_belief[item] = self.human_belief[item]
                    return gpt_message_encoder.generate_chat_response("You are a helpful robot assistant. You would like to know if the human agent have information about the location of {}({}). Please ask in a colloquial and concise fashion: \n".format(id2node[item]["class_name"], item))



        if self.comm_type == "goal-heuristic":


            for edge in obs["edges"]:
                if edge['from_id'] in id2node.keys() and "GRABBABLE" in id2node[edge['from_id']]["properties"] and edge['from_id'] in self.goal_belief_objects[0] and id2node[edge['to_id']]["category"]!="Rooms" and edge['relation_type']!="CLOSE" and edge['relation_type']!="FACING":
                    self.helper_belief[edge['from_id']] = [edge["to_id"], edge['relation_type']]
                    # print(edge['from_id'],self.helper_belief[edge['from_id']], self.id2node[edge['to_id']]["category"])


            # print("helper_belief",self.helper_belief)


            for edge in human_obs["edges"]:
                if edge['from_id'] in id2node.keys() and "GRABBABLE" in id2node[edge['from_id']]["properties"] and edge['from_id'] in self.goal_belief_objects[0] and edge['relation_type']!="CLOSE" and edge['relation_type']!="FACING":
                    self.human_belief[edge['from_id']] = [edge["to_id"], edge['relation_type']]
                    # print(edge['from_id'],self.human_belief[edge['from_id']])


            # print("human_belief",self.human_belief)


            for item in self.goal_belief_objects[0]:
                if item in self.helper_belief.keys():
                    # print("communicate", item, self.helper_belief[item])
                    self.human_belief[item] = self.helper_belief[item]
                    share_txt = " {}({}) is {} {}({})".format(id2node[item]["class_name"], item, self.helper_belief[item][1], self.id2node[self.helper_belief[item][0]]["class_name"], self.helper_belief[item][0])
                    return generate_chat_response("You are a helpful robot assistant. Please communicate this information to the human agent in a colloquial and concise fashion: \n"+share_txt)

            for item in self.goal_belief_objects[1]:
                if item not in self.helper_belief.keys() and item not in self.ask_item and self.time_last_asked >=10 :
                    # print("Ask", item, self.human_belief[item])
                    self.ask_item.append(item)
                    self.time_last_asked = 0

                    # self.helper_belief[item] = self.human_belief[item]
                    return gpt_message_encoder.generate_chat_response("You are a helpful robot assistant. You would like to know if the human agent have information about the location of {}({}). Please ask in a colloquial and concise fashion: \n".format(id2node[item]["class_name"], item))




        if self.comm_type == "goal":


            for edge in obs["edges"]:

                # print(id2node)
                if edge['from_id'] in id2node.keys() and "GRABBABLE" in id2node[edge['from_id']]["properties"] and edge['from_id'] in self.goal_belief_objects[0] and id2node[edge['to_id']]["category"]!="Rooms" and edge['relation_type']!="CLOSE" and edge['relation_type']!="FACING":
                    self.helper_belief[edge['from_id']] = [edge["to_id"], edge['relation_type']]
                    print(edge['from_id'],self.helper_belief[edge['from_id']], id2node[edge['to_id']]["category"])


            # print("helper_belief",self.helper_belief)


            for edge in human_obs["edges"]:
                if edge['from_id'] in id2node.keys() and "GRABBABLE" in id2node[edge['from_id']]["properties"] and edge['from_id'] in self.goal_belief_objects[0] and edge['relation_type']!="CLOSE" and edge['relation_type']!="FACING":
                    self.human_belief[edge['from_id']] = [edge["to_id"], edge['relation_type']]
                    # print(edge['from_id'],self.human_belief[edge['from_id']])


            # print("human_belief",self.human_belief)


            for item in self.goal_belief_objects[0]:
                if item in self.helper_belief.keys() and item not in self.human_belief.keys():
                    # print("communicate", item, self.helper_belief[item])
                    self.human_belief[item] = self.helper_belief[item]

                    share_txt = " {}({}) is {} {}({})".format(id2node[item]["class_name"], item, self.helper_belief[item][1], self.id2node[self.helper_belief[item][0]]["class_name"], self.helper_belief[item][0])
                    return gpt_message_encoder.generate_chat_response("You are a helpful robot assistant. Please communicate this information to the human agent in a colloquial and concise fashion: \n"+share_txt)

            for item in self.goal_belief_objects[1]:
                if item not in self.helper_belief.keys() and item not in self.ask_item and self.time_last_asked >=10 :
                    # print("Ask", id2node[item]["class_name"])
                    self.ask_item.append(item)
                    self.time_last_asked = 0

                    return gpt_message_encoder.generate_chat_response("You are a helpful robot assistant. You would like to know if the human agent have information about the location of {}({}). Please ask in a colloquial and concise fashion: \n".format(id2node[item]["class_name"], item))




        return None
        

    def reset(
        self,
        observed_graph,
        gt_graph,
        task_goal,
        seed=0,
        simulator_type="python",
        is_alice=False,
    ):
        self.last_action = None
        self.last_subgoal = None
        self.init_gt_graph = gt_graph
        self.action_history = []
        self.human_belief = {}
        self.helper_belief = {}
        self.all_names = []
        self.goal_belief_objects = [[],[]]
        self.id2node = {node["id"]: node for node in self.init_gt_graph["nodes"]}
        """TODO: do no need this?"""
        # if 'waterglass' not in [node['class_name'] for node in self.init_gt_graph['nodes']]:
        #    ipdb.set_trace()
        self.belief = belief.Belief(
            gt_graph,
            agent_id=self.agent_id,
            seed=seed,
            belief_params=self.belief_params,
        )

        #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& Reset each mind here? How?

        self.sim_env.reset(gt_graph)
        add_bp = self.num_processes == 0
        self.mcts = MCTS_particles_v2_instance(
            gt_graph,
            self.agent_id,
            self.char_index,
            self.max_episode_length,
            self.num_simulation,
            self.max_rollout_steps,
            self.c_init,
            self.c_base,
            seed=seed,
            agent_params=self.agent_params,
            add_bp=add_bp,
        )
        self.goal_spec = {}

        # self.mcts.should_close = self.should_close
