import sys
import shutil
import os
import logging
from tqdm import tqdm
import traceback
import os
import ipdb
from pickle5 import pickle
import json
import pdb
import random
import numpy as np
from pathlib import Path
sys.path.append("..")
sys.path.insert(0, f"/scratch2/weka/tenenbaum/lanceyin/GOMA/")
from envs.unity_environment import UnityEnvironment
'''
from agents import (
    MCTS_agent,
    MCTS_agent_particle_v2,
    MCTS_agent_particle,
    MCTS_agent_particle_v2_instance,
    MCTS_agent_particle_v2_instance_human_message,
    MCTS_agent_particle_v2_instance_helper_message,
    CoELA_agent
)
'''
from agents import (
    MCTS_agent_particle_v2_instance,
    MCTS_agent_particle_v2_instance_human_message,
    MCTS_agent_particle_v2_instance_helper_message,
    CoELA_agent
)
from arguments import get_args
from algos.arena_mp2 import ArenaMP
from utils import utils_goals
from utils import utils_exception
from utils import utils_environment as utils_env


def get_class_mode(agent_args):
    mode_str = "{}_opencost{}_closecost{}_walkcost{}_forgetrate{}".format(
        agent_args["obs_type"],
        agent_args["open_cost"],
        agent_args["should_close"],
        agent_args["walk_cost"],
        agent_args["belief"]["forget_rate"],
    )
    return mode_str


if __name__ == "__main__":
    args = get_args()

    save_data = False
    num_tries = 1

    args.executable_file = "/scratch2/weka/tenenbaum/kunaljha/virtualhome/virtualhome/simulation/unity_simulator/linux_exec.v2.2.4.x86_64"

    args.dataset_path = "/scratch2/weka/tenenbaum/lanceyin/GOMA/train_env_task_set.pik"
    
    args.max_episode_length = 100

    cachedir = f"record"

    agent_types = [
        ["full", 0, 0.05, False, 0, "uniform"],  # 0
        ["full", 0.5, 0.01, False, 0, "uniform"],  # 1
        ["full", -5, 0.05, False, 0, "uniform"],  # 2
        ["partial", 0, 0.05, False, 0, "uniform"],  # 3
        ["partial", 0, 0.05, False, 0, "spiked"],  # 4
        ["partial", 0, 0.05, False, 0.2, "uniform"],  # 5
        ["partial", 0, 0.01, False, 0.01, "spiked"],  # 6
        ["partial", -5, 0.05, False, 0.2, "uniform"],  # 7
        ["partial", 0.5, 0.05, False, 0.2, "uniform"],  # 8
    ]
    random_start = random.Random()
    agent_types_index = [3]
    env_task_set = pickle.load(open(args.dataset_path, "rb"))


    curr_count = 0
    total_count = len(agent_types_index) * num_tries * len(env_task_set)
    pbar = tqdm(total=total_count)
    for agent_id in agent_types_index:  # len(agent_types):
        if agent_id in [4]:
            continue
        (
            args.obs_type,
            open_cost,
            walk_cost,
            should_close,
            forget_rate,
            belief_type,
        ) = agent_types[agent_id]
        datafile = args.dataset_path.split("/")[-1].replace(".pik", "")
        agent_args = {
            "obs_type": args.obs_type,
            "open_cost": open_cost,
            "should_close": should_close,
            "walk_cost": walk_cost,
            "belief": {"forget_rate": forget_rate, "belief_type": belief_type},
        }
        # args.mode = "{}_".format(agent_id + 1) + get_class_mode(agent_args)
        # args.mode += 'v9_particles_v2'
        args.mode = "template_main_template_helper"

        for env in env_task_set:
            init_gr = env["init_graph"]

            # print(init_gr)
            gbg_can = [
                node["id"]
                for node in init_gr["nodes"]
                if node["class_name"] in ["garbagecan", "clothespile"]
            ]
            init_gr["nodes"] = [
                node for node in init_gr["nodes"] if node["id"] not in gbg_can
            ]
            init_gr["edges"] = [
                edge
                for edge in init_gr["edges"]
                if edge["from_id"] not in gbg_can and edge["to_id"] not in gbg_can
            ]
            for node in init_gr["nodes"]:
                if node["class_name"] == "cutleryfork":
                    node["obj_transform"]["position"][1] += 0.1

        args.record_dir = "{}/{}/{}".format(cachedir, datafile, args.mode)
        error_dir = "{}/logging/{}_{}".format(cachedir, datafile, args.mode)
        process_dir = "{}/processing/{}_{}".format(cachedir, datafile, args.mode)
        if not os.path.exists(args.record_dir):
            os.makedirs(args.record_dir)

        if not os.path.exists(error_dir):
            os.makedirs(error_dir)

        if not os.path.exists(process_dir):
            os.makedirs(process_dir)

        executable_args = {
            "file_name": args.executable_file,
            "x_display": 0,
            "no_graphics": True,
        }

        id_run = 0
        # random.seed(id_run)
        episode_ids = list(range(len(env_task_set)))
        episode_ids = sorted(episode_ids)
        #random_start.shuffle(episode_ids)
        # episode_ids = episode_ids[10:]

        print("episodes", episode_ids)

        S = [[] for _ in range(len(episode_ids))]
        L = [[] for _ in range(len(episode_ids))]

        test_results = {}
        # episode_ids = [episode_ids[0]]

        # episode_ids = [253]
        def env_fn(env_id):
            return UnityEnvironment(
                num_agents=2,
                max_episode_length=args.max_episode_length,
                port_id=env_id,
                convert_goal=True,
                env_task_set=env_task_set,
                observation_types=[args.obs_type, args.obs_type],
                use_editor=args.use_editor,
                executable_args=executable_args,
                base_port=args.base_port,
            )

        speed_debug = False
        args_common = dict(
            recursive=False,
            max_episode_length=args.max_episode_length,
            num_simulation=100,
            max_rollout_steps=5,
            c_init=0.1,
            c_base=100,
            num_samples=1,
            num_processes=2 if speed_debug else args.num_processes,
            num_particles=2 if speed_debug else args.num_belief_particles,
            logging=True,
            logging_graphs=True,
        )
        if args.obs_type == "full":
            args_common["num_particles"] = 1
        else:
            args_common["num_particles"] = 2 if speed_debug else args.num_belief_particles

        args_agent1 = {"agent_id": 1, "char_index": 0}
        args_agent1.update(args_common)
        args_agent1["agent_params"] = agent_args

        args_agent2 = {"agent_id": 2, "char_index": 1}
        args_agent2.update(args_common)
        args_agent2["agent_params"] = agent_args

        if args.model == "single":
            # args_agent2["comm"] = True
            agents = [
                lambda x, y: MCTS_agent_particle_v2_instance_human_message(**args_agent1),
                lambda x, y: MCTS_agent_particle_v2_instance_helper_message(**args_agent2),
                # lambda x, y: CoELA_agent.vision_LLM_agent(agent_id=args_agent2["agent_id"], char_index=args_agent2["char_index"], args=args),
            ]

        if args.model == "llm":
            # args_agent2["comm"] = True
            agents = [
                lambda x, y: MCTS_agent_particle_v2_instance_human_message(**args_agent1),
                # lambda x, y: MCTS_agent_particle_v2_instance_helper_message(**args_agent2),
                lambda x, y: CoELA_agent.vision_LLM_agent(agent_id=args_agent2["agent_id"], char_index=args_agent2["char_index"], args=args),
            ]
        else:
            agents = [
                lambda x, y: MCTS_agent_particle_v2_instance_human_message(**args_agent1),
                lambda x, y: MCTS_agent_particle_v2_instance_helper_message(**args_agent2),
                # lambda x, y: CoELA_agent.vision_LLM_agent(agent_id=args_agent2["agent_id"], char_index=args_agent2["char_index"], args=args),
            ]

        if args.model == "task-agnostic":
            args_agent2["comm_type"] = "task-agnostic"

        if args.model == "heuristic":
            args_agent2["comm_type"] = "goal-heuristic"

        agent_communicate = args.model!="no_comm"

        print("agent_communicate", agent_communicate)


        arena = ArenaMP(
            args.max_episode_length, id_run, env_fn, agents, save_belief=True, agent_communicate=agent_communicate, single_agent = args.model =="single"
        )

        # # episode_ids = [20] #episode_ids
        num_tries = 2
        steps_list, failed_tasks = [], []

        print(episode_ids)

        for iter_id in tqdm(range(num_tries)):
            # if iter_id > 0:

            cnt = 0

            current_tried = iter_id

            if not os.path.isfile(args.record_dir + "/results_{}.pik".format(0)):
                test_results = {}
            else:
                test_results = pickle.load(
                    open(args.record_dir + "/results_{}.pik".format(0), "rb")
                )

            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            
            # for episode_id in episode_ids[79:]:
            if args.runall == False:
                episode_list =[args.id]
            else:
                episode_list =list(np.arange(len(env_task_set)))

            # for episode_id in list(range(276,286)):
            for episode_id in episode_list:

                curr_count += 1

                log_file_name = args.record_dir + "/logs_episode.{}_iter.{}.pik".format(
                    episode_id, iter_id
                )
                text_log_file_name = args.record_dir + "/logs_episode.{}_iter.{}_reward.txt".format(
                    episode_id, iter_id
                )
                failure_file = "{}/{}_{}.txt".format(error_dir, episode_id, iter_id)
                process_file = "{}/{}_{}.txt".format(process_dir, episode_id, iter_id)

                # if os.path.isfile(process_file):
                #     continue
                # if os.path.isfile(log_file_name):  # or os.path.isfile(failure_file):
                #     continue
                # if os.path.isfile(failure_file):
                #     continue

                with open(process_file, "w+") as f:
                    f.write("process_started")

                fileh = logging.FileHandler(failure_file, "a")
                fileh.setLevel(logging.DEBUG)
                logger.addHandler(fileh)

                proceed = False
                fail_count = 0

                while proceed == False and fail_count <= 3:

                    print("episode:", episode_id)

                    for it_agent, agent in enumerate(arena.agents):
                        agent.seed = (it_agent + current_tried * 2) * 5

                    try:
                        arena.reset(episode_id)
                        env_task = env_task_set[episode_id]
                        # for it_agent, agent in enumerate(arena.agents):
                            # agent.init_belief(env_task["init_graph"])
                        # ipdb.set_trace() \
                        # goal = env_task["task_goal"][0]

                        # for g, c in goal.items():
                        #     goal[g] = 2

                        # goal[list(goal.keys())[0]] = 1

                        agent_goal = utils_env.convert_goal(
                            env_task["task_goal"][0], env_task["init_graph"]
                        )
                        # agent_goal = utils_env.convert_goal(
                        #     goal, env_task["init_graph"]
                        # )

                        noise_goal = None

                        print("Agent Goal", agent_goal)
                        print("Noise Goal", noise_goal)
                        curr_graph = env_task["init_graph"]
                        # print("curr_graph",curr_graph["edges"])
                        id2node = {node["id"]: node for node in curr_graph["nodes"]}
                        container = [
                            edge["from_id"]
                            for edge in curr_graph["edges"]
                            if edge["to_id"] == 72 and edge["relation_type"] != "CLOSE"
                        ]
                        for ct in container:
                            print(id2node[ct]["class_name"])
                        print("========s")
                        success, steps, saved_info = arena.run(
                            pred_goal={0: agent_goal, 1: agent_goal}
                        )

                        print("-------------------------------------")
                        print("success" if success else "failure")
                        print("steps:", steps)

                        # out_dir = "/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/project_lance/watch_talk_help/testing_agents/output_steps/"+ str(args.model) + ".npy"
                        if success:
                            with open(f"/data/vision/torralba/frames/data_acquisition/SyntheticStories/MultiAgent/project_lance/watch_talk_help/testing_agents/output_steps/{args.model}_steps.txt", "a") as text_file:
                                text_file.write("episode "+str(episode_id)+ ": " + str(steps))

                            proceed = True

                        # else:
                        #     fail_count +=1
                        # with open(out_dir, 'rb') as fp:
                        #     meta_dict = pickle.load(fp)

                        # meta_dict[episode_id] = [env_task["task_goal"], steps]

                        # with open(out_dir, 'wb') as fp:
                        #     pickle.dump(meta_dict, fp)

                        print("-------------------------------------")
                        # if not success:
                        #     failed_tasks.append(episode_id)
                        # else:
                        #     steps_list.append(steps)
                        is_finished = 1 if success else 0

                        # Path(args.record_dir).mkdir(parents=True, exist_ok=True)
                        if len(saved_info["obs"]) > 0 and save_data:
                            gt_goals = saved_info['gt_goals']
                            inital_state = saved_info['graph'][0]
                            init_id2node = {x['id']:x for x in inital_state['nodes']}

                            edges_to_use = []
                            nodes_to_use = []
                            acceptable_categories = ['Characters', 'Rooms']
                            for edge in inital_state['edges']:
                                fID = edge['from_id']
                                tID = edge['to_id']
                                if init_id2node[fID]['category'] in acceptable_categories and \
                                    init_id2node[tID]['category'] in acceptable_categories:
                                    if init_id2node[fID] not in nodes_to_use:
                                        nodes_to_use.append(init_id2node[fID])
                                    if init_id2node[tID] not in nodes_to_use:
                                        nodes_to_use.append(init_id2node[tID])
                                    if edge not in edges_to_use:
                                        edges_to_use.append(edge)

                            log_string = f"GT Goals: {saved_info['gt_goals']} \n ----------\n"
                            log_string += "Initial Nodes\n__________\n"
                            for n in nodes_to_use:
                                log_string += f"{n}\n"
                            log_string += "\nInitial Edges\n__________\n"
                            for e in edges_to_use:
                                log_string += f"{e}\n"
                            log_string += "\n\nActions\n__________\n"
                            for k, v in saved_info['action'].items():
                                log_string += f"{k}: {v}\n"
                            log_string += "\n\nMessages\n__________\n"
                            for k, v in saved_info['message'].items():
                                log_string += f"{k}:\n"
                                for m in v:
                                    log_string += f"    {m}\n"
                            with open(text_log_file_name, "w") as text_file:
                                text_file.write(log_string)
                            pickle.dump(saved_info, open(log_file_name, "wb"))
                        else:
                            if save_data:
                                with open(log_file_name, "w+") as f:
                                    f.write(json.dumps(saved_info, indent=4))

                        logger.removeHandler(logger.handlers[0])
                        os.remove(failure_file)

                    except utils_exception.UnityException as e:
                        traceback.print_exc()

                        print("Unity exception")
                        arena.reset_env()
                        # ipdb.set_trace()
                        continue

                    except utils_exception.ManyFailureException as e:
                        traceback.print_exc()

                        print("ERRO HERE")
                        logging.exception("Many failure Error")
                        # print("OTHER ERROR")
                        logger.removeHandler(logger.handlers[0])
                        # exit()
                        # arena.reset_env()
                        print("Dione")
                        # ipdb.set_trace()
                        arena.reset_env()
                        continue

                    except Exception as e:

                        traceback.print_exc()

                        logging.exception("Error")
                        print("OTHER ERROR")
                        logger.removeHandler(logger.handlers[0])
                        # exit()
                        arena.reset_env()

                        continue

                    fail_count +=1
                    S[episode_id].append(is_finished)
                    L[episode_id].append(steps)
                    test_results[episode_id] = {"S": S[episode_id], "L": L[episode_id]}

            # ipdb.set_trace()
            # pickle.dump(test_results, open(args.record_dir + '/results_{}.pik'.format(0), 'wb'))
            print(
                "average steps (finishing the tasks):",
                np.array(steps_list).mean() if len(steps_list) > 0 else None,
            )

    pbar.close()
