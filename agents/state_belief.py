import numpy as np
from . import belief
import pdb
import sys
sys.path.append("..")
import utils
from envs.graph_env import VhGraphEnv
import _pickle as pickle
from evolving_graph.environment import Relation

def obs_intersection(base_obs, target_obs):
    '''
    Finds overlapping obs from base to target and returns new obs
    '''
    base_id2node = {n['id']: n for n in base_obs['nodes']}
    target_id2node = {n['id']: n for n in target_obs['nodes']}
    nodes = [target_id2node[n] for n in target_id2node if n in base_id2node]
    for aID in [1, 2]:  # add agent location if agent is missing from obs
        if (aID in target_id2node and aID not in base_id2node):
            nodes.append(target_id2node[aID])
        elif (aID in base_id2node and aID not in target_id2node):
            nodes.append(base_id2node[aID])
        else:
            pass

    partial_id2node = {n['id']: n for n in nodes}
    edges = []
    # only add edges for nodes both agents can see
    for e in target_obs['edges']:
        if e['from_id'] in partial_id2node and e['to_id'] in partial_id2node:
            edges.append(e)
    for e in base_obs['edges']:
        if e['from_id'] in partial_id2node and e['to_id'] in partial_id2node:
            edges.append(e)

    return {'nodes':nodes, 'edges':edges}

def make_partial_belief(base_obs, target_id, target_obs, belief_params, sim_env, target_action_dict):
    '''
    Given partial observation of one agent and another, make the belief the base has about the target's POV
    '''
    try:  # try conventional masking
        partial_obs_graph = sim_env._mask_state(base_obs, target_id)
    except:  # otherwise mask ourselves
        partial_obs_graph = obs_intersection(base_obs, target_obs)

    id2node = {n['id']:n for n in partial_obs_graph['nodes']}

    if target_action_dict is not None and len(target_action_dict) > 0:
        temp = {target_id: target_action_dict[target_id - 1]}

        if temp[target_id] is not None:
            # this is in case we are acting on a node we haven't sampled
            try:
                action_object_id = int(temp[target_id].split("(")[1].split(")")[0])
            except:
                pdb.set_trace()
            if action_object_id not in id2node:
                action_node = None
                for n in target_obs['nodes']:
                    if n['id'] == action_object_id:
                        action_node = n
                        break
                if action_node is None:
                    for n in base_obs['nodes']:
                        if n['id'] == action_object_id:
                            action_node = n 
                            break
                partial_obs_graph['nodes'].append(action_node)
                id2node[action_object_id] = action_node

                # only add edges for nodes both agents can see
                for e in target_obs['edges']:
                    if e['from_id'] in id2node and e['to_id'] in id2node and (e['from_id'] == action_object_id or e['to_id'] == action_object_id):
                        partial_obs_graph['edges'].append(e)
                for e in base_obs['edges']:
                    if e['from_id'] in id2node and e['to_id'] in id2node and (e['from_id'] == action_object_id or e['to_id'] == action_object_id):
                        partial_obs_graph['edges'].append(e) 
    partial_belief = belief.Belief(
        graph_gt=partial_obs_graph, 
        agent_id=target_id,
        belief_params=belief_params
    )

    return partial_belief

def simulate_partial_belief(otherID, partial_belief, other_goal_spec, prev_action_dict, sim_env, transition_env):
    '''
    Simulates our belief one step forward
    '''
    # vh_sampled, belief_prob = partial_belief.sample_from_belief(as_vh_state=True, return_prob=True)
    # sampled_belief = vh_sampled.to_dict()
    for num_loops in range(2):

        sampled_belief = partial_belief.update_graph_from_gt_graph(
            partial_belief.graph_init, 
            resample_unseen_nodes=True, 
            update_belief=False)
        vh_sampled = sim_env.get_vh_state(sampled_belief, partial_belief.name_equivalence, instance_selection=True)
        # for i in [1, 2]:
        #     #pdb.set_trace()
        #     vh_sampled.get_node_ids_from(i, Relation.INSIDE)  
        
        transition_env.id2node_env = {node["id"]: node for node in sampled_belief["nodes"]}

        temp = {otherID-1: prev_action_dict[otherID - 1]}
        if prev_action_dict[otherID - 1] is None:
            next_time_belief = vh_sampled.to_dict()
            break
        try:
            success, next_vh_state = sim_env.transition(vh_sampled, temp)
        except:
            pdb.set_trace()
        next_time_belief = next_vh_state.to_dict()
        if success:
            break
        else:
            print(f"\n \n Looping simulate_partial_belief again \n \n ")

    # if transition_env.env is None:
    #     transition_env.env = VhGraphEnv(n_chars=3)
    #     transition_env.env.pomdp = False
    #     transition_env.env.reset(pickle.loads(pickle.dumps(sampled_belief)))
    # (
    #     success,
    #     next_vh_state,
    #     next_time_belief,
    #     cost,
    #     curr_reward,
    # ) = transition_env.transition(
    #     vh_sampled, temp, other_goal_spec
    # )

    partial_belief.update_belief(next_time_belief, update_mind=True)
