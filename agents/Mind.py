from . import belief
import time
import numpy as np
import scipy
import copy
import pdb

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& Anymore things for this class?

class Mind:
    def __init__(self, agent_belief, agent_goal, human_goal):
        self.agent_belief=agent_belief
        self.agent_goal=agent_goal
        self.human_goal=human_goal
        
def merge_minds(mind_list,mind_type_list,obs,id2node,init_gt_graph):
    #total_mind=Mind(agent_belief=None,agent_goal=None,human_goal=None)
    first_mind=copy.deepcopy(mind_list[0])
    #first_mind=Mind(agent_belief=mind_list[0].agent_belief,agent_goal=mind_list[0].agent_goal,human_goal=mind_list[0].human_goal)

    belief_list=[]
    agent_goal_list=[]
    relevant_goal_ids = []
    for i in mind_list:
        belief_list.append(i.agent_belief)
        agent_goal_list.append(i.agent_goal)
        if i.agent_goal is None:
            continue
        for k, ele in i.agent_goal.items():
            base_id = int(k.split("_")[-1])
            relevant_goal_ids.append(base_id)
            for other_relevant in ele['grab_obj_ids']:
                relevant_goal_ids.append(other_relevant)

    merged_belief_edges, output_node_belief = merge_belief(belief_list, mind_type_list, relevant_goal_ids)
    # for n in output_node_belief:
    #     obs['nodes'].append(id2node[n])


    for key, edge_distrib in merged_belief_edges.items():
        max_inside_place_1=np.argmax(merged_belief_edges[key]['INSIDE'][1])
        max_on_place_1=np.argmax(merged_belief_edges[key]['ON'][1])

        if (max_inside_place_1!=0):
            if (merged_belief_edges[key]['INSIDE'][1][max_inside_place_1]>=np.log(0.9)):
                obs=add_edge_obs(obs,key,merged_belief_edges[key]['INSIDE'][0][max_inside_place_1],"INSIDE",id2node,init_gt_graph)

        if (max_on_place_1!=0):
            if (merged_belief_edges[key]['ON'][1][max_on_place_1]>=np.log(0.9)):
                obs=add_edge_obs(obs,key,merged_belief_edges[key]['ON'][0][max_on_place_1],"ON",id2node,init_gt_graph)

    # # adding goal if it doesn't already exist as a node
    # obs_id2node = [n['id'] for n in obs['nodes']]
    # for goal in agent_goal_list:
    #     if goal is None:
    #         continue
    #     toAdd = []
    #     for key, elements in goal.items():
    #         target = int(key.split("_")[-1])  # what node do we need to check for
    #         if target not in obs_id2node:
    #             toAdd.append(target)
    #         for grabbed_obj_id in elements['grab_obj_ids']:
    #             if grabbed_obj_id not in obs_id2node:
    #                 toAdd.append(grabbed_obj_id)
    #     for belief in belief_list:
    #         if belief is None:
    #             continue
    #         edge_list = belief.graph_init['edges']
    #         for edge in edge_list:  # only add edges that are most essential
    #             if edge not in obs['edges']:
    #                 from_id = edge['from_id']
    #                 to_id = edge['to_id']
    #                 if (from_id in toAdd):
    #                     if (to_id in obs_id2node or to_id in toAdd):
    #                         obs['edges'].append(edge)
    #                 elif (to_id in toAdd):
    #                     if (from_id in obs_id2node or from_id in toAdd):
    #                         obs['edges'].append(edge)
    #                 else:
    #                     pass
    #     for nID in toAdd:
    #         obs['nodes'].append(id2node[nID])
    #     for nID in relevant_goal_ids:
    #         if nID not in obs['nodes']:
    #             obs['nodes'].append(id2node[nID])
    #print(obs)

    first_mind.agent_belief.update_belief(obs)
    merged_goal=merge_goal(agent_goal_list,mind_type_list)
    return Mind(agent_belief=first_mind.agent_belief,agent_goal=merged_goal,human_goal=first_mind.human_goal) 

def add_edge_obs(obs,id1,id2,relationship,id2node,init_gt_graph):
    has_obj1 = False 
    has_obj2 = False
    for node in obs["nodes"]:
        if node["id"]==id1:
            has_obj1=True
        if node["id"]==id2:
            has_obj2=True

    has_relation = False
    if has_obj1 and has_obj2:
        has_relation=True


    Node1=id2node[id1]
    Node2=id2node[id2]

    for edge in init_gt_graph["edges"]:
        if edge["from_id"]==id2:
            if id2node[edge["to_id"]]["category"]=="Rooms":
                RoomNode=id2node[edge["to_id"]]
                break

    if has_obj1==False:
        obs["nodes"].append(Node1)
    if has_obj2==False:
        obs["nodes"].append(Node2)
        if id2!=RoomNode["id"]:
            obs["nodes"].append(RoomNode)
            obs["edges"].append({'from_id':id2,'to_id':RoomNode["id"],'relation_type':"INSIDE"})
    if has_relation==False:
        if relationship.upper()=="ON":
            obs["edges"].append({'from_id':id1,'to_id':RoomNode["id"],'relation_type':"INSIDE"})
        obs["edges"].append({'from_id':id1,'to_id':id2,'relation_type':relationship})

    return obs

#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& How to merge belief?
def merge_belief(belief_list, mind_type_list, relevant_goal_ids):
    assert len(belief_list) == len(mind_type_list)
    assert len(belief_list) > 1
    # we add in order of common_mind, helper mind, human mind
    prioritized_minds_idx = []
    for priority in ['common_mind', 'helper_mind', 'human_mind']:
        if priority in mind_type_list:
            mind_idx = mind_type_list.index(priority)
            mind = belief_list[mind_idx]
            if mind is not None:
                prioritized_minds_idx.append(mind_idx)

    output_edge_belief={}
    output_node_belief = set([])
    for belief_idx in prioritized_minds_idx: # first pass only deals with high confidence things
        belief = belief_list[belief_idx]
        for key, edge_distrib in belief.edge_belief.items():
            output_node_belief.add(key)  # add all nodes
            for other_node in edge_distrib['INSIDE'][0]:
                if other_node is not None:
                    output_node_belief.add(other_node)
            for other_node in edge_distrib['ON'][0]:
                if other_node is not None:
                    output_node_belief.add(other_node)

            if key in relevant_goal_ids:
                output_edge_belief[key] = edge_distrib
                continue

            max_inside_prob = np.max(edge_distrib['INSIDE'][1])
            max_on_prob = np.max(edge_distrib['ON'][1])

            if (max_inside_prob >= np.log(0.9)) or (max_on_prob >= np.log(0.9)):  # if we're confident about an edge relation
                if key not in output_edge_belief:  # add edges in terms of priority
                    output_edge_belief[key] = edge_distrib

    for belief_idx in prioritized_minds_idx:  # second pass add low confidence based on priority heuristic
        for key, edge_distrib in belief_list[belief_idx].edge_belief.items():
            if key not in output_edge_belief:
                output_edge_belief[key] = edge_distrib

    # for add_n in output_node_belief:  # adding low confidence edges for nodes
    #     if add_n not in output_edge_belief:
    #         for belief_idx in prioritized_minds_idx:
    #             if add_n in belief_list[belief_idx].edge_belief:
    #                 output_edge_belief[add_n] = belief_list[belief_idx].edge_belief[add_n]
    #                 break 
    
    return output_edge_belief, output_node_belief  # return the union of all confident beliefs


def KL_divergence_between_actions(action_score_list_1, action_score_list_2):
    '''
    list of dictionaries mapping action to score for action
    '''
    total_kl_div = torch.tensor(0.0)
    
    max_id = min(len(action_score_list_1), len(action_score_list_2))

    for plan_id, action_score_dict_1 in enumerate(action_score_list_1):
        if plan_id == max_id:
            break
        action_score_dict_2 = action_score_list_2[plan_id]
        action_list = set([])
        action_list = list(action_list.union(set(action_score_dict_1.keys()), set(action_score_dict_2.keys())))
        prob_dict = {}
        for prob_name, score_dict in [('p', action_score_dict_1), ('q', action_score_dict_2)]:
            action_log_probs = np.full((len(action_list),), 0)
            for a, score in score_dict.items():
                action_log_probs[action_list.index(a)] = score
            uniform_probs = [1 / len(action_log_probs)] * len(action_log_probs)

            if scipy.special.logsumexp(action_log_probs) < np.log(1e-6):
                action_probs = uniform_probs
            else:
                action_log_probs_normalized = action_log_probs - scipy.special.logsumexp(
                    action_log_probs
                )
                action_probs = np.exp(action_log_probs_normalized)
                action_probs = action_probs / np.sum(action_probs)
            prob_dict[prob_name] = torch.tensor(action_probs)
        total_kl_div += F.kl_div(prob_dict['q'].log(), prob_dict['p'], reduction='sum')

    res = total_kl_div.item()
    if res > 0:
        print(f"\n \n KL IS {res}; \n Action_score_list 1 = {action_score_list_1}; \n Action_score_list 2 =  {action_score_list_2} \n \n")

    return res






# def merge_belief(belief_list,mind_type_list):
#     is_sure_belief=[]
#     belief_keys=[]

#     for key in belief_list[0].edge_belief.keys():
#         belief_keys.append(key)

#     for i in range(len(belief_list)):
#         this_belief=belief_list[i]

#         is_sure_belief_temp=[]
#         for key in this_belief.edge_belief.keys():

#             is_sure_belief1=None
#             max_inside_place_1=np.argmax(this_belief.edge_belief[key]['INSIDE'][1])
#             max_on_place_1=np.argmax(this_belief.edge_belief[key]['ON'][1])
#             # if we're confident about this edge existing, add it to the beliefs
#             if ( ((this_belief.edge_belief[key]['INSIDE'][1][max_inside_place_1]>=0.999)) or 
#                ((this_belief.edge_belief[key]['ON'][1][max_on_place_1]>=0.999)) ) :
#                 is_sure_belief1=True
#             else:
#                 is_sure_belief1=False
#             is_sure_belief_temp.append(is_sure_belief1)

#         is_sure_belief.append(is_sure_belief_temp)
    
#     output_edge_belief={}

#     for j in range(len(is_sure_belief[0])):
#         key=belief_keys[j]

#         for i in range(len(belief_list)):
#             if (is_sure_belief[i][j]==True) and (mind_type_list[i]=="common_mind"):
#                 output_edge_belief[key]=belief_list[i].edge_belief[key]
#                 cnt1+=1
#                 break

#         if key not in output_edge_belief.keys():
#             for i in range(len(belief_list)):
#                 if (is_sure_belief[i][j]==True) and (mind_type_list[i]=="helper_mind"):
#                     output_edge_belief[key]=belief_list[i].edge_belief[key]
#                     cnt2+=1
#                     break

#         if key not in output_edge_belief.keys():
#             for i in range(len(belief_list)):
#                 if (is_sure_belief[i][j]==True) and (mind_type_list[i]=="human_mind"):
#                     output_edge_belief[key]=belief_list[i].edge_belief[key]
#                     cnt3+=1
#                     break

#         if key not in output_edge_belief.keys():
#             for i in range(len(belief_list)):
#                 if (mind_type_list[i]=="helper_mind"):
#                     output_edge_belief[key]=belief_list[i].edge_belief[key]
#                     break

#         if key not in output_edge_belief.keys():
#             for i in range(len(belief_list)):
#                 if (mind_type_list[i]=="human_mind"):
#                     output_edge_belief[key]=belief_list[i].edge_belief[key]
#                     break
#     return output_edge_belief
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& When to merge goal?
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& How to merge goal?
def merge_goal(agent_goal_list,mind_type_list):
    return agent_goal_list[0]

def get_common_belief(belief1, belief2):
    pass

def get_common_goal(goal1, goal2):
    pass

def get_common_mind(human_mind, agent_mind):
    common_mind=Mind(agent_belief=None,agent_goal=None,human_goal=None)

    common_mind.agent_belief=get_common_belief(human_mind.agent_belief,agent_mind.agent_belief)
    common_mind.human_goal=human_mind.human_goal
    common_mind.agent_goal=get_common_goal(human_mind.agent_goal,agent_mind.agent_goal)

    return common_mind
