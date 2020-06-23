import numpy as np
from utils.flatland_training.src.tree_observation import ACTIONS

EMPTY_NODE = np.array([0] * 11)


def norm_obs(obs):
    return (obs - np.mean(obs)) / max(1, np.std(obs))

def norm_obs_clip(obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
    if fixed_radius > 0:
          max_obs = fixed_radius
    else: max_obs = np.max(obs[np.where(obs < 1000)], initial=1) + 1

    min_obs = np.min(obs[np.where(obs >= 0)], initial=max_obs) if normalize_to_range else 0

    if max_obs == min_obs:
          return np.clip(obs / max_obs, clip_min, clip_max)
    else: return np.clip((obs - min_obs) / np.abs(max_obs - min_obs), clip_min, clip_max)


def create_tree_features(node, depth, max_depth, data):
    if node == -np.inf:
        num_remaining_nodes = (4 ** (max_depth - depth + 1) - 1) // (4 - 1)
        data.extend([EMPTY_NODE] * num_remaining_nodes)

    else:
        data.append(np.array(tuple(node)[:-2]))
        if node.childs:
            for direction in ACTIONS:
                create_tree_features(node.childs[direction], depth + 1, max_depth, data)

    return data


def normalize_observation(tree, max_depth, observation_radius=0, feats='all'):
    data = np.concatenate(create_tree_features(tree, 0, max_depth, [])).reshape((-1, 11))

    obs_data = norm_obs(norm_obs_clip(data[:,:6].flatten()))
    distances = norm_obs_clip(data[:,6], normalize_to_range=True)
    agent_data = norm_obs(np.clip(data[:,7:].flatten(), -1, 1))
    
    # This is a bit of a hack, will fix, I promise
    if feats == 'all':
        return np.concatenate((obs_data, distances, agent_data))
    elif feats == 'distance':
        return distances
