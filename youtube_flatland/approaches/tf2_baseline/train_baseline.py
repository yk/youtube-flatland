import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dqn import DQNAgent
from environment.tf2_env.fl_environment import FlatlandEnv

# Keep track of last 100 results
win_rate = deque(maxlen=100)
episode_duration = deque(maxlen=100)

# Params
n_episodes = 500
n_actions = 4 # we are ignoring action 0 (for now)
n_agents = 5
n_nodes = 2 # depth of the tree observation

feats = 'distance' # quick fix for selecting 'all' features vs just 'distance' to target (latter is good for single-agent baselines)
n_feats = {
    'all' : 11,
    'distance' : 1
}

# Dimensions of flatland map
x_dim = 32 
y_dim = 32 
# Maximum number of steps per episode
max_steps = 8 * (x_dim + y_dim) - 1
# How often to update the model
learn_every = 10
# Total feature dimension
total_feats = n_feats[feats] * sum([4**i for i in range(n_nodes+1)])

# Flatland Environment
environment = FlatlandEnv(
        x_dim=x_dim,
        y_dim=y_dim,
        n_cars=n_agents, 
        n_acts=n_actions, 
        min_obs=-1.0, 
        max_obs=1.0, 
        n_nodes=n_nodes,
        feats=feats
) 

# Simple DQN agent
agent = DQNAgent(
    alpha=0.0005, 
    gamma=0.99, 
    epsilon=1.0, 
    input_shape=total_feats, 
    batch_size=512, 
    n_actions=n_actions
)

# Buffer for storing action probabilities over time
action_probs = [1] * n_actions
# Time entire training
total = time.time()

# Train for 300 episodes
for episode in range(n_episodes):
    # Time each episode
    start = time.time()

    # Initialize episode
    steps = 0
    all_done = False

    # Reset env and get initial observation
    old_states, info = environment.reset()

    while not all_done and steps < max_steps:
        # Clear action buffer
        all_actions = [None] * environment.n_cars

        # Pick action for each agent
        for agent_id in range(environment.n_cars):
            action = agent.choose_action(old_states[agent_id])
            all_actions[agent_id] = action
            action_probs[action] += 1

        # Perform actions in environment
        states, reward, terminal, info = environment.step(action=all_actions)

        # Store taken actions
        for agent_id in range(environment.n_cars):
            # If agent took an action or completed
            if all_actions[agent_id] is not None or terminal[agent_id]:
                # Add state to memory
                agent.remember(old_states[agent_id], all_actions[agent_id], reward[agent_id], states[agent_id], terminal[agent_id])
        
        # Learn
        if (steps + 1) % learn_every == 0:
            agent.learn()

        # Update old states        
        old_states = states

        # Calculate percentage complete
        perc_done = [v for k, v in terminal.items() if k is not '__all__'].count(True)/environment.n_cars

        # We done yet?
        all_done = terminal['__all__']
        steps += 1
    
    # Episode stats
    episode_duration.append(time.time() - start)
    win_rate.append(perc_done or 0)
    print(f'Episode: {episode+1} Last 100 win rate: {np.mean(win_rate)}')
    print(f'Action probs: {np.array(action_probs)/np.sum(np.array(action_probs))}')
    print(f'Average Episode duration: {np.mean(episode_duration):.2f}s')

agent.save_model()
environment.close()
print(f'TOTAL TIME: {time.time() - total}')