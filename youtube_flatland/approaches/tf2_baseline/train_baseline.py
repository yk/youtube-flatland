print('''

WELCOME TO FLATLAND      (   )
GET READY TO TRAIN       ====        ________                ___________
YOUR FIRST           _D _|  |_______/        \__I_I_____===__|_________|
BASELINE                 |(_)---  |   H\________/ |   |        =|___ ___|      ____
                      /     |  |   H  |  |     |   |         ||_| |_||     _|
                     |      |  |   H  |__--------------------| [___] |   =|
                     | ________|___H__/__|_____/[][]~\_______|       |   -|
                     |/ |   |-----------I_____I [][] []  D   |=======|____|_____
                   __/ =| o |=-~~\  /~~\  /~~\  /~~\ ____Y___________|__|_______
                    |/-=|___|=    ||    ||    ||    |_____/~\___/          |_D__
                     \_/      \O=====O=====O=====O_/      \_/               \_/
''')


import os, sys
# Add root to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import argparse
import time
import numpy as np
from collections import deque
from dqn import DQNAgent
from environment.tf2_env.fl_environment import FlatlandEnv

def parse_args():
    parser = argparse.ArgumentParser(
        description='''A script demonstrating the usage of Tensorflow 2 to train a simple DQN agent in Flatland'''
    )
    parser.add_argument('--eval', help="Path to saved model for evaluation")
    parser.add_argument('-e', '--n_episodes', type=int, default=200, help='Number of training episodes')
    parser.add_argument('-a', '--n_agents', type=int, default=1, help='Number of agents')
    parser.add_argument('-n', '--n_nodes', type=int, default=1, help='Depth of observation tree')
    parser.add_argument('-s', '--save_path', default='dqn_model.hdf5', help="Path to save best model to (relative to root)")
    parser.add_argument('-f', '--feats', default='distance', help="Use all features or only distance ( all | distance )")
    parser.add_argument('-d', '--dims', default=(35,35), nargs='+', type=int, help="X and Y dimensions of flatland environment")
    parser.add_argument('-l', '--learn_every', default=10, type=int, help="Agent learns every time this many steps are taken.")
    return parser.parse_args()


def is_training(args):
    return True if args.eval is None else False

def main_loop(environment, agent, max_steps=None, args=None):
    '''
    Training / Testing loop
    '''
    training = is_training(args)
    # Keep track of last 100 results
    win_rate = deque(maxlen=100)
    episode_duration = deque(maxlen=100)
    # Buffer for storing action probabilities over time
    action_probs = [1] * len(agent.action_space)
    # Time entire training
    total = time.time()

    # Train for 300 episodes
    for episode in range(args.n_episodes):
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

            if training:
                # Store taken actions
                for agent_id in range(environment.n_cars):
                    # If agent took an action or completed
                    if all_actions[agent_id] is not None or terminal[agent_id]:
                        # Add state to memory
                        agent.remember(
                            state = old_states[agent_id],
                            action = all_actions[agent_id],
                            reward = reward[agent_id],
                            new_state = states[agent_id],
                            done = terminal[agent_id]
                            )
                
                # Learn
                if (steps + 1) % args.learn_every == 0:
                    agent.learn()
            else:
                environment.render()

            # Update old states        
            old_states = states

            # Calculate percentage complete
            perc_done = [v for k, v in terminal.items() 
                            if k is not '__all__'].count(True) / environment.n_cars

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


def build(args):
    # Params
    training = is_training(args)
    # Hack for switching number of DQN input features (see help)
    n_feats = {
        'all' : 11,
        'distance' : 1
    }
    n_actions = 4 # we are ignoring action 0 (for now)

    # Maximum number of steps per episode
    max_steps = 8 * (args.dims[0] + args.dims[1]) - 1
    # Total feature dimension
    total_feats = n_feats[args.feats] * sum([4**i for i in range(args.n_nodes+1)])

    # Flatland Environment
    environment = FlatlandEnv(
        x_dim=args.dims[0],
        y_dim=args.dims[1],
        n_cars=args.n_agents, 
        n_acts=n_actions, 
        min_obs=-1.0, 
        max_obs=1.0, 
        n_nodes=args.n_nodes,
        feats=args.feats
    ) 

    # Simple DQN agent
    agent = DQNAgent(
        alpha=0.0005, 
        gamma=0.99, 
        epsilon=1.0, 
        input_shape=total_feats, 
        sample_size=512, 
        batch_size=32,
        n_actions=n_actions,
        training=training
    )

    if not training:
        agent.load_model()

    return environment, agent, max_steps


if __name__ == "__main__":  

    args = parse_args()
    environment, agent, max_steps = build(args)
    main_loop(environment, agent, max_steps=max_steps, args=args)
