import cv2
import time
import torch
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import deque, namedtuple

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.utils.rendertools import RenderTool

from dueling_double_dqn import Agent
from tree_observation import TreeObservation
from observation_utils import normalize_observation


parser = argparse.ArgumentParser(description="Train an agent in the flatland environment")

# Task parameters
parser.add_argument("--train", default=True, action='store_true', help="Whether the model should be trained or not")
parser.add_argument("--load-from-checkpoint", default=False, action='store_true', help="Whether to load the model from the last checkpoint")
parser.add_argument("--report-interval", type=int, default=100, help="Iterations between reports")
parser.add_argument("--render-interval", type=int, default=0, help="Iterations between renders")
parser.add_argument("--grid-width", type=int, default=35, help="Number of columns in the environment grid")
parser.add_argument("--grid-height", type=int, default=35, help="Number of rows in the environment grid")

# Training parameters
parser.add_argument("--num-episodes", type=int, default=10000, help="Number of episodes to train for")
parser.add_argument("--num-agents", type=int, default=1, help="Number of agents in each episode")
parser.add_argument("--tree-depth", type=int, default=1, help="Depth of the observation tree")
parser.add_argument("--epsilon-decay", type=float, default=0.997, help="Decay factor for epsilon-greedy exploration")

flags = parser.parse_args()
project_root = Path(__file__).resolve().parent.parent


# Load in the precomputed railway networks. If you want to generate railways on the fly, comment these lines out.
with open(project_root / f'railroads/rail_networks_{flags.num_agents}x{flags.grid_width}x{flags.grid_height}.pkl', 'rb') as file:
    data = pickle.load(file)
    rail_networks = iter(data)
    print(f"Loading {len(data)} railways...")
with open(project_root / f'railroads/schedules_{flags.num_agents}x{flags.grid_width}x{flags.grid_height}.pkl', 'rb') as file:
    schedules = iter(pickle.load(file))

rail_generator = lambda *args: next(rail_networks)
schedule_generator = lambda *args: next(schedules)

# speed_ration_map = {
#     1.: 0.,        # Fast passenger train
#     1. / 2.: 1.0,   # Fast freight train
#     1. / 3.: 0.0,   # Slow commuter train
#     1. / 4.: 0.0 }  # Slow freight train
#
# rail_generator = sparse_rail_generator(grid_mode=False, max_num_cities=3, max_rails_between_cities=3, max_rails_in_city=3)
# schedule_generator = sparse_schedule_generator(speed_ration_map)


# Helper function to render the environment
def render(env_renderer):
    env_renderer.render_env(show_observations=False)
    cv2.imshow('Render', cv2.cvtColor(env_renderer.get_image(), cv2.COLOR_BGR2RGB))
    cv2.waitKey(100)


# Main training loop
def main():
    np.random.seed(1)

    env = RailEnv(width=flags.grid_width, height=flags.grid_height, number_of_agents=flags.num_agents,
                  rail_generator=rail_generator,
                  schedule_generator=schedule_generator,
                  malfunction_generator_and_process_data=malfunction_from_params(MalfunctionParameters(1 / 8000, 15, 50)),
                  obs_builder_object=TreeObservation(max_depth=flags.tree_depth))

    # After training we want to render the results so we also load a renderer
    env_renderer = RenderTool(env, gl="PILSVG")

    # Calculate the state size based on the number of nodes in the tree observation
    num_features_per_node = env.obs_builder.observation_dim
    num_nodes = sum(np.power(4, i) for i in range(flags.tree_depth + 1))
    state_size = num_nodes * num_features_per_node
    action_size = 5

    # Now we load a double dueling DQN agent and initialize it from the checkpoint
    agent = Agent(state_size, action_size)
    if flags.load_from_checkpoint:
          start, eps = agent.load(project_root / 'checkpoints', 0, 1.0)
    else: start, eps = 0, 1.0

    # And some variables to keep track of the progress
    action_dict, final_action_dict = {}, {}
    scores_window, steps_window, done_window = deque(maxlen=200), deque(maxlen=200), deque(maxlen=200)
    action_prob = [0] * action_size
    agent_obs = [None] * flags.num_agents
    agent_obs_buffer = [None] * flags.num_agents
    agent_action_buffer = [2] * flags.num_agents

    max_steps = int(8 * (flags.grid_width + flags.grid_height))
    update_values = False
    start_time = time.time()

    # We don't want to retrain on old railway networks when we restart from a checkpoint, so we just loop
    # through the generators to get all the old networks out of the way
    if start > 0: print(f"Skipping {start} railways")
    for _ in range(0, start):
        rail_generator()
        schedule_generator()

    # Start the training loop
    for episode in range(start + 1, flags.num_episodes + 1):
        env_renderer.reset()
        obs, info = env.reset(True, True)
        score, steps_taken = 0, 0

        # Build agent specific observations
        for a in range(flags.num_agents):
            if obs[a]:
                agent_obs[a] = normalize_observation(obs[a], flags.tree_depth)
                agent_obs_buffer[a] = agent_obs[a].copy()

        # Run episode
        for step in range(max_steps):
            for a in range(flags.num_agents):
                # if not isinstance(obs[a].childs['L'], float) or not isinstance(obs[a].childs['R'], float):
                if info['action_required'][a]:
                    # If an action is required, we want to store the obs a that step as well as the action
                    update_values = True

                    # distances = { key: child.dist_min_to_target for key, child in obs[a].childs.items() if not isinstance(child, float) }
                    # action_key = min(distances, key=distances.get)
                    # action = { 'L': 1, 'F': 2, 'R': 3 }[action_key]
                    # action = np.argmin(agent_obs[a])

                    # action = np.random.randint(4)
                    action = agent.act(agent_obs[a], eps=eps)
                    action_dict[a] = action
                    action_prob[action] += 1
                    steps_taken += 1
                else:
                    update_values = False
                    action_dict[a] = 2

            # Environment step
            obs, all_rewards, done, info = env.step(action_dict)

            # Update replay buffer and train agent
            for a in range(flags.num_agents):
                # Only update the values when we are done or when an action was taken and thus relevant information is present
                if update_values or done[a]:
                    agent.step(agent_obs_buffer[a], agent_action_buffer[a], all_rewards[a], agent_obs[a], done[a], flags.train)
                    agent_obs_buffer[a] = agent_obs[a].copy()
                    agent_action_buffer[a] = action_dict[a]
                if obs[a]:
                    agent_obs[a] = normalize_observation(obs[a], flags.tree_depth)

                score += all_rewards[a] / flags.num_agents

            # Render
            if flags.render_interval and episode % flags.render_interval == 0:
                render(env_renderer)
            if done['__all__']: break

        # Epsilon decay
        eps = max(0.01, flags.epsilon_decay * eps)

        # Save some training statistics in their respective deques
        tasks_finished = sum(done[i] for i in range(flags.num_agents))
        done_window.append(tasks_finished / max(1, flags.num_agents))
        scores_window.append(score / max_steps)
        steps_window.append(steps_taken)
        action_probs = ', '.join(f'{x:.3f}' for x in action_prob / np.sum(action_prob))

        print(f'\rTraining {flags.num_agents} Agents on ({flags.grid_width},{flags.grid_height}) \t ' +
              f'Episode {episode} \t ' +
              f'Average Score: {np.mean(scores_window):.3f} \t ' +
              f'Average Steps Taken: {np.mean(steps_window):.1f} \t ' +
              f'Dones: {100 * np.mean(done_window):.2f}% \t ' +
              f'Epsilon: {eps:.2f} \t ' +
              f'Action Probabilities: {action_probs}', end=" ")

        if episode % flags.report_interval == 0:
            print(f'\rTraining {flags.num_agents} Agents on ({flags.grid_width},{flags.grid_height}) \t ' +
                  f'Episode {episode} \t ' +
                  f'Average Score: {np.mean(scores_window):.3f} \t ' +
                  f'Average Steps Taken: {np.mean(steps_window):.1f} \t ' +
                  f'Dones: {100 * np.mean(done_window):.2f}% \t ' +
                  f'Epsilon: {eps:.2f} \t ' +
                  f'Action Probabilities: {action_probs} \t ' +
                  f'Time taken: {time.time() - start_time:.2f}s')

            if flags.train: agent.save(project_root / 'checkpoints', episode, eps)
            start_time = time.time()
            action_prob = [1] * action_size


if __name__ == '__main__':
    main()
