import numpy as np
import cv2
import gym

from utils.flatland_training.src import tree_observation
from utils.flatland_training.src.observation_utils import normalize_observation

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train


# Use a the malfunction generator to break agents from time to time
stochastic_data = MalfunctionParameters(
                    malfunction_rate=1/10000,  # Rate of malfunction occurence
                    min_duration=15,  # Minimal duration of malfunction
                    max_duration=50  # Max duration of malfunction
                    )


class FlatlandEnv(gym.Env):
    def __init__(self, n_cars=3 , n_acts=5, min_obs=-1, max_obs=1, 
            n_nodes=2, ob_radius=10, x_dim=36, y_dim=36, feats='all'):

        self.tree_obs = tree_observation.TreeObservation(n_nodes)
        self.n_cars = n_cars
        self.n_nodes = n_nodes
        self.ob_radius = ob_radius
        self.feats = feats

        rail_gen = sparse_rail_generator(
            max_num_cities=3,
            seed=666,
            grid_mode=False,
            max_rails_between_cities=2,
            max_rails_in_city=3
        )        

        self._rail_env = RailEnv(
            width=x_dim,
            height=y_dim,
            rail_generator=rail_gen,
            schedule_generator=sparse_schedule_generator(speed_ration_map),
            number_of_agents=n_cars,
            malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
            obs_builder_object=self.tree_obs)

        self.renderer = RenderTool(self._rail_env, gl="PILSVG")
        self.action_dict = dict()
        self.info = dict()
        self.old_obs = dict()

    def step(self, action):
        # Update the action of each agent
        for agent_id in range(self.n_cars):
            if action[agent_id] is None:
                action[agent_id] = 2
            self.action_dict.update({agent_id: action[agent_id]+1}) # FIXME: Hack for ignoring action 0 (model only outputs 4)

        # Take actions, get observations
        next_obs, all_rewards, done, self.info = self._rail_env.step(self.action_dict)

        # Normalise observations for each agent
        for agent_id in range(self._rail_env.get_num_agents()):

            # Check if agent is finished
            if not done[agent_id]:
                # Normalise next observation
                next_obs[agent_id] = normalize_observation(
                    tree=next_obs[agent_id], 
                    max_depth=self.n_nodes, 
                    observation_radius=self.ob_radius,
                    feats=self.feats
                )

                # Keep track of last observation for trains that finish    
                self.old_obs[agent_id] = next_obs[agent_id].copy()
            else:
                # Use last observation if agent finished 
                next_obs[agent_id] = self.old_obs[agent_id]

        return next_obs, all_rewards, done, self.info

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        return obs: initial observation of the space
        """
        self.action_dict = dict()
        self.info = dict()
        self.old_obs = dict()

        obs, self.info = self._rail_env.reset(True, True)
        for agent_id in range(self.n_cars):
            if obs[agent_id]:
                obs[agent_id] = normalize_observation(obs[agent_id], self.n_nodes, self.ob_radius, feats=self.feats)
        self.renderer.reset() 
        return obs, self.info

    def render(self, mode=None):
        self.renderer.render_env()
        image = self.renderer.get_image()
        cv2.imshow('Render', image)
        cv2.waitKey(20)
