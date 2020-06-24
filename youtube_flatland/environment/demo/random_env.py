from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator


rail_generator = sparse_rail_generator(
    max_num_cities=5,
    grid_mode=False,
    max_rails_between_cities=4,
    max_rails_in_city=4,
    seed=0,
)

env = RailEnv(
    width=50,
    height=50,
    rail_generator=rail_generator,
    schedule_generator=sparse_schedule_generator(),
    number_of_agents=10,
)

env.reset()
