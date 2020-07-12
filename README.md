# Let's solve flatland!
This is a community effort to solve the [2020 NeurIPS Flatland Challenge](https://www.aicrowd.com/challenges/neurips-2020-flatland-challenge).
The flatland environment is [here](https://gitlab.aicrowd.com/flatland/flatland) and its [Documentation](http://flatland.aicrowd.com/intro.html)
Watch the introductory video [here](https://youtu.be/cvkeWwDQr0A)


## How do I contribute?
Fork this repository and submit a pull request or contribute to any of the other community repositories.
Don't be shy :)

## Other Community Repositories
- [https://github.com/mitchellgoffpc/flatland-training](https://github.com/mitchellgoffpc/flatland-training)
- Look at forks of this and other repositories to find more.

## TODOs
- Write boilerplate to interact with the environment. It is already a gym environment, so unclear if this is necessary, maybe code to visualize a run.
- Write simple baseline algorithms for pure RL and pure hard-coded heuristics.
- Write some documentation here about how to evaluate an algorithm, maybe we define some joint format for results so we can compare.

## Libraries to consider
- [DeepMind ACME](https://github.com/deepmind/acme) has a lot of RL built-in.
- [RLlib](https://docs.ray.io/en/master/rllib.html)
- [SEED](https://github.com/google-research/seed_rl)

## Code Structure
- Different solution approaches go into subfolders in `src/approaches`. Feel free to contribute your own or work on someone else's (maybe coordinate via discord).
- For easiness, there is a `setup.py`, so we can `pip install -e .` and don't have to use relative paths in the approaches, but let's not put requirements globally. Instead, make a `README` or `requirements.txt` in your approach folder (or utility submodule) if you rely on specific things. Not mandatory, most people will figure out what's missing by themselves.
- Shared utilities go in `src/utils`. As with the requirements, let's not have a global `__init__.py` for the utils (but `__init__.py`s are completely fine in submodules).
- Any code that interacts directly with the environment (wrappers, monkeypatches, visualization, etc.) can go into submodules of `src/environment`.


## Links:
- Presentation on [Graph representation for flatland](https://docs.google.com/presentation/d/1ua0iOCnNYB2jFAeD8otJfj5LPl-6SM3vCQ1ckx--8WI/edit#slide=id.g6de2939b6d_3_69)
- [Flatland RLlib baslines](https://flatland.aicrowd.com/research/baselines.html)
