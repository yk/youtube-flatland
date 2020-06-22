# Tensorflow 2 Baseline

This baseline is a copy (with some minor changes) of the original `train_navigation.py`
script from the flatland/baselines repo.


## Requirements

Setting up an environment using conda is recommended (much easier to install tensorflow with GPU support):

```
conda create -n my_env_name python=3.6
```

Then install tensorflow using conda:

```
conda install tensorflow-gpu
```

Now, install other requirements as usual, e.g.:

```
pip install flatland-rl opencv-python numpy gym
```

## Running

Because of how relative imports work in Python 3, please run this from the root
folder, like so:

```
python ./approaches/tf2_baseline/train_baseline.py
```


## Caveats 

Need to rebuild and add clean `requirements.txt` - you may have to instal