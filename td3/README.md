## Associated Blog posts:
1. [Twin Delayed DDPG (TD3): How does the TD3 algorithm work?](https://www.saashanair.com/td3-theory/)
2. Code
3. Additional Experiments

## Twin Delayed Deep Deterministic Policy Gradients (TD3)

This folder contains the Pytorh implementation for TD3, as expressed in [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477).

Two variants of the code have been studied here. The folder '[critic_one_head](https://github.com/saashanair/rl-series/tree/master/td3/critic_one_head)' contains the traditional actor-critic setting, such that two copies of the critic are made to create Q1 and Q2. In comparison, the folder '[critic_two_heads](https://github.com/saashanair/rl-series/tree/master/td3/critic_two_heads)' contains the critic model inspired by the repo released by the authors of the paper (found [here](https://github.com/sfujim/TD3)) which contains two models wrapped within the same critic class. 

## Usage
To train the agent with default settings, switch into the critic_one_head or critic_two_heads, based on whichever you are interested in (both show similar behavior), and run the following:
```sh
python main.py --train
```

To test the agent, within the critic_one_head or critic_two_heads folder run:
```sh
python main.py --results-folder <basepath_where_the_models_are_stored>
```

## Requirements
The code here was developed using:
* Python 3.5
* [Pytorch](https://pytorch.org/get-started/locally/) 1.1.0
* Numpy 1.15
* Matplotlib 3.0.0
* [Gym](https://github.com/openai/gym) -- to run the LunarLander env, you would need to additionally run ```pip install gym[box2d]```