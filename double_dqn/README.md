## Associated Blog posts:
1. [DQN](https://medium.com/@saasha/rl-series-2-dqn-e739eb3ab1d1)
2. [Exploration-Exploitation Dilema](https://medium.com/@saasha/rl-series-3-to-explore-or-not-to-explore-1ff88e4bf5af)

## Deep Q Networks

This folder contains the Pytorh implementation of the Deep Q Networks, as expressed in [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). The code here supports my write up on DQN and the Exploration-Exploitation Dilemma as listed above.

## Usage
To train the agent with default settings
```sh
python main.py --train
```

To test the agent
```sh
python main.py --results-folder <basepath_where_the_models_are_stored>
```

For further examples and/or to run the experiments that generated the graphs in [Exploration-Exploitation Dilemma](https://medium.com/@saasha/rl-series-3-to-explore-or-not-to-explore-1ff88e4bf5af), experiments.sh can be used by uncommenting the corresponding command.

## Requirements
The code here was developed using:
* Python 3.5
* [Pytorch](https://pytorch.org/get-started/locally/) 1.1.0
* Numpy 1.15
* Matplotlib 3.0.0
* [Gym](https://github.com/openai/gym) -- to run the LunarLander env, you would need to additionally run ```pip install gym[box2d]```

