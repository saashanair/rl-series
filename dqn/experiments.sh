# To run experiments described in RL Series#2: To explore or not to explore (https://medium.com/@saasha/rl-series-3-to-explore-or-not-to-explore-1ff88e4bf5af)
# Uncomment the experiment that you wish to execute

# no exploration
#python main.py --eps-max 0.0 --eps-min 0.0 --train

# epsilon-greedy, with epsilon at 0.01
#python main.py --eps-max 0.01 --eps-min 0.01 --train

# pure exploration
#python main.py --eps-max 1.0 --eps-min 1.0 --train

# epsilon-greedy with annealing at decay rate of 0.99
#python main.py --eps-max 1.0 --eps-min 0.01 --eps-decay 0.99 --train

# epsilon-greedy with annealing at decay rate of 0.9
#python main.py --eps-max 1.0 --eps-min 0.01 --eps-decay 0.9 --train

# epsilon-greedy with annealing at decay rate of 0.999
#python main.py --eps-max 1.0 --eps-min 0.01 --eps-decay 0.999 --train
