"""
Script that contains the details about the OU Noise generating process.

Based on:
1. https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab (Formula to be implemented -- Matlab)
2. https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py (Implementation used in OpenAI Baselines)
"""

import numpy as np

class OrnsteinUhlenbeckNoise():
    """
    Class that OU Process
    """
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x_start=None):
        self.mu = mu # mean value around which the random values are generated
        self.sigma = sigma # amount of noise to be applied to the process
        self.theta = theta # amount of frictional force to be applied
        self.dt = dt
        self.x_start = x_start # the point from where the random walk is started

        self.reset()

    def reset(self):
        """
        Function to revert the OU process back to default settings. If x_start is specified, use it, else, start from zero.

        Parameters
        ---
        none

        Returns
        ---
        none
        """
        self.prev_x = x_start if self.x_start is not None else np.zeros_like(self.mu)

    def generate_noise(self):
        """
        Function to generate the next value in the random walk which is then used a noise added to the action during training to encourage exploration.
        Formula:
            X_next = X_prev + theta * (mu - X_prev) * dt + sigma * sqrt(dt) * n, where 'n' is a random number sampled from a normal distribution with mean 0 and standard deviation 1

        Parameters
        ---
        none

        Returns
        ---
        none
        """
        x = self.prev_x + self.theta * (self.mu - self.prev_x) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(loc=0.0, scale=1.0, size=self.mu.shape)

        self.prev_x = x
        return x

