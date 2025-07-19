import random
from typing import List

import gym
import numpy as np


class Environment:
    def __init__(self):
        self.steps_left = None
        self.env = gym.make('CartPole-v1', render_mode="human")

    def get_observation(self) -> List[float]:
        state, info = self.env.reset()
        return state

    def is_done(self) -> bool:
        # when steps are completed, gives indication for agent to move here and there
        return self.steps_left == 0

    def get_actions(self, weights, state) -> int:
        # is nothing that, when agent performs actions, he/she should get some reward either 1 or 0
        # (considering 1 as positive reward, and 0 as negative reward)
        action = 0 if np.matmul(weights, state) < 0 else 1
        return action

    def render(self):
        self.env.render()

    def get_reward(self, action):
        state, reward, is_done, truncated, info = self.env.step(
            action)  # is done, or terminated is when the pendulum is obviously falling down
        return reward, state, is_done


class Agent:
    def __init__(self):
        pass

    def step(self, env: Environment, weights, episode, state, high_reward, total_reward=0):
        is_done = False

        while not is_done:
            env.render()

            action = env.get_actions(weights, state)

            reward, state, is_done = env.get_reward(action)

            total_reward += reward

            print('episode {}, action {}, total_reward {}, reward {}, high_reward {}'.format(episode, action,
                                                                                             total_reward, reward,
                                                                                             high_reward))
            if is_done:
                break

        return total_reward
