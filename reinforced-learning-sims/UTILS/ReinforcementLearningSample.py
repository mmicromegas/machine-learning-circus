import random
from typing import List


class SampleEnvironment:
    def __init__(self):
        self.steps_left = 20

    def get_observation(self) -> List[float]:
        # can be anything, some information regarding the environment
        return [0.0, 0.0, 0.0]  # this can be coordinates, three, four etc. information regarding the environment

    def get_actions(self) -> List[int]:
        # is nothing that, when agent performs actions, he/she should get some reward either 1 or 0
        # (considering 1 as positive reward, and 0 as negative reward)
        return [0, 1]

    def is_done(self) -> bool:
        # when steps are completed, gives indication for agent to move here and there
        return self.steps_left == 0

    def action(self, action: int) -> float:
        # returns reward
        if self.is_done():
            raise Exception("Game is over")
        self.steps_left -= 1
        return random.random()


class Agent:
    def __init__(self):
        self.total_reward = 0.0

    def step(self, env: SampleEnvironment):
        current_obs = env.get_observation()
        print("Observation {}".format(current_obs))
        actions = env.get_actions()
        print(actions)
        reward = env.action(random.choice(actions))
        self.total_reward += reward



