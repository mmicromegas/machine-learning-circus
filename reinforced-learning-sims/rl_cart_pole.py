# https://www.youtube.com/watch?v=g_8gw2POOYE

from UTILS.ReinforcementLearningCartPole import Agent
from UTILS.ReinforcementLearningCartPole import Environment

import numpy as np


# entities in RL world
# the Agent Class: piece of code implementing some policy
# the environment Class: model of the world that is external to the agent e.g. road, etc.

# many areas contributing: comp science, neuroscience, math
# RL == machine learning reward system

# 1. preparing agent
# 2. observation of the env
# 3. selection of optimal strategy
# 4. execution of the action
# 5. receiving reward/penalty
# repeat 2-5 until agent learn the optimal strategy

# agent monitors the env (input) and changes its status
# action changes the status of agent
# goal is to achieve highest possible reward

def main():
    env = Environment()
    agent = Agent()

    high_reward = 0
    best_weights = None

    n_episode = 30

    for episode in range(n_episode):
        state = env.get_observation()
        weights = np.random.uniform(-1, 1, 4)
        total_reward = agent.step(env, weights, episode, state, high_reward)

        if total_reward > high_reward:
            high_reward = total_reward
            best_weights = weights

    # save BestWeights to file
    np.save("BestWeights_RL_basic.npy", best_weights)


# call main
if __name__ == "__main__":
    main()
