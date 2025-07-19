# https://www.youtube.com/watch?v=g_8gw2POOYE

from UTILS.ReinforcementLearningSample import Agent
from UTILS.ReinforcementLearningSample import SampleEnvironment

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
    env = SampleEnvironment()
    agent = Agent()

    i = 0

    while not env.is_done():
        i = i + 1
        print(f"Step {i}")
        agent.step(env)

    print(f"Total reward got: {agent.total_reward}")


# call main
if __name__ == "__main__":
    main()