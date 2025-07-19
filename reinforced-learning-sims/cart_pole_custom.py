# inverted pendulum problem = cart pole system
# control controlled by force +1, -1 to cart to keep the pole upright
# if the pole deviates by 15% from upright, the procedure ends

# https://gymnasium.farama.org/environments/classic_control/
# https://gymnasium.farama.org/content/basic_usage/

# inverted pendulum simulation
# https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling


import gymnasium as gym
import numpy as np
np.random.seed(1)

HighReward = 0
BestWeights = None

env = gym.make('CartPole-v1', render_mode="human")

if 1 == 1:
    for i in range(300):
        observation, info = env.reset()
        Weights = np.random.uniform(-1,1,4)
        SumReward = 0
        for j in range(300):
            #env.render()
            action = 0 if np.matmul(Weights, observation) < 0 else 1 # choose action for given weights and observation (results from previous action))
            observation, reward, terminated, truncated, info = env.step(action) # do action, and observation changes (i.e. cart position, cart velocity, pole angle, pole velocity at tip)
            SumReward += reward # collect reward
            print('i {}, j {}, action {}, SumReward {}, reward {}, highreward {}'.format(i,j, action , SumReward, reward, HighReward))
        if SumReward > HighReward:
            HighReward = SumReward
            BestWeights = Weights


    # save BestWeights to file
    np.save("BestWeights.npy", BestWeights)

# load BestWeights from file
BestWeights = np.load("BestWeights.npy")

observation, info = env.reset()

for j in range(1000):
    env.render()
    action = 0 if np.matmul(BestWeights, observation) < 0 else 1
    observation, reward, terminated, truncated, info = env.step(action)
    print(j,action, reward)



# action: push the cart to the left (0) and push the cart to the right (1)

# observation: An environment-specific object representing your observation of the environment. Contains 4 parameters: cart position, cart velocity, pole angle, pole velocity at tip

# reward: Amount of reward achieved by the previous action. The scale varies between environments, but the goal is always to increase your total reward.

# terminated: Whether it’s time to reset the environment again. Most tasks are divided up into episodes, and this flag is set when the episode is finished.

# info: Diagnostic information useful for debugging. It can sometimes be useful for learning



env.close()

