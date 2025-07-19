# inverted pendulum problem = cart pole system
# control controlled by force +1, -1 to keep the pole upright
# if the pole deviates by 15% from upright, the procedure ends

# https://gymnasium.farama.org/environments/classic_control/
# https://gymnasium.farama.org/content/basic_usage/

import gymnasium as gym
import gymnasium as gym
env = gym.make('CartPole-v1',render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    print(observation, reward, terminated, truncated, info)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

