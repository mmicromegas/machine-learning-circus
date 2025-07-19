import gym
import numpy as np

# https://gymnasium.farama.org/environments/classic_control/cart_pole/
# observation space cart position, cart velocity, pole agnle, pole angular velocity

env = gym.make('CartPole-v1', render_mode="human")
np.random.seed(1)

high_reward = 0
best_weights = None

if  1 == 1:
    def run_episode(episode, state, weights, total_reward=0):
        is_done = False
        while not is_done:  # for j in range(1000):
            env.render()

            action = 0 if np.matmul(weights, state) < 0 else 1

            state, reward, is_done, truncated, info = env.step(action)  # is done, or terminated is when the pendulum is obviously falling down

            total_reward += reward

            print('episode {}, action {}, total_reward {}, reward {}, high_reward {}'.format(episode, action, total_reward, reward, high_reward))

            if is_done:
                break

        return total_reward


    n_episode = 30

    for episode in range(n_episode):
        state, info = env.reset()
        weights = np.random.uniform(-1, 1, 4)
        total_reward = run_episode(episode, state, weights)

        if total_reward > high_reward:
            high_reward = total_reward
            best_weights = weights

    # save BestWeights to file
    np.save("BestWeights_RL_basic.npy", best_weights)

# load BestWeights from file
best_weights = np.load("BestWeights_RL_basic.npy")

if 1 == 0:
    state, info = env.reset()

    for j in range(1000):
        env.render()
        action = 0 if np.matmul(best_weights, state) < 0 else 1
        state, reward, is_done, truncated, info = env.step(action)
        print(j, action, reward)
