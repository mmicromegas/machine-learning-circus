import sys

import torch
import gym
import numpy as np

env = gym.make('CartPole-v1', render_mode="human")
# env.seed(1)

n_state = env.observation_space.shape[0]
n_action = env.action_space.n

high_reward = 0
best_weight_tensor = None

if 1 == 1:
    def run_episode(env, weight):
        is_done = False

        state = env.reset()

        # convert state to ndarray
        state = np.array(state[0])

        grads = []
        total_reward = 0

        while not is_done:
            env.render()

            state = torch.from_numpy(state).float()  # convert state to tensor

            z = torch.matmul(state, weight) # This line performs a matrix multiplication of the state and the weight.
            probs = torch.nn.Softmax()(z) # This line applies the softmax function to z to get a probability distribution over actions.

            action = int(torch.bernoulli(probs[1]).item()) # this line samples an action according to the probability distribution probs.

            # calculate the gradient of the log probability of the action and append it to the grads list
            d_softmax = torch.diag(probs) - probs.view(-1, 1) * probs
            d_log = d_softmax[action] / probs[action]
            grad = state.view(-1, 1) * d_log
            grads.append(grad)

            state, reward, is_done, truncated, info = env.step(action)

            total_reward += reward

            print('episode {}, action {}, total_reward {}, reward {}, high_reward {}'.format(episode, action, total_reward, reward, high_reward))

            if is_done:
                break

        return total_reward, grads


    n_episode = 100
    learning_rate = 0.001

    total_rewards = []

    weight_tensor = torch.rand(n_state, n_action)

    for episode in range(n_episode):
        total_reward, gradients = run_episode(env, weight_tensor)
        for i, gradient in enumerate(gradients):
            weight_tensor += learning_rate * gradient * (total_reward - i)

        if total_reward > high_reward:
            high_reward = total_reward
            best_weight_tensor = weight_tensor

    # save BestWeights to file
    np.save("BestWeights_RL_gradient.npy", best_weight_tensor)

if 1 == 0:

    # load BestWeights from file
    best_weights = np.load("BestWeights_RL_gradient.npy")

    # convert best_weights to tensor
    best_weights = torch.from_numpy(best_weights)

    state = env.reset()

    # convert state to ndarray
    state = np.array(state[0])

    for j in range(1000):
        env.render()

        state = torch.from_numpy(state).float()

        z = torch.matmul(state, best_weights)
        probs = torch.nn.Softmax()(z)

        action = int(torch.bernoulli(probs[1]).item())
        state, reward, is_done, truncated, info = env.step(action)

        print(j, action, reward)
