import numpy as np


# from tqdm import trange # Processing Bar

# epsilon-greedy exploration strategy
def epsilon_greedy(q1, epsilon1, n_actions, s):
    """
    Q: Q Table
    epsilon: exploration parameter
    n_actions: number of actions
    s: state
    """
    # selects a random action with probability epsilon
    if np.random.random() <= epsilon1:
        return np.random.randint(n_actions)
    else:
        return np.argmax(q1[s, :])


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    # initialising the agent’s Q-table to zeros
    q = np.zeros((env.n_states, env.n_actions))

    # SARSA Process
    """
    eta: learning rate
    gamma: exploration parameter
    max_episode: max number of episodes
    """

    reward_array = np.zeros(max_episodes)

    for i in range(max_episodes):
        s = env.reset()  # initial state/Resetting the environment
        a = epsilon_greedy(q, epsilon[i], env.n_actions, s)  # initial action
        done = False
        while not done:
            s_, reward, done = env.step(a)
            a_ = epsilon_greedy(q, epsilon[i], env.n_actions, s_)

            # update Q table
            q[s, a] += eta[i] * (reward + (gamma * q[s_, a_]) - q[s, a])
            s, a = s_, a_

    policy = q.argmax(axis=1)
    value = q.max(axis=1)
    return policy, value
