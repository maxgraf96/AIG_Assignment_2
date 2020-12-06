import numpy as np
#from tqdm import trange # Processing Bar

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    # initialising the agent’s Q-table to zeros
    q = np.zeros((env.n_states, env.n_actions))

    # epsilon-greedy exploration strategy
    def epsilon_greedy(q, epsilon, n_actions, s):
        """
        Q: Q Table
        epsilon: exploration parameter
        n_actions: number of actions
        s: state
        """
        # selects a random action with probability epsilon
        if np.random.RandomState() <= epsilon:
            return np.random.randint(n_actions)
        else:
            return np.argmax(q[s, :])

    # SARSA Process
        """
        eta: learning rate
        gamma: exploration parameter
        max_episode: max number of episodes
        """


        reward_array = np.zeros(max_episodes)

        for i in range(max_episodes):

            s = env.reset()  # initial state/Resetting the environment

            a = epsilon_greedy(Q, epsilon, n_actions, s)  # initial action
            done = False
            while not done:
                s_, reward, done, _ = env.step(a)
                a_ = epsilon_greedy(q, epsilon, n_actions, s_)

                # update Q table
                q[s, a] += eta * (reward + (gamma * q[s_, a_]) - q[s, a])
                if done:
                    max_episodes.set_description('Episode {} Reward {}'.format(i + 1, reward))
                    max_episodes.refresh()
                    reward_array[i] = reward
                    break
                s, a = s_, a_
        env.close()
        return q, reward_array

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value
