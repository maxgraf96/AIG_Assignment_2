import numpy as np
#from tqdm import trange # Processing Bar

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    reward_array = np.zeros(max_episodes)

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
            return np.random.RandomState(n_actions)
        else:
            return np.argmax(q[s, :])

#Qlearning/agent taking random actions
    for i in range(max_episodes):
        # reset environment
        s = env.reset()
        a = epsilon_greedy(q, epsilon, n_actions, s)

        # done flag
        done = False
        while not done:
            s_, reward, done, info = env.step(a)
            a_= epsilon_greedy(q, epsilon, n_actions, s_)
            a_max = np.argmax(q[s_])  # estimation policy

            q[s, a] += eta * (reward + (gamma * q[s_, a_]) - q[s, a])

            if done:
                # update processing bar
                max_episodes.set_description('Episode {} Reward {}'.format(i + 1, reward))
                max_episodes.refresh()
                reward_array[i] = reward
                break
            s, a = s_, a_
    env.close()


    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value