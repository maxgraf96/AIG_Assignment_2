import numpy as np


# epsilon-greedy exploration strategy
def epsilon_greedy(q, epsilon1, n_actions):
    A = np.ones(n_actions, dtype=float) * epsilon1 / n_actions
    best_action = np.argmax(q)
    A[best_action] += (1.0 - epsilon1)
    action = np.random.choice(n_actions, p=A)
    return action


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)

        # TODO:
        e = 0
        done = False
        action = epsilon_greedy(q, epsilon[i], env.n_actions)  # Get action for current state
        while not done:
            e = gamma * 0.9 * e + features[action]
            features_, reward, done = env.step(action)  # Step into next environment
            delta = reward - features[action]

            q_ = features_.dot(theta)  # Get Value for next state
            action_ = epsilon_greedy(q_, epsilon[i], env.n_actions)  # Find out next action from policy for state_
            delta = delta + gamma * features_[action_]

            theta = theta + eta[i] * delta * e

            if done:
                break
            # update our state
            features = features_
            action = action_
    return theta
