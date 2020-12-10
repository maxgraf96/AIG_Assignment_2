import numpy as np


# epsilon-greedy exploration strategy
def epsilon_greedy(features, epsilon1, n_actions):
    A = np.ones(n_actions, dtype=float) * epsilon1 / n_actions
    best_action = np.argmax(features)
    A[best_action] += (1.0 - epsilon1)
    action = np.random.choice(n_actions, p=A)
    return action


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        # TODO:
        q = features.dot(theta)
        done = False
        while not done:
            action = epsilon_greedy(q, epsilon[i], env.n_actions)  # Get action for current state
            features_, reward, done = env.step(action)  # Step into next environment
            delta = reward - features[action]

            q_ = features_.dot(theta)
            action_ = epsilon_greedy(q_, epsilon[i], env.n_actions)  # Find out next action from policy for state_
            delta = delta + gamma * np.argmax(features_[action_])

            theta = theta + eta[i] * delta * features[action]

            if done:
                break
            # update our state
            features = features_
    return theta
