import numpy as np


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)

    # TODO:
    delta = 0
    while delta >= 0:
        delta = 0
        for i in range(env.n_states):
            v = value[i]
            result = 0


    return value


def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)

    # TODO:

    return policy


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    # TODO:
    for i in range(max_iterations):

        value_function = np.zeros(env.n_states)
        eps = 1e-10
        while True:
            prev_v = np.copy(value_function)
            for s in range(env.n_states):
                policy_a = policy[s]
                value_function[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
            if np.sum((np.fabs(prev_v - value_function))) <= eps:
                # value converged
                break

        # Get new policy from old one
        new_policy = np.zeros(env.n_states)
        for s in range(env.n_states):
            q_sa = np.zeros(env.n_actions)
            for action in range(env.n_actions):
                q_sa[action] = sum([p * (r + gamma * value_function[s_]) for p, s_, r, _ in env.P[s][action]])
            new_policy[s] = np.argmax(q_sa)
        return policy

    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    # TODO:

    return policy, value