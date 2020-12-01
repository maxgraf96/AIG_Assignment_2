import numpy as np


def average_reward_for_action_in_state(env, state, action):
    """
    Helper function that calculates the average reward for a given action (success + slips) at a given state
    :param env: The environment
    :param state: The state
    :param action: The action
    :return: The average reward
    """
    reward = 0.0
    for probability, next_state, rew, done in env.probabilities[state][action]:
        reward += rew

    return reward


def greedy_osla(env, state, value_function, gamma):
    action_values = np.zeros(env.n_actions)
    # One step lookahead
    for action in range(env.n_actions):
        average_reward = average_reward_for_action_in_state(env, state, action)
        for probability, next_state, reward, done in env.probabilities[state][action]:
            average_reward += gamma * probability * value_function[next_state]
        action_values[action] = average_reward

    return action_values


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    old_value = np.zeros(env.n_states, dtype=np.float)
    for i in range(max_iterations):
        delta = 0
        new_value = np.zeros(env.n_states, dtype=np.float)
        for state in range(env.n_states):
            action_values = greedy_osla(env, state, old_value, gamma)

            new_value[state] = action_values[policy[state]]
            delta = max(delta, np.fabs(new_value[state] - old_value[state]))

        if delta < theta:
            break

        old_value = new_value

    return new_value


def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)

    for state in range(env.n_states):
        action_values = greedy_osla(env, state, value, gamma)
        # Select best action greedily
        policy[state] = np.argmax(action_values)

    return policy


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.random.randint(0, env.n_actions, size=env.n_states)
    else:
        policy = np.array(policy, dtype=int)

    for i in range(max_iterations):
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)

        # Get new policy greedily
        new_policy = policy_improvement(env, value, gamma)

        # Break if converged
        if np.array_equal(policy, new_policy):
            break

        # Update old policy
        policy = new_policy

    return policy, value


def bellman(env, value, state, gamma):
    policy = np.zeros((env.n_states, env.n_actions))
    e = np.zeros(env.n_actions)

    # Find action giving maximum value
    for action in range(env.n_actions):
        q = 0
        P = np.array(env.probabilities[state][action])
        if np.size(P) == 0:
            rows = 0
        else:
            rows = np.shape(P)[0]

        for i in range(rows):
            successor = int(P[i][1])
            trans_proba = P[i][0]
            reward = P[i][2]
            q += trans_proba * (reward + gamma * value[successor])
        # Assign action value
        e[action] = q
    move = np.argmax(e)
    policy[state][move] = 1

    # Greedy take action and update value
    u = 0
    P = np.array(env.probabilities[state][move])
    if np.size(P) == 0:
        rows = 0
    else:
        rows = np.shape(P)[0]

    for i in range(rows):
        successor = int(P[i][1])
        trans_proba = P[i][0]
        reward = P[i][2]

        u += trans_proba * (reward + gamma * value[successor])

    value[state] = u

    return value[state]


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    # TODO:
    for iteration in range(max_iterations):
        delta = 0
        for state in range(env.n_states):
            val = value[state]
            bellman(env, value, state, gamma)
            delta = max(delta, abs(val - value[state]))
        if delta < theta:
            break

    policy = np.zeros(env.n_states, dtype=int)
    for state in range(env.n_states):
        policy = best_policy_for_state(env, value, policy, state, gamma)

    return policy, value


def best_policy_for_state(env, value, policy, state, gamma):
    e = np.zeros(env.n_actions)
    # Find action giving maximum value
    for action in range(env.n_actions):
        q = 0
        P = np.array(env.probabilities[state][action])
        if np.size(P) == 0:
            rows = 0
        else:
            rows = np.shape(P)[0]

        for i in range(rows):
            successor = int(P[i][1])
            trans_proba = P[i][0]
            reward = P[i][2]
            q += trans_proba * (reward + gamma * value[successor])
            # Assign action value
            e[action] = q

    # Get highest scoring move
    move = np.argmax(e)
    policy[state] = int(move)

    return policy
