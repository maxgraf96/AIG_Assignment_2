import numpy as np


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    """
    Policy iteration algorithm. Takes a game environment, discount factor threshold value and
    number of maximum iterations to iteratively generate the best policy for a given environment.
    :param env: The game environment
    :param gamma: The discount factor
    :param theta: The threshold for ending the calculations
    :param max_iterations: The maximum number of iterations
    :param policy: The policy (optimal in the end)
    :return: The optimal policy, and its action values for each state
    """
    if policy is None:
        policy = np.random.randint(0, env.n_actions, size=env.n_states)
    else:
        policy = np.array(policy, dtype=int)

    # Store number of iterations until threshold reached for evaluation
    n_iterations = max_iterations

    for iteration in range(max_iterations):
        # Evaluate the current policy
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)

        # Get a new policy greedily
        new_policy = policy_improvement(env, value, gamma)

        # Break if converged
        if np.array_equal(policy, new_policy):
            # print("Policy iteration finished after " + str(iteration + 1) + " iterations.")
            n_iterations = iteration + 1  # +1 since iteration starts at 0
            break

        # Update old policy
        policy = new_policy

    return policy, value, n_iterations


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    # Store number of iterations until threshold reached for evaluation
    n_iterations = max_iterations

    # Find optimal value function
    for iteration in range(max_iterations):
        delta = 0
        for state in range(env.n_states):
            # Save value before applying Bellman operator
            old_val = value[state]
            # This call updates the value[state] we just saved
            bellman(env, value, state, gamma)
            # Get difference between old and new value
            delta = max(delta, abs(old_val - value[state]))
        # Break if we're below the tolerance level
        if delta < theta:
            # print("Value iteration finished after " + str(iteration + 1) + " iterations.")
            n_iterations = iteration + 1  # +1 since iteration starts at 0
            break

    # Policy extraction
    # Initialise policy
    policy = np.zeros(env.n_states, dtype=int)
    # And get the best policy (i.e. the best move for each state)
    # with the optimised value vector calculated above
    for state in range(env.n_states):
        # This call updates the policy state-by-state
        policy = best_policy_for_state(env, value, policy, state, gamma)

    return policy, value, n_iterations


# ------------------------- Helper functions for policy iteration -------------------------
def policy_evaluation(env, policy, gamma, theta, max_iterations):
    """
    Improve the value function iteratively
    :param env: The game environment
    :param policy: The current policy
    :param gamma: The discount factor
    :param theta: The threshold below which to break the calculations
    :param max_iterations: Maximum number of iterations
    :return: The best value function after max_iterations
    """
    # Initialise placeholder policy
    old_value = np.zeros(env.n_states, dtype=np.float)
    for i in range(max_iterations):
        # Stores the difference between most recent value function
        # and value function from one iteration before
        delta = 0
        # Initialise new value function which will be set greedily
        new_value = np.zeros(env.n_states, dtype=np.float)
        # Go through all states
        for state in range(env.n_states):
            # Get the value the current state and all possible actions (4 in this case)
            action_values = greedy_osla(env, state, old_value, gamma)

            # Set value function value for current state to the value of
            # the move currently selected in the policy
            new_value[state] = action_values[policy[state]]
            # Calculate difference between new and old value
            delta = max(delta, np.fabs(new_value[state] - old_value[state]))

        # Break if value difference smaller than threshold
        if delta < theta:
            # print("Policy evaluation finished after " + str(i + 1) + " iterations.")
            break

        # Update policy
        old_value = new_value

    return new_value


def policy_improvement(env, value, gamma):
    """
    Greedily improve policy using an existing value function
    :param env: The game environment
    :param value: The value function
    :param gamma: The discount factor
    :return: The improved policy
    """
    # Initialise policy
    policy = np.zeros(env.n_states, dtype=int)

    for state in range(env.n_states):
        # Get average rewards for each of the actions (up, down, left, right)
        action_values = greedy_osla(env, state, value, gamma)
        # Select best action
        policy[state] = np.argmax(action_values)

    return policy


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
    """
    Helper function: Greedy one-step lookahead to select the "best" possible move in a given state
    :param env: The game environment
    :param state: The current game state
    :param value_function: The current value function
    :param gamma: The discount factor
    :return: Array of average rewards for each action
    """
    # Initialise array of average rewards for each action
    action_values = np.zeros(env.n_actions)
    # One step lookahead: For each action (in our case up, down, left, right)
    for action in range(env.n_actions):
        # Get average reward for action in current state
        average_reward = average_reward_for_action_in_state(env, state, action)
        # Get transition probability from matrix and use it to calculate the average reward
        # for taking this action
        for probability, next_state, reward, done in env.probabilities[state][action]:
            average_reward += gamma * probability * value_function[next_state]
        action_values[action] = average_reward

    return action_values


# ------------------------- Helper functions for value iteration -------------------------
def bellman(env, value, state, gamma):
    """
    Apply the optimality Bellman operator to the current value function with the current state
    Note: This method modifies the "value" vector passed into it
    :param env: The game environment
    :param value: Vector of the action values for each state
    :param state: Index of the current state
    :param gamma: Discount factor
    :return: Action value after Bellman operation is applied
    """
    # Initialise policy and actions
    policy = np.zeros((env.n_states, env.n_actions))
    action_values = np.zeros(env.n_actions)

    # Find the action giving the maximum value
    for action in range(env.n_actions):
        action_value = 0
        # Get state-action data
        # Structure of one entry: (transition probability, next state index, reward, finished)
        state_action_p = env.probabilities[state][action]
        # Edge-case: Final state
        if np.size(state_action_p) == 0:
            n_transitions = 0
        else:
            n_transitions = np.shape(state_action_p)[0]

        # Go through all next possible next states and save their action values
        for i in range(n_transitions):
            successor_state = int(state_action_p[i][1])
            trans_proba = state_action_p[i][0]
            reward = state_action_p[i][2]
            # Calculate action value
            action_value += trans_proba * (reward + gamma * value[successor_state])
        # Assign action value
        action_values[action] = action_value
    # Get index highest action value
    move = np.argmax(action_values)
    # Take best move (all other moves for the state stay 0)
    policy[state][move] = 1

    # Greedy take action and update value
    # Reset action value and set state-action data to current state, and move taken above
    action_value = 0
    state_action_p = env.probabilities[state][move]
    # Again, guard for edge-case: Final state
    if np.size(state_action_p) == 0:
        n_transitions = 0
    else:
        n_transitions = np.shape(state_action_p)[0]

    # Accumulate action values for all possible transitions
    # for the current state with the best move
    for i in range(n_transitions):
        # Get successor state
        successor_state = int(state_action_p[i][1])
        # Get transition probability
        trans_proba = state_action_p[i][0]
        # Get reward
        reward = state_action_p[i][2]
        # Calculate action value
        action_value += trans_proba * (reward + gamma * value[successor_state])

    # Update in value function passed into this function
    value[state] = action_value

    return value[state]


def best_policy_for_state(env, value, policy, state, gamma):
    """
    Helper function for value iteration that selects the best policy for a given state
    :param env: The game environment
    :param value: The value function
    :param policy: The current policy
    :param state: The current game state
    :param gamma: The discount factor
    :return: The best policy for a given state
    """
    # Initialise action values
    action_values = np.zeros(env.n_actions)
    # Find action giving maximum value
    for action in range(env.n_actions):
        action_value = 0
        # Get state-action data for state and current action
        state_action_p = env.probabilities[state][action]
        # Edge-case: Final state
        if np.size(state_action_p) == 0:
            n_transitions = 0
        else:
            n_transitions = np.shape(state_action_p)[0]

        # For all possible transitions:
        for i in range(n_transitions):
            # Get successor state
            successor_state = int(state_action_p[i][1])
            # Get transition probability
            trans_proba = state_action_p[i][0]
            # Get reward
            reward = state_action_p[i][2]
            # Calculate action value
            action_value += trans_proba * (reward + gamma * value[successor_state])
            # Assign action value
            action_values[action] = action_value

    # Get highest scoring move
    move = np.argmax(action_values)
    # Set "best" move for state in policy
    policy[state] = int(move)

    return policy
