from time import process_time
import numpy as np
from FrozenLake import FrozenLake
from ModelBasedAlgorithms import policy_iteration, value_iteration


def measure_policy_iteration(env, gamma, theta, max_iterations_policy_iteration, n_iterations_timing):
    """
    Runs policy iteration n_iterations_timing times and prints the mean execution time
    :param env: The game environment
    :param gamma: The discount factor
    :param theta: The threshold for early stopping
    :param max_iterations_policy_iteration: Maximum number of iterations for policy iteration
    :param n_iterations_timing: The number of times to run the timing
    :return:
    """
    print('## Policy iteration: Evaluation')
    # Store execution times of each run
    durations = []
    # How many iterations policy iteration took until reaching the threshold
    n_iterations_until_threshold = []

    for i in range(n_iterations_timing):
        # Measure execution time
        tic = process_time()
        policy, value, n_iterations = policy_iteration(env, gamma, theta, max_iterations_policy_iteration)

        n_iterations_until_threshold.append(n_iterations)

        toc = process_time()
        duration = toc - tic
        durations.append(duration)

    print()
    print("Mean elapsed time:", round(np.mean(durations), 4), "seconds")
    print("Mean number of iterations before reaching threshold", round(np.mean(n_iterations_until_threshold), 4))


def measure_value_iteration(env, gamma, theta, max_iterations_value_iteration, n_iterations_timing):
    """
    Runs policy iteration n_iterations_timing times and prints the mean execution time
    :param env: The game environment
    :param gamma: The discount factor
    :param theta: The threshold for early stopping
    :param max_iterations_value_iteration: Maximum number of iterations for value iteration
    :param n_iterations_timing: The number of times to run the timing
    :return:
    """
    print('## Value iteration: Evaluation')
    # Store execution times of each run
    durations = []
    # How many iterations value iteration took until reaching the threshold
    n_iterations_until_threshold = []

    for i in range(n_iterations_timing):
        # Measure execution time
        tic = process_time()
        policy, value, n_iterations = value_iteration(env, gamma, theta, max_iterations_value_iteration)
        n_iterations_until_threshold.append(n_iterations)

        toc = process_time()
        duration = toc - tic
        durations.append(duration)

    print()
    print("Mean elapsed time:", round(np.mean(durations), 4), "seconds")
    print("Mean number of iterations before reaching threshold", round(np.mean(n_iterations_until_threshold), 4))


def main():
    seed = 0

    # Small lake
    small_lake = [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]

    # Big lake
    big_lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
                  ['.', '.', '.', '.', '.', '.', '.', '.'],
                  ['.', '.', '.', '#', '.', '.', '.', '.'],
                  ['.', '.', '.', '.', '.', '#', '.', '.'],
                  ['.', '.', '.', '#', '.', '.', '.', '.'],
                  ['.', '#', '#', '.', '.', '.', '#', '.'],
                  ['.', '#', '.', '.', '#', '.', '#', '.'],
                  ['.', '.', '.', '#', '.', '.', '.', '$']]

    # Use big lake for evaluating performance of policy and value iteration
    lake = big_lake
    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)

    gamma = 0.9
    theta = 0.001
    max_iterations = 100

    # How often to run the simulation
    n_iterations_timing = 100

    # Measure policy iteration execution time
    # measure_policy_iteration(env, gamma, theta, max_iterations, n_iterations_timing)

    # Measure value iteration execution time
    measure_value_iteration(env, gamma, theta, max_iterations, n_iterations_timing)


# Call main() when running Main.py script
if __name__ == "__main__":
    main()
