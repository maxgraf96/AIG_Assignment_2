from FrozenLake import FrozenLake
from ModelBasedAlgorithms import policy_iteration, value_iteration
from TabularModelFree.Sarsa import sarsa
from TabularModelFree.QLearning import q_learning
from NonTabularModelFree.LinearWrapper import LinearWrapper
from NonTabularModelFree.LinearSarsa import linear_sarsa
from NonTabularModelFree.LinearQLearning import linear_q_learning


def main():
    seed = 0

    # Small lake
    small_lake = [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]

    big_lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '#', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '#', '#', '.', '.', '.', '#', '.'],
                ['.', '#', '.', '.', '#', '.', '#', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '$']]

    # Use this to switch between small and big lake
    lake = small_lake
    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)

    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

    # print('')

    # print('## Policy iteration')
    # policy, value, n_iterations = policy_iteration(env, gamma, theta, max_iterations)
    # env.render(policy, value)
    #
    # print('')
    #
    # print('## Value iteration')
    # policy, value, n_iterations = value_iteration(env, gamma, theta, max_iterations)
    # env.render(policy, value)

    print('')
    #
    # print('# Model-free algorithms')
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5
    #
    # print('')
    #

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    #
    # print('')
    #
    # print('## Q-learning')
    # policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    # env.render(policy, value)
    #
    # print('')
    #
    # linear_env = LinearWrapper(env)
    #
    # print('## Linear Sarsa')
    #
    # parameters = linear_sarsa(linear_env, max_episodes, eta,
    #                           gamma, epsilon, seed=seed)
    # policy, value = linear_env.decode_policy(parameters)
    # linear_env.render(policy, value)
    #
    # print('')
    #
    # print('## Linear Q-learning')
    #
    # parameters = linear_q_learning(linear_env, max_episodes, eta,
    #                                gamma, epsilon, seed=seed)
    # policy, value = linear_env.decode_policy(parameters)
    # linear_env.render(policy, value)


# Call main() when running Main.py script
if __name__ == "__main__":
    main()
