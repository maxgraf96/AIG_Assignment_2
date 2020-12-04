import numpy as np
def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))  # Q table initialisation

    def greedy_policy(state):
        action = 0
        if np.random.RandomState(0, 1) < epsilon:
            a = env.n_actions()
        else:
            a = np.argmax(q[state, :])
        return a

    '''def learn(s, next_s, reward, a):
        predict = q[s, a]
        target = reward + gamma * np.max(q[next_s, :])
        q[s, a] = q[s, a] + eta * (target - predict)'''

    for i in range(max_episodes):
        step = 0
        total_rewards = 0
        done = False  # Done will tell if we have reached the goal or not. It will be True once goal is reached

        s = env.reset()  # Resetting the environment

        while step < range:
            next_s, reward, done, info = env.step(a)  # stepping into the environment
            next_a = greedy_policy(next_s, q) # env.n_actions?

            learn = reward + gamma * q[next_s][next_a] - q[s, a]
            q[s, a] += eta * learn  # exploration/learn

            step += 1
            total_rewards += 1
            total_rewards += reward
            s, a = next_s, next_a  # defining next state as a current state

            if done == True:
                env.render()
                print("Episode {i} took {t} steps ")
                break


    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value
