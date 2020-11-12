import random
import numpy as np
from Environment import Environment, play
from Util import _printoptions
from itertools import product


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
         lake =  [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """

        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip

        n_states = self.lake.size + 1
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        self.absorbing_state = n_states - 1

        # TODO:
        # Call parent constructor
        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed)

        # Up, left, down, right (corresponding to w, a, s, d)
        self.actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        # Matrix containing rewards for TAKING AN ACTION at a state
        self.reward_map = np.zeros(self.lake.shape, dtype=np.float)
        # Set goal state to 1
        self.reward_map[np.where(self.lake == '$')] = 1

        # Matrix indicating where the absorbing states are (holes & goal states are 1, others are 0)
        self.abs_states = np.zeros(self.lake.shape, dtype=np.float)
        # Set goal state to 1
        self.abs_states[np.where(self.lake == '$')] = 1
        self.abs_states[np.where(self.lake == '#')] = 1

        # Helpers for conversions from indices to states (coordinates) and states (coordinates) to indices
        self.state_idx_to_coords = list(product(range(self.reward_map.shape[0]), range(self.reward_map.shape[1])))
        self.coords_to_state_idx = {s: i for (i, s) in enumerate(self.state_idx_to_coords)}

        # Precompute allowed transitions
        self.allowed_transitions = np.zeros((self.n_states, self.n_states, self.n_actions))
        for state_index, state in enumerate(self.state_idx_to_coords):
            for action_index, action in enumerate(self.actions):
                # Get next state for action
                next_state = (state[0] + action[0], state[1] + action[1])

                # If next_state is not valid, default to current state index
                next_state_index = self.coords_to_state_idx.get(next_state, state_index)

                self.allowed_transitions[next_state_index, state_index, action_index] = 1.0


    def step(self, action):
        # Slip: With a chance of "self.slip" (set to 0.1) the agent slips and takes a random direction
        # Note: random.random() returns a random float between 0.0 and 1.0
        if random.random() < self.slip:
            print("Ooops, slipped...")
            # Assign random action
            action = np.random.randint(0, 3)

        # This call updates self.step as well
        state, reward, done = Environment.step(self, action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):
        # For normal, non-absorbing states:
        # Check the valid moves in the pre-calculated table
        tp = self.allowed_transitions[next_state, state, action]

        # For absorbing states:
        # Convert current state idx to coordinates
        current_coords = self.state_idx_to_coords[state]
        # Check whether current state is absorbing state
        if self.abs_states[current_coords] == 1:
            # If in absorbing state, only the current (absorbing) state is permitted
            if next_state == state:
                tp = 1.0
            else:
                tp = 0.0

        return tp

    def r(self, next_state_idx, state_idx, action):
        # Convert current state idx to coordinates
        current_coords = self.state_idx_to_coords[state_idx]

        # Check whether current state is absorbing state
        if self.abs_states[current_coords] == 1:
            # If in absorbing state, actions give 1 for goal
            if self.reward_map[current_coords] == 1:
                # In goal state :)
                return 1.0

        return 0.0

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))


if __name__ == "__main__":
    seed = 0

    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    play(env)
