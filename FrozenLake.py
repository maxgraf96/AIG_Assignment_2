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

        # Precompute probabilities for transitions
        # self.probabilities = np.zeros((self.n_states, self.n_states, self.n_actions))
        self.probabilities = {state: {action: [] for action in range(n_actions)} for state in range(n_states)}

        def increment(row, col, action):
            """
            Helper function for generating transition probability matrix.
            Makes sure that our agents don't leave the playing field.
            :param row: Current row
            :param col: Current column
            :param action: Incoming action
            :return: New row and column coordinates
            """
            # Boundary checks to make sure that agents don't leave the field
            if action == 0:  # up
                row = max(row - 1, 0)
            elif action == 1:  # left
                col = max(col - 1, 0)
            elif action == 2:  # down
                row = min(row + 1, self.lake.shape[0] - 1)
            elif action == 3:
                col = min(col + 1, self.lake.shape[1] - 1)
            return row, col

        def update_prob_matrix(row, col, action):
            """
            Helper function for generating transition probability matrix.
            Takes a current position (row, col) and an action and returns
            the new state, the reward for the state and whether the agent is done (won or fell in hole)
            :param row:
            :param col:
            :param action:
            :return:
            """
            # Get next field coordinates with boundary checks
            new_row, new_col = increment(row, col, action)
            # Convert coordinates to state ID
            new_state = self.coords_to_state_idx[(new_row, new_col)]
            # Get new coordinate type
            f_type = self.lake[row][column]
            # Check whether we reached the goal or fell in hole
            done = f_type == '$' or f_type == '#'
            # Reward = 1 if agent is at goal, 0 otherwise
            reward = float(f_type == '$')
            return new_state, reward, done

        # Generate transition probabilities
        # Adapted from the openai gym implementation
        # https://gym.openai.com/envs/FrozenLake-v0/
        # Go through all fields and
        for row in range(self.lake.shape[0]):
            for column in range(self.lake.shape[1]):
                # Get state index from coordinates
                state_idx = self.coords_to_state_idx[(row, column)]
                # Go through all actions (up, left, down, right)
                for action in range(n_actions):
                    # Get transition probabilities for current state and action
                    current_list = self.probabilities[state_idx][action]
                    # Check if field is goal or hole
                    field_type = self.lake[row][column]
                    # If goal or hole, make inescapable (probability 1.0)
                    if field_type == '$':  # goal => reward 1, done = True
                        current_list.append((1.0, state_idx, 1.0, True))
                    elif field_type == '#':  # hole => reward 0, done = True
                        current_list.append((1.0, state_idx, 0.0, True))
                    # Otherwise check where we can go from here:
                    else:
                        # Add probabilities for successful action and slips
                        for b in range(n_actions):
                            if b == action:
                                # Successful action
                                # The asterisk ('*') unpacks the return value of the update_prob_matrix() function
                                # The probability for a successful action is 1 minus the slip probability
                                # for each remaining move
                                # Note: For this to work the slip probability has to be in the range [0...1]
                                current_list.append(
                                    (1.0 - (n_actions - 1.0) * self.slip, *update_prob_matrix(row, column, b))
                                )
                            else:
                                # Slip :(
                                current_list.append((self.slip, *update_prob_matrix(row, column, b)))

    def step(self, action):
        # This call updates self.step as well
        state, reward, done = Environment.step(self, action)

        # Check whether agent is done
        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):
        # Initialise transition probability
        transition_p = 0.0
        # Get possible moves for the state and action (successful move + slips)
        possible_moves = self.probabilities[state][action]
        for proba, next_s, reward, done in possible_moves:
            if next_state == next_s:
                transition_p += proba

        return transition_p

    def r(self, next_state_idx, state_idx, action):
        possible_actions = self.probabilities[state_idx][action]
        for proba, next_state, reward, done in possible_actions:
            if next_state == next_state_idx:
                return reward

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
