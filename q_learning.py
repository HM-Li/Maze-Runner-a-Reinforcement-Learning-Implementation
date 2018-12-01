import numpy as np
import os
import copy
import sys


class BasicPackage:

    def read_maze(path):
        """read_file read a file from a path and return a list for each row

        Args:
            path (string): os
            splitter (string): splitter
        """
        with open(path, 'r') as f:
            lines = f.read().splitlines()
            contents = [list(line) for line in lines]
        return np.array(contents)

    def read_actions(path):
        """
        file has exactly one line"""
        with open(path, 'r') as f:
            lines = f.read().splitlines()
            contents = lines[0].split(" ")
        return np.array(contents)

    def split_X_y(dataset):
        """split_X_y split X and y from the dataset and do transformation

        Args:
            dataset (list): formatted input data

        Returns:
            list: [description]
        """

        y = [int(line[0]) for line in dataset]
        # remove the additional '' caused by an extral '\t',convert elements to int
        X = [dict([map(int, element.split(':')) for element in line[1:-1]])
             for line in dataset]
        return X, y

    def write_file(list, path):
        """write_file #  write to a file

        Args:
            list ([type]): [description]
            path ([type]): [description]
        """

        with open(path, 'w') as file:
            for line in list:
                file.write(" ".join(str(element) for element in line)+'\n')


class ActionStateReaction:
    """
    Stores data for an action at a single state
    an action can have multiple result states
    """

    def __init__(self):
        self.action_state_pairs = []

    def addActionStatePair(self, prob, next_state, reward, is_terminal):
        self.action_state_pairs.append([prob, next_state, reward, is_terminal])


class Environment:
    """it supports three methods. The environment also keeps track of the current state of the
        agent. Suppose the current state is s and the agent takes an action a. The environment returns
        the following three values when the agent takes an action a at state s:
         next state: the state of the agent after taking action a at state s
         reward: the reward received from taking action a at state s
         is terminal: an integer value indicating whether the agent reaches a terminal state after
        taking action a at state s. The value is 1 if terminal state is reached and 0 otherwise.
    """

    def __init__(self, actions):
        self.raw_data = None
        self.actions = actions
        self.states = None
        self.agent_state = None
        self.initial_state = None

    def move_west(self, x, y, num_rows, num_columns):
        if y == 0:
            return x, y
        return x, y - 1

    def move_north(self, x, y, num_rows, num_columns):
        if x == 0:
            return x, y
        return x - 1, y

    def move_east(self, x, y, num_rows, num_columns):
        if y == num_columns - 1:
            return x, y
        return x, y + 1

    def move_south(self, x, y, num_rows, num_columns):
        if x == num_rows - 1:
            return x, y
        return x + 1, y

    def parseInputToStates(self, raw_data):
        """
        parse input data to states
        data structure: each states have requried number of actions, each action has arbitrary number of resulting states
        this function only support one single result state for each action by assuming the maze is a square
        if obtacle then 0 
        [[action, action, action],[action]]
        """
        assert self.states is not None, "State matrix hasn't been initialized."
        # assuming the maze is a square
        for row in range(self.raw_data.shape[0]):
            for column in range(self.raw_data.shape[1]):
                # generate actions for each states
                # in this program, assume each action only has one result state (next_state)
                # check obstacle
                if (raw_data[row][column] == "*"):
                    self.states[row][column] = None
                    continue
                if (self.raw_data[row][column] == "S"):
                    #set initial state
                    self.agent_state = (row, column)
                    self.initial_state = (row, column)
                action_switcher = {
                    0: self.move_west,
                    1: self.move_north,
                    2: self.move_east,
                    3: self.move_south
                }
                # reinitiate state to a list that can contain state action pairs
                self.states[row][column] = list()
                for action in actions:
                    move_method = action_switcher.get(
                        action, lambda: "Invalid Action!")
                    next_x, next_y = move_method(
                        row, column, raw_data.shape[0], raw_data.shape[1])
                    is_terminal = False
                    #if obstacle return
                    if (self.raw_data[next_x][next_y] == "*"):
                        next_x = row
                        next_y = column
                    if (self.raw_data[next_x][next_y] == "G"):
                        #if goal
                        is_terminal = True
                    #initiate a action state relationship
                    single_action = ActionStateReaction()
                    prob = 1
                    next_state = (next_x, next_y)
                    reward = -1
                    single_action.addActionStatePair(
                        prob, next_state, reward, is_terminal)
                    # add the action state pair to the state
                    self.states[row][column].append(single_action)

    def fit(self, file_name):
        """constructor"""
        self.raw_data = BasicPackage.read_maze(file_name)
        # initiate states
        self.states = np.zeros(
            (self.raw_data.shape[0], self.raw_data.shape[1])).tolist()
        # parse maze + initialize agent state
        self.parseInputToStates(self.raw_data)

    def step(self, a):
        """
        This function takes in an action a, simulates a step, sets the current state to the next state,
        and returns next state, reward, is terminal.
        Input arguments: a
        Return value: next state, reward, is terminal"""
        action_switcher = {
            0: self.move_west,
            1: self.move_north,
            2: self.move_east,
            3: self.move_south
        }
        move_method = action_switcher.get(a, lambda: "Invalid Action!")
        # the moving method can only detect walls
        next_x, next_y = move_method(
            self.agent_state[0], self.agent_state[1], self.raw_data.shape[0], self.raw_data.shape[1])
        is_terminal = 0
        obstacle = None
        #if obstacle return
        if (self.raw_data[next_x][next_y] == "*"):
            obstacle = (next_x, next_y)
            next_x = self.agent_state[0]
            next_y = self.agent_state[1]
        if (self.raw_data[next_x][next_y] == "G"):
            #if goal
            is_terminal = 1
        reward = -1
        # update current state
        self.agent_state = (next_x, next_y)
        return (next_x, next_y), reward, is_terminal, obstacle

    def reset(self):
        """This function resets the agent state to the initial state and returns the initial state.
        Input arguments: (Does not take any arguments)
        Return value: the initial state"""
        self.agent_state = copy.deepcopy(self.initial_state)
        return copy.deepcopy(self.initial_state)


class RLQLearner:
    """ a model free, epsilon greedy reinforcement learner"""

    def __init__(self, environment, num_episodes, max_episode_length, learning_rate, discount_factor, epsilon):
        self.env = environment
        self.num_episodes = int(num_episodes)
        self.max_episode_length = int(max_episode_length)
        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.epsilon = float(epsilon)
        #storing the V matrix for all expected future discount reward; null if obtacle
        self.V = None
        #storing the Q matrix for all expected future discount rewards for all actions in each state
        self.Q = None
        #storing optimal policies for each state; null if obtable
        self.P = None

#     def get_epsilon_greedy_policy(self, current_state):
#         # epsilon is a float
#         ep = np.ones(len(self.env.actions)) * self.epsilon / len(self.env.actions)
#         best_action_index = np.argmax(self.Q[current_state[0]][current_state[1]])
#         ep[best_action_index] += (1.0 - self.epsilon)
#         return ep

    def transform(self):
        """
        learn based on the initial data"""
        episode_counter = {}
        for episode_index in range(self.num_episodes):
            # use asynchronized update
            current_state = self.env.reset()
            step_counter = 0
            for step in range(self.max_episode_length):
                step_counter += 1
                # generate action using epsilon greedy (can use the policy generator and use random.choice)
                action_generator = np.random.uniform(0, 1)
                best_action_index = np.argmax(
                    self.Q[current_state[0]][current_state[1]])
                a = best_action_index if action_generator < 1 - \
                    self.epsilon else np.random.randint(0, 4)
                next_state, reward, is_terminal, obstacle = self.env.step(a)
                # if obstacle detected then mark it
                if obstacle is not None:
                    self.Q[obstacle[0]][obstacle[1]] = None
                    self.V[obstacle[0]][obstacle[1]] = None
                    self.P[obstacle[0]][obstacle[1]] = None
                #update Q
                self.Q[current_state[0]][current_state[1]][a] = (1 - self.learning_rate)*self.Q[current_state[0]][current_state[1]][a]+self.learning_rate*(
                    reward + self.discount_factor*np.max(self.Q[next_state[0]][next_state[1]]))
                if is_terminal == 1:
                    break
                current_state = next_state
            # keep tracking number of steps per episode
            episode_counter[episode_index+1] = step_counter
            print(episode_index)
            # update V & P
            # assuming the maze is a square
            for row in range(self.env.raw_data.shape[0]):
                for column in range(self.env.raw_data.shape[1]):
                    # check obstacle
                    if (self.Q[row][column] is None):
                        continue
                    self.V[row][column] = np.max(self.Q[row][column])
                    self.P[row][column] = np.argmax(self.Q[row][column])
        return episode_counter


#         return states, states_index


    def fit(self):
        # initiate V
        self.V = np.zeros(
            (self.env.raw_data.shape[0], self.env.raw_data.shape[1])).tolist()
        # initiate Q
        self.Q = np.zeros(
            (self.env.raw_data.shape[0], self.env.raw_data.shape[1])).tolist()
        # initiate P
        self.P = np.zeros(
            (self.env.raw_data.shape[0], self.env.raw_data.shape[1])).tolist()
        for row in range(self.env.raw_data.shape[0]):
            for column in range(self.env.raw_data.shape[1]):
                self.Q[row][column] = np.zeros(len(self.env.actions))

    def get_value_list(self):
        # print out value list as forma
        value_list = []
        for row in range(self.env.raw_data.shape[0]):
            for column in range(self.env.raw_data.shape[1]):
                # ignore obstacles
                if self.V[row][column] is None:
                    continue
                value_list.append([row, column, self.V[row][column]])
        return value_list

    def get_q_value_list(self):
        # print out q value list as forma
        q_value_list = []
        for row in range(self.env.raw_data.shape[0]):
            for column in range(self.env.raw_data.shape[1]):
                # ignore obstacles
                if self.Q[row][column] is None:
                    continue
                for action in self.env.actions:
                    q_value_list.append(
                        [row, column, action, self.Q[row][column][action]])
        return q_value_list

    def get_policy_list(self):
        # print out q value list as forma
        policy_list = []
        for row in range(self.env.raw_data.shape[0]):
            for column in range(self.env.raw_data.shape[1]):
                # ignore obstacles
                if self.P[row][column] is None:
                    continue
                policy_list.append([row, column, self.P[row][column]])
        return policy_list


# load data
# file_path = os.path.join(
#     "D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 8\handin\data")
# output_path = os.path.join(
#     "D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 8\handin\output")

# maze_input_path = os.path.join(file_path, "tiny_maze.txt")
# value_file_path = os.path.join(output_path, "value_output.txt")
# q_value_file_path = os.path.join(output_path, "q_value_output.txt")
# policy_file_path = os.path.join(output_path, "policy_output.txt")
# num_episodes = "1000"
# max_episode_length = "20"
# learning_rate = "0.8"
# discount_factor = "0.9"
# epsilon = "0.05"

maze_input_path = os.path.join('./', sys.argv[1])
value_file_path = os.path.join('./', sys.argv[2])
q_value_file_path = os.path.join('./', sys.argv[3])
policy_file_path = os.path.join('./', sys.argv[4])
num_episodes = sys.argv[5]
max_episode_length = sys.argv[6]
learning_rate = sys.argv[7]
discount_factor = sys.argv[8]
epsilon = sys.argv[9]


actions = [0, 1, 2, 3]
environment = Environment(actions)
environment.fit(maze_input_path)

rl = RLQLearner(environment, num_episodes, max_episode_length,
                learning_rate, discount_factor, epsilon)

rl.fit()

episode_step_counter = rl.transform()

# episode_step_counter

value_list = rl.get_value_list()
q_value_list = rl.get_q_value_list()
policy_list = rl.get_policy_list()

BasicPackage.write_file(value_list, value_file_path)
BasicPackage.write_file(q_value_list, q_value_file_path)
BasicPackage.write_file(policy_list, policy_file_path)
