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


class RLValueIterationLearner:
    def __init__(self, raw_data, num_epoch, discount_factor, actions):
        self.raw_data = raw_data
        self.num_epoch = float(num_epoch)
        self.discount_factor = float(discount_factor)
        self.actions = actions
        self.states = None
        #storing the V matrix for all expected future discount reward; null if obtacle
        self.V = None
        #storing the Q matrix for all expected future discount rewards for all actions in each state
        self.Q = None
        #storing optimal policies for each state; null if obtable
        self.P = None

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
                        row, column, self.raw_data.shape[0], self.raw_data.shape[1])
                    is_terminal = False
                    #if obstacle return
                    if (raw_data[next_x][next_y] == "*"):
                        next_x = row
                        next_y = column
                    if (raw_data[next_x][next_y] == "G"):
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

    def one_state_move(self, row, column):
        """
        do Q calculation for one state based on all actions"""
        q = np.zeros(len(self.actions))
        for action in self.actions:
            # add all posible results due to this action to Q
            temp = 0
            for prob, next_state, reward, is_terminal in self.states[row][column][action].action_state_pairs:
                temp += prob * self.V[next_state[0]][next_state[1]]
            # add reward; assume same reward for all posible result state
            q[action] = reward + self.discount_factor * temp
        return q

    def transform(self):
        """
        learn based on the initial data"""
        epoch_counter = 0

        while epoch_counter < self.num_epoch:
            # tempory v for synchronized update
            v = np.zeros(
                (self.raw_data.shape[0], self.raw_data.shape[1])).tolist()
            # assuming the maze is a square
            for row in range(self.raw_data.shape[0]):
                for column in range(self.raw_data.shape[1]):
                    # check obstacle
                    if (raw_data[row][column] == "*"):
                        v[row][column] = None
                        self.V[row][column] = None
                        self.Q[row][column] = None
                        self.P[row][column] = None
                        continue
                    # check goal; if true remain the V with the original 0
                    if (raw_data[row][column] == "G"):
                        continue
                    # Do a one-step lookahead to find the best action
                    q = self.one_state_move(row, column)
                    q_max = np.max(q)
                    v[row][column] = q_max

            # check convergence; if absolute changes of all states < 0.001 then converge
            difference = np.abs(
                np.array(self.V, dtype=np.float) - np.array(v, dtype=np.float))
            difference[np.isnan(difference)] = 0
            if (difference < 0.001).all():
                break

            # update V
            self.V = v
            epoch_counter += 1
            print(epoch_counter)
            # update Q & P
            # assuming the maze is a square
            for row in range(self.raw_data.shape[0]):
                for column in range(self.raw_data.shape[1]):
                    # check obstacle
                    if (self.raw_data[row][column] == "*"):
                        continue
                    # if goal, make every q to zero
                    if (self.raw_data[row][column] == "G"):
                        self.Q[row][column] = np.zeros(len(self.actions))
                        continue
                    # Do a one-step lookahead to find the best action
                    q = self.one_state_move(row, column)
                    self.Q[row][column] = q
                    self.P[row][column] = np.argmax(q)
        return epoch_counter


#         return states, states_index


    def fit(self):
        # initiate states
        self.states = np.zeros(
            (self.raw_data.shape[0], self.raw_data.shape[1])).tolist()
        # initiate V
        self.V = np.zeros(
            (self.raw_data.shape[0], self.raw_data.shape[1])).tolist()
        # initiate Q
        self.Q = np.zeros(
            (self.raw_data.shape[0], self.raw_data.shape[1])).tolist()
        # initiate P
        self.P = np.zeros(
            (self.raw_data.shape[0], self.raw_data.shape[1])).tolist()
        # parse maze
        self.parseInputToStates(self.raw_data)

    def get_value_list(self):
        # print out value list as forma
        value_list = []
        for row in range(self.raw_data.shape[0]):
            for column in range(self.raw_data.shape[1]):
                # ignore obstacles
                if self.V[row][column] is None:
                    continue
                value_list.append([row, column, self.V[row][column]])
        return value_list

    def get_q_value_list(self):
        # print out q value list as forma
        q_value_list = []
        for row in range(self.raw_data.shape[0]):
            for column in range(self.raw_data.shape[1]):
                # ignore obstacles
                if self.Q[row][column] is None:
                    continue
                for action in self.actions:
                    q_value_list.append(
                        [row, column, action, self.Q[row][column][action]])
        return q_value_list

    def get_policy_list(self):
        # print out q value list as forma
        policy_list = []
        for row in range(self.raw_data.shape[0]):
            for column in range(self.raw_data.shape[1]):
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
# num_epoch = "5"
# discount_factor = "0.9"

maze_input_path = os.path.join('./', sys.argv[1])
value_file_path = os.path.join('./', sys.argv[2])
q_value_file_path = os.path.join('./', sys.argv[3])
policy_file_path = os.path.join('./', sys.argv[4])
num_epoch = sys.argv[5]
discount_factor = sys.argv[6]


raw_data = BasicPackage.read_maze(maze_input_path)

actions = [0, 1, 2, 3]
rl = RLValueIterationLearner(raw_data, num_epoch, discount_factor, actions)

rl.fit()

rl.transform()

value_list = rl.get_value_list()
q_value_list = rl.get_q_value_list()
policy_list = rl.get_policy_list()

BasicPackage.write_file(value_list, value_file_path)
BasicPackage.write_file(q_value_list, q_value_file_path)
BasicPackage.write_file(policy_list, policy_file_path)
