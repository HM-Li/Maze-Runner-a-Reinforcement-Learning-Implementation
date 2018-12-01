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
        next_x, next_y = move_method(
            self.agent_state[0], self.agent_state[1], self.raw_data.shape[0], self.raw_data.shape[1])
        is_terminal = 0
        #if obstacle return
        if (self.raw_data[next_x][next_y] == "*"):
            next_x = self.agent_state[0]
            next_y = self.agent_state[1]
        if (self.raw_data[next_x][next_y] == "G"):
            #if goal
            is_terminal = 1
        reward = -1
        # update current state
        self.agent_state = (next_x, next_y)
        return (next_x, next_y), reward, is_terminal

    def reset(self):
        """This function resets the agent state to the initial state and returns the initial state.
        Input arguments: (Does not take any arguments)
        Return value: the initial state"""
        self.agent_state = copy.deepcopy(self.initial_state)
        return copy.deepcopy(self.initial_state)


# load data
# file_path = os.path.join(
#     "D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 8\handin\data")
# output_path = os.path.join(
#     "D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 8\handin\output")

# maze_input_path = os.path.join(file_path, "medium_maze.txt")
# output_file_path = os.path.join(output_path, "output.feedback")
# action_seq_file_path = os.path.join(file_path, "medium_maze_action_seq.txt")

maze_input_path = os.path.join('./', sys.argv[1])
output_file_path = os.path.join('./', sys.argv[2])
action_seq_file_path = os.path.join('./', sys.argv[3])


actions = [0, 1, 2, 3]
environment = Environment(actions)
environment.fit(maze_input_path)

action_seq = BasicPackage.read_actions(action_seq_file_path)

action_seq

with open(output_file_path, 'w') as f:
    for a in action_seq:
        next_state, reward, is_terminal = environment.step(float(a))
        f.write(" ".join(str(element) for element in [
                next_state[0], next_state[1], reward, is_terminal])+'\n')
