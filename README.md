# maze-runner-a-reinforcement-learning-implementation
Value iteration model and Q learning model accomplish to help the maze runner get out of the maze!

# Value iteration
For Python: $ python value_iteration .py [ args ...]    
Where above [args...] is a placeholder for 6 command-line arguments: <maze input> <value file> <q value file>, <policy file>, <num epoch>, <discount factor>.    
These arguments are described in detail below:  
1. <maze input>: path to the environment input .txt described previously  
2. <value file>: path to output the values V (s)  
3. <q value file>: path to output the q-values Q(s; a)  
4. <policy file>: path to output the optimal actions (s)  
5. <num epoch>: the number of epochs your program should train the agent for. In one epoch,  
your program should update the value V (s) once for each s 2 S.  
6. <discount factor>: the discount factor   

As an example, if you implemented your program in Python, the following command line would
run your program with maze input le tiny maze.txt for 5 epochs and a discount factor of 0:9.
$ python value_iteration .py tiny_maze .txt value_output .txt \
q_value_output .txt policy_output .txt 5 0.9

## output
The <value file> should be formatted as \x y value" for each state s = (x; y) and its corresponding
value V (s). You should output one line for each state that is not occupied by an obstacle. The
order of the states in the output does not matter. Use \n to create a new line.  
The <q value file> should be formatted as \x y action q value" for each state s = (x; y) and
action a pair and the corresponding Q value Q(s; a). You should output one line for each state-
action pair where the state is not occupied by an obstacle. The order of the state-action pair of the
output does not matter. Use \n to create a new line. Please compute Q(s; a) using the values
V (s) after the specied number of epochs.  
The <policy file> should be formatted as \x y optimal action" for each state s = (x; y) and the
corresponding optimal policy (s). You should output one line for each state that is not occupied
by an obstacle. The order of the states in the output does not matter. If there is a draw in
the Q values, pick the action represented by the smallest number. For example, for state s, if
Q(s; 0) = Q(s; 2), then output action 0. Use \n to create a new line. Please compute (s) using
the values V (s) after the specied number of epochs.  
  
## Convergence rule for value iteration
the algorithm converges when the change of V (s) for all s is small (less than 0:001).

# Environment
 next state: the state of the agent after taking action a at state s
 reward: the reward received from taking action a at state s
 is terminal: an integer value indicating whether the agent reaches a terminal state after
taking action a at state s. The value is 1 if terminal state is reached and 0 otherwise.  
For Python: $ python environment .py [ args ...]  
1. <maze input>: path to the environment input .txt described in section 3.1  
2. <output file>: path to output feedback from the environment after the agent takes the
sequence of actions specied with the next argument.  
3. <action seq file>: path to the le containing a sequence of 0, 1, 2, 3s that indicates the
actions to take in order. This le has exactly one line. A white space separates any two
adjacent numbers. For example, if the le contains the line 0 3 2, the agent should rst go
left, then go down, and nally go right.  

## output
Your program should write one output le whose lename ends with .feedback. It should contain
the return values of the step method in order. Each line should be formatted as \next state x
next state y reward is terminal", where \next state x" and \next state y" are the x and y
coordinates of the next state returned by the method. Exactly one space separates any adjacent
values on the same line. Use \n to create a new line.

# Q learning
Args  
1. <maze input>: path to the environment input .txt described previously  
2. <value file>: path to output the values V (s)  
3. <q value file>: path to output the q-values Q(s; a)  
4. <policy file>: path to output the optimal actions (s)  
5. <num episodes>: the number of episodes your program should train the agent for. One
episode is a sequence of states, actions and rewards, which ends with terminal state or ends
when the maximum episode length has been reached.  
6. <max episode length>: the maximum of the length of an episode. When this is reached, we
terminate the current episode.  
7. <learning rate>: the learning rate  of the q learning algorithm  
8. <discount factor>: the discount factor   
9. <epsilon>: the value  for the epsilon-greedy strategy  
