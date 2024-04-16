# April 16, 2024 Yifan Xu Rice University

# this is some code to help me understand the Consensus problem
# and the finite-time Consensus property of some graphs. 

# INPUTS:

N = 50

# an undirected simple graph graph representing the connectivity between the agents
# assumed to be a connected graph
GRAPH_TOPOLOGY = [[]] 

# the initial values that the agents hold at t = 0.
INIT_VALUES = []
AVERAGE = sum(INIT_VALUES) / len(INIT_VALUES)

# functions to iterate

def assign_weights(incidence_matrix: list) -> list:
    '''
    Takes in an undirected graph ,
    and returns the same topology but with assigned weights.
    Returned matrix is doubly stochastic
    '''
    return []

def update_agents(agent_values: list) -> None:
    '''
    Modifies values of the agents according to their neighbors
    '''

# functions to calculate and plot error

def calculate_squared_error(agent_values: list) -> float:
    '''
    Takes in the agent values and calculates the squared error
    '''
    return 0.0

def plot_error(errors: list) -> None:
    '''
    Takes in squared error values and
    plots error vs. # of iterations
    '''

# functions to generate graphs



# initialize values and weights 
agent_values = INIT_VALUES
errors = []
