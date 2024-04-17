# April 16, 2024 Yifan Xu Rice University

# this is some code to help me understand the Consensus problem
# and the finite-time Consensus property of some graphs. 

# imports

import numpy as np
import matplotlib as plt

# functions to iterate

def degree(agent: int, incidence_matrix: list) -> int:
    '''
    Takes in an agent and the topology,
    return the degree, defined as the number of neighbors, of the agent
    '''
    deg = 0
    for nb in incidence_matrix[agent]:
        if nb != 0 and nb != agent: # excludes self
            deg += 1
    return deg

def assign_weights(topology: list) -> list:
    '''
    Takes in a topology,
    and returns the same topology but with assigned weights.
    Returned matrix should be doubly stochastic
    '''
    # initialize incidence matrix to be zeroes
    graph = np.zeros([len(topology), len(topology)], dtype = int)
    
    for i in range(len(topology)):
        for j in range(len(topology[i])):
            if i != j and topology[i][j] != 0:
                graph[i][j] = 1 / (1 + max(degree(i, topology), degree(j, topology)))
    
    for i in range(len(topology)):
        graph[i][i] = 1 - sum(graph[i])
    
    return graph.tolist()

def update_agents(agent_values: list, incidence_matrix: list) -> list:
    '''
    Calculate and return the agents' values 
    according to their neighbors and their respective weights
    '''
    agent_values = np.transpose(agent_values)
    incidence_matrix = np.array(incidence_matrix)
    new_agent_values = np.matmul(incidence_matrix, agent_values)

    return list(np.transpose(new_agent_values))


def iterate_step(agent_values: list, incidence_matrix: list) -> None:
    '''
    modifies values of agents and, if applicable, the graph topology
    '''
    agent_values = update_agents(agent_values, incidence_matrix)
    # if need to modify topology, can add code below:

# functions to calculate and plot error

def calculate_squared_error(agent_values: list, average: float) -> float:
    '''
    Takes in the agent values and the average; calculates the squared error
    '''
    error = 0
    for value in agent_values:
        error += (average - value)**2
    return error

def plot_error(errors: list) -> None:
    '''
    Takes in squared error values and
    plots error vs. # of iterations
    '''
    iterations = np.array([i for i in range(len(errors))])

    plt.plot(iterations, np.arrary(errors))
    plt.show()

# functions to generate graphs



# functions to run main experiment

def run_experiment(num_of_iter: int, graph_topology: list, init_values: list) -> None:
    '''
    Inputs:
    - num_of_iter: the number of iterations to run
    - graph_topology: an incidence matrix represented by a list of lists
    that represents the topology between the agents. Assumed to be connected.
    - init_values: a list of floats representing the initial values
    held by the agents

    Runs the experiment and plots the squared error against number of iterations up to num_of_iter
    '''
    average = sum(init_values) / len(init_values)

    # initialize values and weights 
    agent_values = init_values
    incidence_matrix = assign_weights(graph_topology)
    errors = []

    # run the experiment
    for idx in range(num_of_iter):
        iterate_step(agent_values, incidence_matrix)
        errors.append(calculate_squared_error(agent_values, average))

    # plot the error
    plot_error(errors)
