# April 16, 2024 Yifan Xu Rice University

# this is some code to help me understand the Consensus problem
# and the finite-time Consensus property of some graphs. 

# imports

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# functions to iterate

def degree(agent: int, incidence_matrix: list) -> int:
    '''
    Takes in an agent and the topology,
    return the degree, defined as the number of neighbors, of the agent
    '''
    deg = 0
    for i in range(len(incidence_matrix[agent])):
        if incidence_matrix[agent][i] != 0 and i != agent: # excludes self
            deg += 1
    return deg

# test case
# topology = [
#     [0, 1, 1, 0],  
#     [1, 0, 1, 1],  
#     [1, 1, 0, 0],  
#     [0, 1, 0, 0]   
# ]

# Expected degrees [2, 3, 2, 1]
# print([degree(i, topology) for i in range(len(topology))])

def assign_weights(topology: list) -> list:
    '''
    Takes in a topology,
    and returns the same topology but with assigned weights.
    Returned matrix should be doubly stochastic
    '''
    # initialize incidence matrix to be zeroes
    graph = np.zeros([len(topology), len(topology)], dtype = float)
    
    for i in range(len(topology)):
        for j in range(len(topology[i])):
            if i != j and topology[i][j] != 0:
                graph[i][j] = 1 / (1 + max(degree(i, topology), degree(j, topology)))
    
    for i in range(len(topology)):
        graph[i][i] = 1 - sum(graph[i])
    
    return graph.tolist()

# # test case
# topology = [
#     [0, 1, 1, 0],  
#     [1, 0, 1, 1],  
#     [1, 1, 0, 0],  
#     [0, 1, 0, 0]   
# ]

# # Expected output
# [
#     [5/12, 1/4, 1/3, 0],
#     [1/4, 1/4, 1/4, 1/4],
#     [1/3, 1/4, 5/12, 0],
#     [0, 1/4, 0, 3/4]
# ]

# print(assign_weights(topology))


def update_agents(agent_values: list, incidence_matrix: list) -> list:
    '''
    Calculate and return the agents' values 
    according to their neighbors and their respective weights
    '''
    agent_values = np.transpose(agent_values)
    incidence_matrix = np.array(incidence_matrix)
    new_agent_values = np.matmul(incidence_matrix, agent_values)

    return list(np.transpose(new_agent_values))

# # test case
# initial_values = [10, 20, 30, 40]
# topology = [
#     [0, 1, 1, 0],
#     [1, 0, 1, 1],
#     [1, 1, 0, 0],
#     [0, 1, 0, 0]
# ]

# # expected output
# [115/6, 25, 125/6, 35]

# print(update_agents(initial_values, assign_weights(topology)))

def iterate_step(agent_values: list, incidence_matrix: list) -> tuple:
    '''
    modifies values of agents and, if applicable, the graph topology
    '''
    agent_values = update_agents(agent_values, incidence_matrix)
    # if need to modify topology, can add code below:
    return agent_values, incidence_matrix
# functions to calculate and plot error

def calculate_squared_error(agent_values: list, average: float) -> float:
    '''
    Takes in the agent values and the average; calculates the squared error
    '''
    error = 0
    for value in agent_values:
        error += (average - value)**2
    return error

# # quick test

# # expected output 500
# print(calculate_squared_error([10, 20, 30, 40], 25))

def exponential_func(x, a, b):
    return a * np.exp(b * x)

def plot_error(errors: list) -> float:
    '''
    Takes in squared error values and
    plots a scatter graph of squared error vs. # of iterations

    returns the exponential constant (between 0 and 1) 
    of the exponential line of best fit. 
    This should represent connectivity of the topology
    '''
    iterations = np.array([i+1 for i in range(len(errors))]) # iterations starts with 1

    # plotting the scatter of errors
    plt.scatter(iterations, errors, color='blue', label='Errors')

    # finding parameters of best fit exponential
    params, covariance = scipy.optimize.curve_fit(exponential_func, iterations, errors, p0=(1, 0.01))

    # generating data for the fitted curve
    fitted_values = exponential_func(iterations, *params)
    
    # plotting
    plt.plot(iterations, fitted_values, color='red', label=f'Best fit exp: y = {params[0]:.2f} * exp({params[1]:.2f} * x)')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Squared Error')
    plt.title('Squared Error vs. Number of Iterations')
    plt.legend()
    plt.show()

    # calculating lambda
    return np.exp(params[1])

# # test case
# errors_example = [500, 200, 120, 80, 30, 10, 5]
# print(plot_error(errors_example))

# functions to generate graphs



# functions to run main experiment

def run_experiment(num_of_iter: int, graph_topology: list, init_values: list) -> float:
    '''
    Inputs:
    - num_of_iter: the number of iterations to run
    - graph_topology: an incidence matrix represented by a list of lists
    that represents the topology between the agents. Assumed to be connected.
    - init_values: a list of floats representing the initial values
    held by the agents

    Runs the experiment and plots the squared error against number of iterations up to num_of_iter
    returns the exponential constant (between 0 and 1) 
    of the exponential line of best fit. 
    This should represent connectivity of the topology
    '''
    average = sum(init_values) / len(init_values)

    # initialize values and weights 
    agent_values = init_values
    incidence_matrix = assign_weights(graph_topology)
    errors = []

    # run the experiment
    for idx in range(num_of_iter):
        agent_values, incidence_matrix = iterate_step(agent_values, incidence_matrix)
        errors.append(calculate_squared_error(agent_values, average))

    # plot the error and return lambda
    return plot_error(errors)

# test cases
# topology = [
#     [0, 1, 1, 0],  
#     [1, 0, 1, 1],  
#     [1, 1, 0, 0],  
#     [0, 1, 0, 0]   
# ]
# print(run_experiment(50, topology, [10, 20, 30, 40]))