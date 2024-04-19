# April 16, 2024 Yifan Xu Rice University

# this is some code to help me understand the Consensus problem
# and the finite-time Consensus property of some graphs. 

# imports

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import math
import networkx as nx

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
    agent_values = np.transpose(np.array(agent_values))
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

def iterate_step(agent_values: list, incidence_matrix: list, iteration: int, one_peer_exp_flag = False, p_peer_flag = False) -> tuple:
    '''
    modifies values of agents and, if applicable, the graph topology
    '''
    agent_values = update_agents(agent_values, incidence_matrix)

    # if need to modify topology, can add code below:

    if one_peer_exp_flag:
        # for one-peer exponential:
        incidence_matrix = one_peer_exponential(len(incidence_matrix), iteration)
    elif p_peer_flag:
        # for p-peer hypercuboid:
        incidence_matrix = p_peer_hypercuboids(len(incidence_matrix), iteration)

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
    params = scipy.optimize.curve_fit(exponential_func, iterations, errors, p0=(1, 0.01))[0]

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

def p_base_rep(i: int, p: int, tau: int) -> list:
    '''
    given an integer i in base 10 representation,
    returns i in base p representation in the form of a tuple:
    [i_{tau-1}, i_{tau-2}, ..., i_0]
    '''
    p_base = []
    while i > 0:
        p_base.append(i % p)
        i = i // p
    
    p_base.extend([0]*(tau - len(p_base)))
    p_base.reverse()

    return p_base

# print(p_base_rep(236, 5))

def de_bruijn(p: int, tau: int) -> list:
    '''
    Given integers p and tau, where the number of agents is p**tau,
    returns the de bruijn graph with its appropriate weights
    '''
    n = p**tau
    graph = [[0] * n for i in range(n)]
    p_base_nums = []
    for num in range(n):
        p_base_nums.append(p_base_rep(num, p, tau))
    
    for i in range(n):
        for j in range(n):
            if tuple(p_base_nums[i][1:]) == tuple(p_base_nums[j][:-1]):
                graph[i][j] = 1/p
    return graph

# print(de_bruijn(2, 3))

def one_peer_exponential(n: int, iteration: int) -> list:
    '''
    given the number of nodes n and the iteration,
    returns the one-peer exponential graph of current stage
    '''

    tau = math.ceil(math.log(n, 2))
    graph = [[0] * n for i in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                graph[i][j] = 1/2
            elif ((j - i) % n) == (2 ** (iteration % tau)):
                graph[i][j] = 1/2
    
    return graph

def find_prime_factors(n: int) -> tuple:
    '''
    given a positive integer, 
    returns a tuple containing its prime factors,
    where the factors repeat its respective order number of times
    '''
    primes = []
    while n % 2 == 0:
        primes.append(2)
        n = n // 2

    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            primes.append(i)
            n = n // i
    
    if n > 2:
        primes.append(n)

    return tuple(primes)

# print(find_prime_factors(12))
# print(find_prime_factors(36))

def condition(i: int, j: int, multibased_representations: list, iteration: int) -> bool:
    '''
    given two base-10 integers i and j,
    determines whether there should be an edge between i and j in the p-peer hypercuboid.
    returns a tuple of integers with length equal to len(primes)
    '''
    tau = len(multibased_representations[0])
    count = 0 # counts number of different digits
    new_i = multibased_representations[i]
    new_j = multibased_representations[j]
    for idx in range(tau):
        if new_i[idx] != new_j[idx]:
            count += 1
            if count > 1:
                return False
            elif idx != (tau - iteration % tau - 1):
                return False
    
    return True

def p_peer_hypercuboids(n: int, iteration: int) -> list:
    '''
    given the number of agents and iteration,
    return the corresponding hypercuboid
    in the form of a list of lists
    '''
    # we first find prime factorization of n
    primes = find_prime_factors(n)
    tau = len(primes)
    graph = [[0] * n for i in range(n)]

    # find multibased_representation of all numbers
    multibased_representations = []
    for num in range(n):
        new_num = []
        for p in primes[::-1]:
            new_num.insert(0, num % p)
            num = num // p
        multibased_representations.append(tuple(new_num))

    # build the hypercuboid
    for i in range(n):
        for j in range(n):
            if condition(i, j, multibased_representations, iteration):
                graph[i][j] = 1 / (primes[-(iteration % tau) - 1])
            elif i == j:
                graph[i][j] = 1 / (primes[-(iteration % tau) - 1])
    return graph

# test cases

# print(p_peer_hypercuboids(12, 0))
# print(p_peer_hypercuboids(12, 1))
# print(p_peer_hypercuboids(12, 2))

# functions to visualize the graphs

def plot_multiple_graphs(incidence_matrices: list, names = None) -> None:
    """
    Takes a list of incidence matrices and plots each graph as subplots in a single figure.
    """
    num_graphs = len(incidence_matrices)
    cols = 3  # Number of columns in subplot grid
    rows = (num_graphs + cols - 1) // cols  # Calculate required number of rows
    
    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if num_graphs == 1:
        axes = [axes]  # Make it iterable if only one subplot
    
    # Flatten axes array if more than one row and column
    axes = axes.flatten() if isinstance(axes, np.ndarray) else axes
    
    for idx, matrix in enumerate(incidence_matrices):
        matrix = np.array(matrix)
        # Create a graph from an incidence matrix
        G = nx.from_numpy_array(matrix, create_using=nx.Graph)
        
        # Plot the graph
        nx.draw(G, ax=axes[idx], with_labels=True, node_color='skyblue', 
                node_size=700, font_size=12, font_color='darkred')
        if not names:
            axes[idx].set_title(f"Graph {idx+1}")
        else:
            axes[idx].set_title(names[idx])
    
    # Turn off axes for unused subplots
    for idx in range(num_graphs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_directed_graphs(incidence_matrices: list, names = None) -> None:
    """
    Takes a list of incidence matrices where a positive value at position (i, j) indicates
    an edge from node i to node j, and plots each directed graph as subplots in a single figure.
    """
    num_graphs = len(incidence_matrices)
    cols = 3  # Number of columns in subplot grid
    rows = (num_graphs + cols - 1) // cols  # Calculate required number of rows
    
    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if num_graphs == 1:
        axes = [axes]  # Make it iterable if only one subplot
    
    # Flatten axes array if more than one row and column
    axes = axes.flatten() if isinstance(axes, np.ndarray) else axes
    
    for idx, matrix in enumerate(incidence_matrices):
        matrix = np.array(matrix)
        G = nx.DiGraph()
        num_nodes = matrix.shape[0]
        
        # Building the graph from the incidence matrix
        for i in range(num_nodes):
            for j in range(num_nodes):
                if matrix[i, j] > 0:  # There is an edge from i to j
                    G.add_edge(i, j)
        
        # Plot the graph
        pos = nx.spring_layout(G)  # Positions for all nodes
        nx.draw(G, pos, ax=axes[idx], with_labels=True, node_color='skyblue', 
                node_size=700, font_color='darkred', arrows=True, arrowstyle='-|>')
        if names:
            axes[idx].set_title(names[idx])
        else:
            axes[idx].set_title(f"Directed Graph {idx+1}")
    
    # Turn off axes for unused subplots
    for idx in range(num_graphs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# functions to run main experiment

def run_experiment(num_of_iter: int, graph_topology: list, init_values: list, one_peer_exp_flag = False, p_peer_flag = False) -> float:
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
    # incidence_matrix = assign_weights(graph_topology)
    incidence_matrix = graph_topology
    errors = []

    # run the experiment
    for iteration in range(num_of_iter):
        agent_values, incidence_matrix = iterate_step(agent_values, incidence_matrix, iteration + 1, one_peer_exp_flag, p_peer_flag)
        error = calculate_squared_error(agent_values, average)
        if error < 1e-6:
            print("Consensus reached in " + str(iteration + 1) + " iterations")
            break
        errors.append(error)
        
    # plot the error and return lambda
    # print(errors)
    if errors:
        return plot_error(errors)
    else:
        return 0

# ------------------------------------------------------------------------------------------------
# # EXPERIMENT 1: connectivity and lambda

# # test graph No. 1: C_4
# topology1 =  [
#     [0, 1, 0, 1],  
#     [1, 0, 1, 0],  
#     [0, 1, 0, 1],  
#     [1, 0, 1, 0]   
# ]
# c_4 = assign_weights(topology1)
# # test graph No. 2: more connected than C_4, less than K_4
# topology2 = [
#     [0, 1, 1, 0],  
#     [1, 0, 1, 1],  
#     [1, 1, 0, 0],  
#     [0, 1, 0, 0]   
# ]
# four_nodes = assign_weights(topology2)
# # test graph No. 3: K_4
# topology3 = [ 
#     [0, 1, 1, 1],  
#     [1, 0, 1, 1],  
#     [1, 1, 0, 1],  
#     [1, 1, 1, 0]   
# ]
# k_4 = assign_weights(topology3)

# print(run_experiment(50, c_4, [10, 20, 30, 40]))
# print(run_experiment(50, four_nodes, [10, 20, 30, 40]))
# print(run_experiment(50, k_4, [10, 20, 30, 40]))

# # result: It seems that this relation is quite straightforward,
# # though more data is needed for a better understanding of how this works.
# # perhaps we can quantify graph connectivity (with its number of veritices and edges..?)
# # so that we can plot lambda against the connectivity quantifier
# ------------------------------------------------------------------------------------------------
# # EXPERIMENT 2: de bruijn Graphs

# 2.1 plotting some de bruijn graphs
# let us plot de bruijn graphs with p = 2, tau = 1, 2, 3, 4, 5
# as well as p = 3, tau = 1, 2, 3, 4, 5

# de_bruijn_graphs = []
# p = 2
# for tau in range(1, 6):
#     de_bruijn_graphs.append(de_bruijn(p, tau))    
# plot_directed_graphs(de_bruijn_graphs)

# 2.2 convergence 
# print(run_experiment(1000, de_bruijn(2, 3), [1, 2, 3, 4, 5, 6, 7, 8]))

# # result: consensus is reached in ~10 iterations, depending on the cutoff. NOT consisent.
# # THE CONSENSUS SHOULD HAVE BEEN REACHED IN three (3) ITERATIONS
# ------------------------------------------------------------------------------------------------
# # EXPERIMENT 3: One-peer Exponential graphs

# 3.1: plotting some one-peer exponential graphs
# let us plot the first 4 one-peer exponential graphs, starting with tau = 2

# one_p_graphs = []
# tau = 2
# for iteration in range(tau):
#     one_p_graphs.append(one_peer_exponential(2**tau, iteration))
# plot_directed_graphs(one_p_graphs)

# 3.2 consensus convergence
# print(run_experiment(100, one_peer_exponential(32, 0), [i+1 for i in range(32)], one_peer_exp_flag = True))

# # results: consistent with the argument made in paper which says that consensus should be reached in 5 iterations.
# ------------------------------------------------------------------------------------------------
# # EXPERIMENT 4: p-peer hyper-cuboids

# 4.1 plotting some p-peer exponential graphs
# lets plot graphs for n=8, 12, 16, 18, 24

# p_peer_graphs = []
# n = 24
# for iteration in range(len(find_prime_factors(n))):
#     p_peer_graphs.append(p_peer_hypercuboids(n, iteration))
# plot_multiple_graphs(p_peer_graphs)

# 4.2 consensus convergence
# print(run_experiment(100, p_peer_hypercuboids(12, 0), [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], p_peer_flag=True))

# # results: consistent with the argument made in paper which says that consensus should be reached in 3 iterations.