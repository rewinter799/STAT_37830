import numpy as np
import matplotlib.pyplot as plt

def model(population):
    """
    Initiates and event at the center of the population
    and runs a simulation of opinion spread.

    Parameters
    ----------
    population : class population
    
    Returns: counter, int, number of time steps performed

    """
    # Initiate an event at the center of the population:
    population.event(int(np.size(population.agents)*population.inter))
    counter = 0
    c = 1
    # While opinions are still changing, perform a time step on the population:
    while c > 0:
        c = population.time_step()
        counter += 1
    # Run 10 more time steps in case the stop in opinion changes was an
    # an unlikely early stop due to stochasticity.
    for i in range(10):
        population.time_step()
        counter += 1
    return counter
        
def opinion_count(population):
    """
    Counts percentage of population with each opinion

    Parameters
    ----------
    population : class population

    Returns: Percentage holding opinions 'L', 'R' and neutral respectively.

    """
    N = np.size(population.agents)
    n = 0
    l = 0
    r = 0
    for i in range(N):
        if population.agents[i].final_opinion == 'L':
            l += 1
        elif population.agents[i].final_opinion == 'R':
            r += 1
        else:
            n += 1
    return np.array([l/N, r/N, n/N])
                   
def cluster(population):
    """
    Measures the degree of clusterinig, or percentage of neighbors that
    have the same opinion

    Parameters: population class

    Returns: m, mean of clustering across population
            var, variance of clustering across population

    """
    N = np.size(population.agents)
    C = np.zeros(N)
    for i in range(N):
        n = 0
        f = population.agents[i].final_opinion
        s = np.size(population.agents[i].neighbors)
        for j in range(s):
            if population.agents[i].neighbors[j].final_opinion == f:
                   n += 1
        C[i] = n/s
    m = np.mean(C)
    var = np.var(C)
    return np.array([m, var])

