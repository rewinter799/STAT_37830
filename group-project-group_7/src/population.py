from agents import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spatial

class population:
    
    def __init__(self, pop_size, interconnectivity, prob_declaration, 
                opinion_strength, beta, frac_influencer = 0, radius_influencer = 0.5):
        """
        Initialize a population of agents.

        Parameters
        ----------
        pop_size : int
            Number of desired agents.
        interconnectivity : float
            % of population any given agent is neighbors with.
        prob_declaration : float in [0, 1]
            Probability an agent will delcare an opinion
        opinion_strength : float
            Effect of other's opinions on an agent
        beta : float
            Learning paramenter or specifically
            Standard deviation of the stochasitc effect of learning.
        frac_influencer: float in [0, 1]
            The proportion of agents in the population who
            are influencers. Equivalently, the probability
            that each agent is an influencer.
        radius_influencer : float
            The radius of each influencer's influence.

        Attributes
        ----------
        agents : ndarray
            Set of agents comprising the population.
        inter : float
            % of population any given agent is neighbors with.
        pop_size : int
            Number of agents in population.
        tree : KDTree
            kd-tree partitioning agents for quick nearest-neighbor lookup.
        influencers : ndarray
            Set of agents in the population who are influencers.
        """
        # Initialize empty array of agents
        agents = np.empty(shape = pop_size, dtype = agent)
        agent_coords = np.empty(shape = (pop_size, 2))

        # Populate array of agents and corresponding array of their coordinates
        for i in range(len(agents)):
            agents[i] = agent(i, pop_size, interconnectivity, prob_declaration, 
                                  opinion_strength, beta, frac_influencer, radius_influencer)
            agent_coords[i] = agents[i].coords

        # Assign attributes to population object
        self.agents = agents
        self.inter = interconnectivity
        self.pop_size = pop_size
        self.tree = spatial.KDTree(agent_coords)

        # Determine set of influencers in the population
        influencers = []
        for i in range(len(agents)):
            if agents[i].influencer == 1:
                influencers.append(i)
        self.influencers = np.array(influencers, dtype = int)

        # Each agent identifies their own neighbors
        for i in range(len(agents)):
            agents[i].find_neighbors(self)

    def __str__(self):
        return f'Population of agents size: {self.pop_size}'
    
    def __repr__(self):
        return f'Class population of {self.pop_size} agent class objects'
    
    ### Old ("brute force") approach to modeling an event 
    #
    # def event(self, nwitness, location = np.array([0, 0])):
    #     """
    #     Determines which agents initially witnessed the neutral event, and who
    #     will start the chain of opinion propagation.
    
    #     Parameters:
    #     nwitness: The number of agents who initially witness the neutral
    #                    event.
    #     location: 2-vector- location of event, default [0, 0].
    #     """
    #     # Initialize empty array of witnesses
            
    #     # Compute distances from each agent to the event (at 0,0), and identify 
    #     # the num_witnesses agents closest to the event.
    #     distances = np.zeros(len(self.agents))
    #     for i in range(len(self.agents)):
    #         distances[i] = np.linalg.norm(self.agents[i].coords - location)   
    #     witness_indices = np.argpartition(distances, nwitness)
    #     witness_indices = witness_indices[:nwitness]

    #     # Populate the neighbors array of the agent.
    #     for i in witness_indices:
    #         self.agents[i].learn()

    def event(self, nwitness, location = np.array([0, 0])):
        """
        Determines which agents initially witnessed the neutral event, and who
        will start the chain of opinion propagation, using a kd-tree.
    
        Parameters:
        nwitness: The number of agents who initially witness the neutral
                       event.
        location: 2-vector- location of event, default [0, 0].
        """
        witness_indices = self.tree.query(location, nwitness)[1]
        for i in witness_indices:
            self.agents[i].learn()
    
    def time_step(self):
        """
        Cycle through all agents and allows agents with an opinion to 
        delcare their opinion (spread opinion to others.)

        All agents will be updated with new 'final opinion' when applicable.
        
        Returns: c, int, number of changes in opinion
        (used to terminate models).
        """
        for j in range(len(self.agents)):
            self.agents[j].declare()
        c = 0
        for j in range(len(self.agents)):
            g = self.agents[j].gradient_opinion
            self.agents[j].attend()
            if self.agents[j].gradient_opinion != 0:
                self.agents[j].learn()
                if g - self.agents[j].gradient_opinion != 0:
                    c += 1
        return c
    
    def display(self, filename=False):
        """
        Displays the current state of the simulation.
    
        Parameters:
        filename: str or bool. 
        If a string is input, save the 
        plot with file name filename instead of displaying.
        """
        population = self.agents
        # Set grid dimensions
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
      
        # Compile positions and opinions (color-coded) of population of agents
        population_xcoords = np.zeros(shape = len(population))
        population_ycoords = np.zeros(shape = len(population))
        population_opinions = np.empty(shape = len(population), dtype = str)
        population_influencers = np.empty(shape = len(population), dtype = str)
        opinion_dict = {"neutral": "k",
                        "L": "b",
                        "R": "r"}
        influencer_dict = {"1" : "s",
                           "0" : "o"}

        for i in range(len(population)):
            population_xcoords[i] = population[i].coords[0]
            population_ycoords[i] = population[i].coords[1]
            population_opinions[i] = opinion_dict[population[i].final_opinion]
            population_influencers[i] = population[i].influencer
    
        # Plot
        for i in range(len(population)):
            plt.scatter(population_xcoords[i], population_ycoords[i],
                        c = population_opinions[i],
                        marker = influencer_dict[population_influencers[i]])

        # Aesthetics
        plt.grid(color = "k", alpha = 0.5)

        # Place initial event at (0,0)
        plt.scatter(0, 0, marker = "*", c = "k", s = 150)
        
        # Save figure
        if filename== False:
            plt.show()
        elif type(filename) == str:
            plt.savefig(filename)
            plt.clf()