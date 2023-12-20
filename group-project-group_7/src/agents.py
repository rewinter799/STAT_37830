import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import numpy.random as rand
import scipy.spatial as spatial

class agent:
    """
    Class of agents who participate in information propagation.
    """
    def __init__(self, index, pop_size, interconnectivity, prob_declaration, 
                 opinion_strength, beta, frac_influencer = 0,
                 radius_influencer = 0.5):
        """
        Instantiates an agent object.

        Parameters:
        index : int
            A unique integer identifier for the agent.
        pop_size : int
            The total number of agents in the population being modeled.
        interconnectivity : float in [0, 1]
            The proportion of the population in the agent's group of
            neighbors.
        prob_declaration : float in [0, 1]
            The agent's probability of declaring their beliefs in any given
            turn.
        opinion_strength : float
            The intensity with which an agent declares her opinion to her
            neighbor. Used by an agent to update his opinion after attending
            to each of his declaring neighbors. 
        beta : float
            Learning parameter. Captures how impactful the learning stage
            is on the agent's opinion formation.
        frac_influencer : float in [0, 1]
            The probability that the agent is an influencer.
        radius_influencer : float
            The radius of each influencer's influence.

        Attributes:
        index : int
            A unique integer identifier for the agent.
        coords : ndarray in [[-1, 1], [-1, 1]]
            The x- and y-coordinates of the agent in space.
        gradient_opinion : float in [-1, 1]
            The agent's opinion of the event, ranging from -1 to 1, and
            starting at 0.
        final_opinion : str
            The agent's final opinion of the event, based on their
            gradient_opinion. Starts at "neutral." "L" corresponds 
            to gradient_opinion values between [-1, 0) and "R" 
            corresponds to gradient-opinion values between (0, 1].
        interconnectivity : float in [0, 1]
            What proportion of the population is in the agent's group of
            neighbors.
        neighbors : ndarray
            The agent's neighbors, consisting of (a) the num_neighbors agents
            nearest to the agent in space, and (b) any additional neighbors
            who are influencers (i.e., have influencer == 1) and are within
            distance radius_influencer from the agent.
        prob_declaration : float in [0, 1]
            The agent's probability of declaring their beliefs in any given
            turn.
        opinion_strength : float 
            The intensity with which the agent declares her opinion to her
            neighbor. Used by an agent to update his opinion after attending
            to each of his declaring neighbors.
        beta : float
            Learning parameter. Captures how impactful the learning stage
            is on the agent's opinion formation.
        influencer : int
            Indicator variable for whether agent is an influencer.
        """
        # Basic information: Index and X-Y Coordinates
        self.index = index
        self.coords = rand.uniform(low = -1, high = 1, size = 2)

        # Initial neutral opinions
        self.gradient_opinion = 0
        self.final_opinion = "neutral"

        # Initial empty network of neighbors (to be generated after entire
        # population has been generated)
        self.interconnectivity = interconnectivity
        num_neighbors = int(pop_size * interconnectivity)
        self.neighbors = np.empty(shape = num_neighbors, dtype = agent)

        # Parameters affecting propagation of opinion
        self.prob_declaration = prob_declaration
        self.beta = beta
        self.opinion_strength = opinion_strength
        self.declare_flag = False

        # Determine if agent is an influencer
        if rand.rand() <= frac_influencer:
            self.influencer = 1
        else:
            self.influencer = 0

        # Set radius of influencers the agent listens to
        self.radius_influencer = radius_influencer

    def __str__(self):
        """
        String representation of agents is given by their index.

        Examples: "Agent 0", "Agent 1", "Agent 100"
        """
        return f"Agent {self.index}"

    def __repr__(self):
        """
        Official representation of agent, consisting of index and X-Y 
        coordinates.
        """
        return f"Agent({self.index}, ({self.coords[0]}, {self.coords[1]}))"

    ### Old ("brute force") approach to finding neighbors
    
    def find_neighbors_old(self, population):
        """
        Identifies and populates the agent's nearest neighbors.

        Parameters:
        population: An ndarray consisting of all nagents agents in the
        simulation.
        """
        # Compute distances between agent and all other agents in the
        # population (including itself). Set distance of agent to itself to
        # infinity, since we don't want to count an agent as its own neighbor.
        distances = np.zeros(len(population.agents))
        for i in range(len(population.agents)):
            distances[i] = np.linalg.norm(self.coords - population.agents[i].coords)
            if distances[i] == 0:
                distances[i] = np.inf
        
        # For debugging:
        # self.distances = distances

        # Identify the num_neighbors smallest distances between the
        # agent and all other agents in the population.
        neighbor_indices = np.argpartition(distances, len(self.neighbors))
        
        # Populate the neighbors array of the agent.
        for i in range(len(self.neighbors)):
            self.neighbors[i] = population.agents[neighbor_indices[i]]

    def find_neighbors(self, population):
        """
        Identifies and populates the agent's nearest neighbors using a
        KD Tree algorithm.

        Parameters:
        population: A population object corresponding to the population to which
                    the agent belongs.
        """
        tree = population.tree
        
        # Identify the agent's num_neighbors nearest neighbors.
        # We start with the "2nd-nearest neighbor" b/c the tree identifies
        # the agent's nearest neighbor as itself.
        num_neighbors = len(self.neighbors)
        neighbors_indices = tree.query(self.coords,
                                       k = num_neighbors + 1)[1]
        neighbors_indices = np.setdiff1d(neighbors_indices, np.array([self.index]))
        neighbors_indices = neighbors_indices.astype(int)
        
        # Identify the influencers within radius r of agent.
        # Again, we remove the agent itself.
        influencers_indices = np.intersect1d(tree.query_ball_point(self.coords, r = self.radius_influencer),
                                             population.influencers)
        influencers_indices = np.setdiff1d(influencers_indices, np.array([self.index]))
        influencers_indices = influencers_indices.astype(int)

        # Agent's neighbors are the combination of their nearest neighbors
        # and their influencers.
        neighbors = population.agents[np.union1d(neighbors_indices, influencers_indices)]
        self.neighbors = neighbors

    def attend(self):
        """
        Neutral agent attends to each neighbor in her network, checks if he
        is declaring his opinion, and updates her gradient opinion
        accordingly.
        """
        # Simplification assumption: Only agents who have neutral opinions
        # can update their opinions.
        if self.final_opinion == "neutral":
            for i in range(len(self.neighbors)):
                # Neighbor i declares L
                if self.neighbors[i].declare_flag and self.neighbors[i].final_opinion == "L":
                    self.gradient_opinion -= self.opinion_strength
                # Neighbor i declares R
                elif self.neighbors[i].declare_flag and self.neighbors[i].final_opinion == "R":
                    self.gradient_opinion += self.opinion_strength

    def learn(self):
        """
        Agent evaluates her opinion in light of new evidence using a
        reinforcement learning model (see Rescorla & Wagner, 1972).

        Agent then reaches her final opinion, provided her gradient opinion
        is not still 0.

        Parameters:
        delta_sd: A non-negative float representing the standard deviation
                  of the agent's "prediction error." Essentially determines
                  how much the agent's opinion can swing one way or the other
                  during the learning stage.
        """
        # Simplification assumption: Only agents who have neutral opinions
        # can update their opinions.
        if self.final_opinion == "neutral":
            delta = rand.randn() # Random "prediction error"
            self.gradient_opinion = self.gradient_opinion + self.beta * delta
            # Determine final opinion
            if self.gradient_opinion < 0:
                self.final_opinion = "L"
            elif self.gradient_opinion > 0:
                self.final_opinion = "R"

    def declare(self):
        """
        Randomly determines whether the agent will declare her opinion to
        her neighbors in a given turn.
        """
        if self.final_opinion != 'neutral':
            draw = rand.uniform()
            if draw <= self.prob_declaration:
                self.declare_flag = True
            else:
                self.declare_flag = False

