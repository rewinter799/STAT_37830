import sys
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
import numpy as np
sys.path.append("D:/zxm2023/Python/group-project-group_7/src")
#sys.path.append("/home/mjfraizer/Desktop/Python_Course/group-project-group_7/src")
from tools import *
from agents import *
from population import *

# Create a population of 5 agents, define the interconnectivity to be 0.2
# making the neighbor to be 1 person with the smallest distance
nagents = 5
interconnectivity = 0.2
pop = np.empty(shape = nagents, dtype = agent)
for i in range(len(pop)):
    pop[i] = agent(i, nagents, interconnectivity, prob_declaration=0.5,
                          opinion_strength=0.1, beta=0.2)

class TestAgentMethods(unittest.TestCase):

    def setUp(self):
        # Set up a simple population with a few agents for testing
        self.test_population = population(
            pop_size=5,
            interconnectivity=0.2,
            prob_declaration=0.2,
            opinion_strength=0.3,
            beta=0.1,
            frac_influencer=0.2,
            radius_influencer=0.5
        )

    def test_find_neighbors(self):
        # Create an agent with known coordinates
        self.test_population.agents[0].coords = np.array([0, 0])
        self.test_population.agents[1].coords = np.array([-0.8, 0.3])
        self.test_population.agents[2].coords = np.array([0.3, -0.1])
        self.test_population.agents[3].coords = np.array([-0.2, 0.1])
        self.test_population.agents[4].coords = np.array([0, 0.1])

        test_agent = self.test_population.agents[0]

        # Create an ndarray collecting the coordinates of all the agents
        test_pop_coords = np.empty(shape = (len(self.test_population.agents), 2))
        for i in range(len(self.test_population.agents)):
            test_pop_coords[i] = self.test_population.agents[i].coords

        self.test_population.tree = spatial.KDTree(test_pop_coords)

        # Recalculate neighbors with our new (manual) agent positions and KDTree
        for agent in self.test_population.agents:
            agent.find_neighbors(self.test_population)

        # Get the influencer_indicies, exclude influencer if either:
        # agent itself
        # or agent 1 which Euclidean distance is beyond the defined radius 0.5
        influencer_indices = [i for i, agent in enumerate(self.test_population.agents) 
                              if agent.influencer == 1 & agent.index != 0 & agent.index != 1]

        # Call the find_neighbors method
        test_neighbors = test_agent.neighbors
        test_neighbor_indices = [neighbor.index for neighbor in test_neighbors]

        # Check if the neighbors attribute is populated correctly
        self.assertIn(4, test_neighbor_indices)
        if influencer_indices != []:
            self.assertIn(influencer_indices, test_neighbor_indices)


class TestAttend(unittest.TestCase):
    '''
    Test attend method of the Agent Class
    '''
    # Has to use different test agent in each test or reset the
    # population to initial state
    def setUp(self):
	    pass

    def test_attend_neighbor_declares_L(self):
        # Create a neutral agent for testing
        self.test_agent = pop[0]
        self.test_agent.final_opinion = "neutral"
        # Mock behavior of a neighbor declaring "L"
        neighbor1 = MagicMock(declare=MagicMock(return_value=True), final_opinion="L")
        neighbor2 = MagicMock(declare=MagicMock(return_value=True), final_opinion="L")
        self.test_agent.neighbors = [neighbor1, neighbor2]

        # Call the attend method
        self.test_agent.attend()

        # Check if gradient_opinion is updated correctly
        expected_gradient_opinion = -0.2  # opinion_strength for "L" declaration
        self.assertEqual(self.test_agent.gradient_opinion, expected_gradient_opinion)

    def test_attend_neighbor_declares_R(self):
        self.test_agent = pop[1]
        self.test_agent.final_opinion = "neutral"
        # Mock behavior of a neighbor declaring "R"
        neighbor3 = MagicMock(declare=MagicMock(return_value=True), final_opinion="R")
        self.test_agent.neighbors = [neighbor3]

        # Call the attend method
        self.test_agent.attend()

        # Check if gradient_opinion is updated correctly
        expected_gradient_opinion = 0.1  # opinion_strength for "R" declaration
        self.assertEqual(self.test_agent.gradient_opinion, expected_gradient_opinion)

    def test_attend_neighbor_does_not_declare(self):
        self.test_agent = pop[2]
        self.test_agent.final_opinion = "neutral"
        # Mock behavior of a neighbor not declaring
        neighbor4 = MagicMock(declare_flag = False, final_opinion="R")
        self.test_agent.neighbors = [neighbor4]

        # Call the attend method
        self.test_agent.attend()

        # Check if gradient_opinion remains unchanged
        self.assertEqual(self.test_agent.gradient_opinion, 0)

class TestLearn(unittest.TestCase):
    '''
    Test the learn method of the Agent Class
    '''
    def setUp(self):
	    pass

    def test_learn_opinion_update_negative(self):
        self.test_agent = pop[0]
        self.test_agent.final_opinion = "neutral"
        self.test_agent.gradient_opinion = -0.1
        # Mock behavior of rand.normal to always return a negative value
        with patch('numpy.random.randn', return_value=-0.2):
            self.test_agent.learn()

        # Check if gradient_opinion is updated correctly
        expected_gradient_opinion = -0.14  # beta * delta
        self.assertAlmostEqual(self.test_agent.gradient_opinion, expected_gradient_opinion, places=3)

        # Check if final_opinion is updated correctly
        self.assertEqual(self.test_agent.final_opinion, "L")

    def test_learn_opinion_update_positive(self):
        self.test_agent = pop[1]
        self.test_agent.final_opinion = "neutral"
        self.test_agent.gradient_opinion = 0.2
        # Mock behavior of rand.normal to always return a positive value
        with patch('numpy.random.randn', return_value=0.5):
            self.test_agent.learn()

        # Check if gradient_opinion is updated correctly
        expected_gradient_opinion = 0.3  # beta * delta
        self.assertAlmostEqual(self.test_agent.gradient_opinion, expected_gradient_opinion, places=3)

        # Check if final_opinion is updated correctly
        self.assertEqual(self.test_agent.final_opinion, "R")

    def test_learn_opinion_no_update(self):
        self.test_agent = pop[1]
        self.test_agent.final_opinion = "neutral"
        self.test_agent.gradient_opinion = 0
        # Mock behavior of rand.normal to return 0
        with patch('numpy.random.randn', return_value=0.0):
            self.test_agent.learn()

        # Check if gradient_opinion remains unchanged
        self.assertEqual(self.test_agent.gradient_opinion, 0)

        # Check if final_opinion is still "neutral"
        self.assertEqual(self.test_agent.final_opinion, "neutral")


class TestDeclare(unittest.TestCase):
    '''
    Test the declare method of the agent class
    '''
    def setUp(self):
	    pass

    def test_declare_probability_true(self):
        # Let the test_agent be the first aget in the defined population
        # With the predefined prob_declaration value = 0.5
        test_agent = pop[0]
        test_agent.final_opinion = 'L'

        # Patch rand.uniform to always return a value less than or equal to 0.5
        # Run 100 trials
        for _ in range(100):
            test_agent.declare_flag = False
            with patch('numpy.random.uniform', return_value=0.5):
                test_agent.declare()
                result = test_agent.declare_flag
            # The result should be True since the draw is <= prob_declaration=0.5
            self.assertTrue(result)


class TestPopulationInitialization(unittest.TestCase):
    '''
    test whether the attribute are correctly set up
    and intialized in the population class
    '''
    def setUp(self):
        '''
        Initializes a population of agents, generates agent objects,
        and sets up the KDTree for quick nearest-neighbor lookup.
        It also identifies influencers within the population.
        '''
        self.pop_size = 10
        self.interconnectivity = 0.5
        self.prob_declaration = 0.2
        self.opinion_strength = 0.7
        self.beta = 0.1
        self.frac_influencer = 0.2
        self.radius_influencer = 0.5
        self.population = population(self.pop_size, self.interconnectivity,
                                     self.prob_declaration, self.opinion_strength,
                                     self.beta, self.frac_influencer,
                                     self.radius_influencer)

    def test_population_attributes(self):
        self.assertEqual(len(self.population.agents), self.pop_size)
        self.assertEqual(self.population.inter, self.interconnectivity)
        self.assertEqual(self.population.pop_size, self.pop_size)
        self.assertIsNotNone(self.population.tree)
        self.assertIsInstance(self.population.influencers, np.ndarray)
        for i in range(self.pop_size):
            self.assertEqual(len(self.population.agents[i].neighbors), 5)

    def test_agent_initialization(self):
        for agent_obj in self.population.agents:
            self.assertIsInstance(agent_obj, agent)
            self.assertIn(agent_obj.influencer, [0, 1])
            self.assertGreaterEqual(agent_obj.radius_influencer, 0.0)

class TestPopulationEvent(unittest.TestCase):
    '''
    Test the event method is the population Class
    '''
    def test_event(self):
        # Create a population with known coordinates
        self.pop_size = 5
        self.interconnectivity = 0.5
        self.prob_declaration = 0.2
        self.opinion_strength = 0.7
        self.beta = 0.1
        self.frac_influencer = 0.2
        self.radius_influencer = 0.5
        self.pop = population(self.pop_size, self.interconnectivity,
                                     self.prob_declaration, self.opinion_strength,
                                     self.beta, self.frac_influencer,
                                     self.radius_influencer)

        # Call the event method
        nwitness = 2
        location = np.array([0, 0])
        self.pop.event(nwitness, location)
        print(self.pop.event)
        # Check there are in total 2 agents (witness) having final_opinion that is not "neutral"
        count_non_neutral = sum(agent.final_opinion != "neutral" for agent in self.pop.agents)
        self.assertEqual(count_non_neutral, 2)

class TestPopulationTimeStep(unittest.TestCase):
    '''
    Test the time_step method in the population Class
    '''

    def setUp(self):
        # Set up a population
        self.pop_size = 10
        self.interconnectivity = 0.5
        self.prob_declaration = 0.2
        self.opinion_strength = 0.7
        self.beta = 0.1
        self.frac_influencer = 0.2
        self.radius_influencer = 0.5
        self.population = population(self.pop_size, self.interconnectivity,
                                     self.prob_declaration, self.opinion_strength,
                                     self.beta, self.frac_influencer,
                                     self.radius_influencer)

    def test_time_step_method(self):
        # Test the time_step method
        changes = self.population.time_step()
        # Add assertions based on the expected behavior of the time_step method
        self.assertGreaterEqual(changes, 0)
