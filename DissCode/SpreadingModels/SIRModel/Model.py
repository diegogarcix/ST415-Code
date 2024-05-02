from abcpy.continuousmodels import ProbabilisticModel, Continuous, InputConnector
import numpy as np
import networkx as nx
import copy

"""
#### This class implements a simple SIR model for the spread of fake news on a network
----------
Parameters:
----------
parameters: list
    Contains the probabilistic models and hyper-parameters from which the model derives.
            theta: float
                (Optional) The spreading parameter for the fake news (in the interval [0,1].)
            gamma: float
                (Optional) The recovery rate for infected nodes (in the interval [0,1])
            seed_node: integer
                The node where the infection starts.
network: NetworkX.Graph
    The fixed network structure on which news spreads.
time_observed: numpy.ndarray
    The time-points at which the observation were made.
seed: int
    seed for RNG. Default: None
----------
"""


class FakeNewsSIR(ProbabilisticModel, Continuous):

    def __init__(self, parameters, network, time_observed, seed=None, name="FakeNewsSIR"):

        self.node_count = network.number_of_nodes()
        self.network = network
        self.T = max(time_observed) + 1
        self.time_observed = time_observed
        self.rng = np.random.RandomState(seed)

        # We expect input of type parameters = [theta, gamma, seed_node]
        self._check_input(parameters)
        input_parameters = InputConnector.from_list(parameters)
        super().__init__(input_parameters, name)

    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        # Initialize local parameters
        #       theta - spreading rate for the fake news
        #       gamma - recovery rate for infected nodes
        theta = input_values[0]
        gamma = input_values[1]
        seed_node = [input_values[2]]

        #print(theta, gamma, seed_node)
        result = self.simulate(theta, gamma, seed_node, k)

        if not self._check_output(result):
            raise RuntimeError("Result is not a 1-dimensional numpy array.")

        # The results of forward simulate should be an 1-dimensional numpy array for each repetition. So need to think
        # how to compress the data. One option is code infected/non-infected as 0/1 and recovered/ow as 0/1 .. then
        # for each repetition you will have the same length.
        #print(result)
        return result

    def simulate(self, theta, gamma, seed_node, n_simulate):
        diffusion_state_array = [None] * n_simulate
        # Initialize local parameters
        for k in range(n_simulate):
            # Initialize the time-series
            tmp_diffusion_states = list()

            # node_status encodes the status of each node as:
            #   0: susceptible, 1: infected, -1: recovered
            # All nodes initialised to susceptible
            node_status = np.zeros(self.node_count)

            # Setting the status for the seed node
            infected_nodes = list(seed_node)
            present_infected_nodes = copy.deepcopy(infected_nodes)
            for seed_nodes in infected_nodes:
                node_status[seed_nodes] = 1

            # Adding the observation at t=0 (if applicable)
            if 0 in self.time_observed:
                tmp_diffusion_states.append(copy.deepcopy(node_status))

            for t in range(1, self.T):
                for i in present_infected_nodes:
                    # Infected nodes attempting to debunk; skips spreading phase if successful
                    if self.rng.binomial(1, gamma) == 1:
                        infected_nodes.remove(i)
                        node_status[i] = -1
                        continue

                    # Choosing one neighbouring node to spread the fake news to
                    chosen_node_for_infection = self.rng.choice(list(self.network.neighbors(i)), 1)[0]

                    # Attempt to infect the neighbour if it is susceptible
                    if node_status[chosen_node_for_infection] == 0:
                        # Attempting to infect the neighbour
                        if self.rng.binomial(1, theta) == 1:
                            infected_nodes.append(chosen_node_for_infection)
                            node_status[chosen_node_for_infection] = 1

                present_infected_nodes = copy.deepcopy(infected_nodes)
                current_node_status = copy.deepcopy(node_status)

                if t in self.time_observed:
                    tmp_diffusion_states.append(current_node_status)

            # add results of the kth simulation
            diffusion_state_array[k] = np.array(tmp_diffusion_states).flatten()

        return diffusion_state_array

    def _check_input(self, input_values):
        # raises exceptions if the input is of wrong type or has the wrong format.
        # returns False if the values of the input models are not compatible, True otherwise.
        if len(input_values) != 3:
            raise RuntimeError('Input parameters must be a list with 3 elements, parameters=(theta, gamma, seed_node).')

        if not isinstance(input_values, list):
            raise TypeError('Input parameters must be of type: list')

        '''
        if input_values[0] < 0 or input_values[0] > 1 or input_values[1] < 0 or input_values[1] > 1 \
                or input_values[2] < 0 or input_values[2] > self.node_count - 1:
            raise ValueError("The parameter values are out of the model parameter domain.")
        
        # self.theta = input_values[0]
        # self.gamma = input_values[1]
        # self.seed_node = input_values[2]
        '''
        return True

    def _check_output(self, values):
        for simulation_results in values:
            if len(simulation_results) != self.get_output_dimension():
                return False
        return True
    
    def update_network(self, network):
        self.network = network

    def get_output_dimension(self):
        return len(self.time_observed) * self.node_count
