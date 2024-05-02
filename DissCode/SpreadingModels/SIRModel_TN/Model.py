from abcpy.continuousmodels import ProbabilisticModel, Continuous, InputConnector
import numpy as np
import networkx as nx
import copy
from SpreadingModels.SIRModel.Model import FakeNewsSIR

"""
#### This class inherits the SBFCModel for modelling the spread of fake news on a temporal network
----------
Parameters:
----------
parameters: list
    Contains the probabilistic models and hyper-parameters from which the model derives.
            theta: float
                (Optional) The spreading parameter for the fake news (in the interval [0,1].)
            delta: float
                (Optional) The recovery parameter (in the interval [0,1])
            seed_node: integer
                The node where the infection starts.
temporal_net: a list of NetworkX.Graph objects
    The temporal network structure on which news spreads.
time_observed: numpy.ndarray
    The time-points on the spreading process at which the observation were made.
seed: int
    seed for RNG. Default: None
----------
"""


class FakeNewsSIR_TN(FakeNewsSIR):

    def __init__(self, parameters, temporal_net, time_observed, seed=None, name="FakeNewsSBFC_TN"):
        self.temporal_net = temporal_net
        self.temporal_net_time_steps = len(temporal_net)
        self.T = max(time_observed) + 1

        if self.temporal_net_time_steps > self.T:
            raise ValueError("The duration of time observed on the spreading process must be greater than "
                             "the number of time steps of the temporal network")
        else:
            self.time_sync = int(np.ceil(self.T/self.temporal_net_time_steps))

        super().__init__(parameters, temporal_net[0], time_observed, seed, name)

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
                if t % self.time_sync == 0:
                    next_snapshot = self.temporal_net[int(t / self.time_sync)]
                    self.update_network(next_snapshot)

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