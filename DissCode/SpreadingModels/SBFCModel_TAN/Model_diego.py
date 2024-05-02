from abcpy.continuousmodels import ProbabilisticModel, Continuous, InputConnector
import numpy as np
import networkx as nx
import copy
from pathpy.classes import temporal_network, higher_order_network, network
from pathpy.path_extraction import temporal_paths
from SpreadingModels.SBFCModel.Model import FakeNewsSBFC

"""
#### This class inherits the SBFCModel for modelling the spread of fake news on a temporal network
----------
Parameters:
----------
parameters: list
    Contains the probabilistic models and hyper-parameters from which the model derives.
            theta: float
                (Optional) The spreading parameter for the fake news (in the interval [0,1].)
            gamma: float
                (Optional) The spreading parameter for the debunking (in the interval [0,1])
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


class FakeNewsSBFC_TAN_diego(FakeNewsSBFC):

    def __init__(self, parameters, temporal_net, time_observed, k_order=2, seed=None, name="FakeNewsSBFC_TN"):
        self.temporal_net = temporal_net
        self.temporal_net_time_steps = len(temporal_net)
        self.T = max(time_observed) + 1
        self.fake_news_exposed_observed_times = []
        self.debunking_exposed_observed_times = []

        if self.temporal_net_time_steps > self.T:
            raise ValueError("The duration of time observed on the spreading process must be greater than "
                             "the number of time steps of the temporal network")
        else:
            self.time_sync = int(np.ceil(self.T/self.temporal_net_time_steps))

        self.K = k_order
        k_order_network = self.compute_time_aggregated_network()
        self.k_order_network = k_order_network
        self.time_aggregated_network = network.network_to_networkx(network=k_order_network)

        super().__init__(parameters, temporal_net[0], time_observed, seed, name)

    def compute_time_aggregated_network(self):
        t_edges = self.get_time_indexed_edges()
        temp_net = temporal_network.TemporalNetwork(tedges=t_edges)

        # This is the point where consistency is lost, cannot add seed so as to maintain replication
        causal_paths = temporal_paths.paths_from_temporal_network_dag(tempnet=temp_net, delta=self.time_sync,
                                                                      max_subpath_length=self.K)
        #print(causal_paths)
        k_order_network = higher_order_network.HigherOrderNetwork(paths=causal_paths, k=self.K)
        return k_order_network
        '''time_aggregated_network = network.network_to_networkx(network=k_order_network)
        return time_aggregated_network
'''
    def get_time_indexed_edges(self):
        t_edges = list()
        for t in range(self.temporal_net_time_steps):
            for edge in self.temporal_net[t].edges():
                # The list of time-indexed edges are directed as per pathpy requirements
                u = edge[0]
                v = edge[1]
                t_edges.append((u, v, t))
                t_edges.append((v, u, t))
        return t_edges

    

    def simulate(self, theta, gamma, delta, seed_node, n_simulate):
        diffusion_state_array = [None] * n_simulate
        # Initialize local parameters

        for k in range(n_simulate):
            # Initialize the time-series
            tmp_diffusion_states = list()

            self.reset_node_status()
            self.reset_exposures()

            # Initialising the initially infected nodes, fact-checking nodes and their statuses
            fact_checker_nodes, believer_nodes = self.init_nodes(seed_node, theta)

            # Recording the results at t=0 if observed
            if 0 in self.time_observed:
                tmp_diffusion_states.append(copy.deepcopy(self.node_status))

            for t in range(1, self.T):
                # Move onto the next snapshot based on time_sync
                if t % self.time_sync == 0:
                    next_snapshot = self.temporal_net[int(t / self.time_sync)]
                    self.update_network(next_snapshot)

                # Run Spreading Processes
                new_fact_checker_nodes, new_believer_nodes = self.run_spreading_process(fact_checker_nodes,
                                                                                        believer_nodes, theta,
                                                                                        gamma, delta)

                fact_checker_nodes = copy.deepcopy(new_fact_checker_nodes)
                believer_nodes = copy.deepcopy(new_believer_nodes)

                # Record the results if t is observed
                if t in self.time_observed:
                    current_node_status = copy.deepcopy(self.node_status)
                    tmp_diffusion_states.append(current_node_status)
                    #print('self.fake news exposures:', self.fake_news_exposures)
                    self.record_exposures()

            # add results of the kth simulation
            diffusion_state_array[k] = np.array(tmp_diffusion_states).flatten()

        return diffusion_state_array

    def colour_TAN_all(self, infection_array):
        HO_nodes = self.k_order_network.nodes
        HO_ia = {}
        #print('Infection array', infection_array)
        for node in HO_nodes:
            split_node = node.split(',')
            state = -1
            for element in split_node:
                #print(element)
                if infection_array[int(element)] == 1:
                    state = 1
                if infection_array[int(element)] == 0:
                    state = 0
                    break
            HO_ia[node] = state
        return HO_ia

    def colour_TAN_fake_news_exposures(self, infection_array, Exposure_array):
        HO_nodes = self.k_order_network.nodes
        HO_ia = {}
        state = 1
        #print('Infection array', infection_array)
        for node in HO_nodes:
            split_node = node.split(',')
            for element in split_node:
                #print(element)
                if infection_array[int(element)] != 1:
                    if Exposure_array[int(element)] == 0:
                        state = 0
                        break
            HO_ia[node] = state
        return HO_ia
    
    def colour_TAN_debunking_exposures(self, infection_array, Exposure_array):
        HO_nodes = self.k_order_network.nodes
        HO_ia = {}
        state = 1
        #print('Infection array', infection_array)
        for node in HO_nodes:
            split_node = node.split(',')
            for element in split_node:
                #print(element)
                if infection_array[int(element)] != -1:
                    if Exposure_array[int(element)] == 0:
                        state = 0
                        break
            HO_ia[node] = state
        return HO_ia

    def _check_input(self, input_values):
        # raises exceptions if the input is of wrong type or has the wrong format.
        # returns False if the values of the input models are not compatible, True otherwise.
        if len(input_values) != 4:
            raise RuntimeError('Input parameters must be a list with 4 elements, parameters=(theta, gamma, delta, '
                               'seed_node).')

        if not isinstance(input_values, list):
            raise TypeError('Input parameters must be of type: list')

        return True

    def _check_output(self, values):
        for simulation_results in values:
            if len(simulation_results) != self.get_output_dimension():
                return False
        return True

    def get_time_aggregated_network(self):
        return self.time_aggregated_network

    def get_k_order_network(self):
        return self.k_order_network