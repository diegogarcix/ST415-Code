import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathpy.classes import temporal_network, higher_order_network, network
from pathpy.path_extraction import temporal_paths
from SpreadingModels.SBFCModel.Model import FakeNewsSBFC

"""
#### This class inherits the SBFCModel class for modelling the spread of fake news on a time-aggregated temporal network
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
k_order: int
    the order of which the higher order time-aggregated network belongs to. Default: 2
seed: int
    seed for RNG. Default: None
----------
"""


class FakeNewsSBFC_TAN(FakeNewsSBFC):
    def __init__(self, parameters, temporal_net, time_observed, k_order=2, seed=None, name="FakeNewsSBFC_TAN"):
        self.temporal_net = temporal_net
        self.temporal_net_time_steps = len(temporal_net)
        self.K = k_order
        self.T = max(time_observed) + 1

        if self.temporal_net_time_steps > self.T:
            raise ValueError("The duration of time observed on the spreading process must be greater than "
                             "the number of time steps of the temporal network")
        else:
            self.time_sync = int(np.ceil(self.T/self.temporal_net_time_steps))

        k_order_network = self.compute_time_aggregated_network()
        self.k_order_network = k_order_network
        self.time_aggregated_network = network.network_to_networkx(network=k_order_network)
        super().__init__(parameters, self.time_aggregated_network, time_observed, seed, name)
        self.node_dict = self.init_node_dict()

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

    def run_spreading_process(self, fact_checker_nodes, believer_nodes, theta, gamma, delta):
        # FACT CHECKING SPREADING PROCESS
        # Create a placeholder variable to track new fact-checker nodes from the debunking process
        new_fact_checker_nodes = copy.deepcopy(fact_checker_nodes)
        for i in fact_checker_nodes:
            neighbours = list(self.network.neighbors(i))
            chosen_nodes_for_debunking = self.rng.choice(neighbours, int(np.ceil(len(neighbours) * gamma)))
            #print("Fact-checking Node:", i)
            for node in chosen_nodes_for_debunking:
                # Skip debunking process if node is not a believer
                # node_ind uses node_dict to retrieve the index for the node
                node_ind = self.node_dict[node]
                if self.node_status[node_ind] != 1:
                    continue

                success_probability = self.compute_infection_probability(node, gamma, process_type="debunking")
                #print("Spreading fact-checking to node:", node)
                #print("Success probability = ", success_probability)
                if self.rng.binomial(1, success_probability) == 1:
                    new_fact_checker_nodes.append(node)
                    believer_nodes.remove(node)
                    self.node_status[node_ind] = -1

        # FAKE NEWS SPREADING PROCESS
        # Get new believer nodes based on the ones debunked successfully during the fact checking process and
        # create a placeholder variable to track new believer nodes from the fake news spreading process
        new_believer_nodes = copy.deepcopy(believer_nodes)
        for j in believer_nodes:
            #print("\nBeliever Node:", j)
            # Believer node attempts to self fact-check using Bernoulli(delta), where delta is the recovery rate
            if self.rng.binomial(1, delta) == 1:
                new_fact_checker_nodes.append(j)
                new_believer_nodes.remove(j)
                self.node_status[self.node_dict[j]] = -1
                continue

            neighbours = list(self.network.neighbors(j))
            chosen_nodes_for_infecting = self.rng.choice(neighbours, int(np.ceil(len(neighbours) * theta)))
            #print("Neighbours:", neighbours)
            #print("Chosen for Infection:", chosen_nodes_for_infecting)
            for node in chosen_nodes_for_infecting:
                # Skip infection process if node is not susceptible
                # node_ind uses node_dict to retrieve the index for the node
                node_ind = self.node_dict[node]
                if self.node_status[node_ind] != 0:
                    continue

                success_probability = self.compute_infection_probability(node, theta, process_type="fake_news")
                #print("Spreading fake news to node:", node)
                #print("Success probability = ", success_probability)
                if self.rng.binomial(1, success_probability) == 1:
                    new_believer_nodes.append(node)
                    self.node_status[node_ind] = 1

        return new_fact_checker_nodes, new_believer_nodes

    def update_debunking_exposures(self, node, node_neighbours):
        FC_neighbours_count = 0
        for adj_node in node_neighbours:
            if self.node_status[self.node_dict[adj_node]] == -1:
                FC_neighbours_count += 1

        # Increment the value for the key representing the number of fact-checking neighbours
        # by 1 exposure
        # node_ind uses node_dict to retrieve the index for the node
        node_ind = self.node_dict[node]
        if FC_neighbours_count not in self.debunking_exposures[node_ind].keys():
            self.debunking_exposures[node_ind][FC_neighbours_count] = 1
        else:
            self.debunking_exposures[node_ind][FC_neighbours_count] += 1

        return copy.deepcopy(self.debunking_exposures[node_ind])

    def update_fake_news_exposures(self, node, node_neighbours):
        B_neighbours_count = 0
        for adj_node in node_neighbours:
            if self.node_status[self.node_dict[adj_node]] == 1:
                B_neighbours_count += 1

        # Increment the value for the key representing the number of believer neighbours
        # by 1 exposure
        # node_ind uses node_dict to retrieve the index for the node
        node_ind = self.node_dict[node]
        if B_neighbours_count not in self.fake_news_exposures[node_ind].keys():
            self.fake_news_exposures[node_ind][B_neighbours_count] = 1
        else:
            self.fake_news_exposures[node_ind][B_neighbours_count] += 1

        return copy.deepcopy(self.fake_news_exposures[node_ind])
    

    def init_nodes(self, seed_node, theta):
        # Initialising the initially infected nodes, fact-checking nodes and their statuses
        fact_checker_nodes = list()
        believer_nodes = list()
        seed_nodes = self.get_seed_nodes(seed_node[0])

        for sn in seed_nodes:
            # Add seed node in believer_nodes
            believer_nodes.append(sn)

            # Initial attempt to infect neighbours of the seed node
            neighbours = list(self.network.neighbors(sn))
            infected_neighbours = self.rng.choice(neighbours, int(np.ceil(len(neighbours) * theta)), replace=False)
            for inf_n in infected_neighbours:
                believer_nodes.append(inf_n)

        for sn in believer_nodes:
            # Update node status for initial believer nodes
            self.node_status[self.node_dict[sn]] = 1

        return fact_checker_nodes, believer_nodes


    def get_seed_nodes(self, seed_node):
        seed_nodes = list()
        for node in self.network.nodes():
            # Extracts the first item of the node tuple
            if tuple(map(int, node.split(',')))[0] == seed_node:
                seed_nodes.append(node)
        return seed_nodes

    def get_config(self):
        print("----------------\nTime-Aggregated Network Properties:\n----------------",
              "\nOrder, k:", self.K,
              "\nNumber of nodes:", self.node_count,
              "\nNumber of edges:", self.network.number_of_edges(),
              "\nTime observed:", self.T,
              "\n----------------")
        '''
        plt.subplot(111)
        nx.draw(self.network, with_labels=True, pos=nx.shell_layout(self.network))
        plt.title("Time-Aggregated Network")
        plt.show()'''

    def get_time_aggregated_network(self):
        return self.network

    def get_k_order_network(self):
        return self.k_order_network
    
    def init_node_dict(self):
        # Node dict maps all nodes to a corresponding index through enumeration
        node_dict = {}
        for index, node in enumerate(self.network.nodes()):
            node_dict[node] = index
        return node_dict




