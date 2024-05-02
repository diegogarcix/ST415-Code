import networkx as nx
import numpy as np
import copy
import matplotlib.pyplot as plt

"""
#### This class implements a neighbour exchange model [Volz and Meyers (2007)] for the generation of a temporal network
----------
Parameters:
----------
init_network: NetworkX.Graph
    The initial static network structure at t=0.
t_max: int
    The maximum time - the number of iterations for which the temporal network is generated. Default: 10
rho: float
    The success probability of a neighbour exchange occurring. Default: 0.95
seed: int
    seed for RNG. Default: None
----------
"""


class NeighbourExchangeTN:
    def __init__(self, init_network, t_max=10, rho=0.95, seed=None):
        self.init_network = init_network
        self.node_count = init_network.number_of_nodes()
        self.t_max = t_max + 1
        self._set_param(rho)
        self.rng = np.random.RandomState(seed)

    def get_temporal_network(self):
        snapshot = copy.deepcopy(self.init_network)
        temporal_network = [snapshot]

        '''
        # Visualisation of the initial network (at t=0)
        ax = plt.subplot(111)
        nx.draw(snapshot, with_labels=True, pos=nx.shell_layout(snapshot))
        plt.title("t = 0")
        plt.show()
        '''

        for t in range(1, self.t_max):
            # print("----------------\nt =", t)
            # Exchange neighbours at a rate rho
            if self.rng.binomial(1, self.rho):
                # print("Exchange Initiated!")
                # Randomly choose 2 unique nodes for edge swapping and retrieve their neighbours
                chosen_node_a, chosen_node_b = self.rng.choice(self.init_network.nodes(), 2, replace=False)
                choices_a = set(snapshot.neighbors(chosen_node_a))
                choices_b = set(snapshot.neighbors(chosen_node_b))

                # Choose neighbours of chosen nodes which are not neighbouring the other node to preserve their degree
                if len(choices_a - choices_b - {chosen_node_b}) > 0 and len(
                        choices_b - choices_a - {chosen_node_a}) > 0:
                    neighbour_a = self.rng.choice(list(choices_a - choices_b - {chosen_node_b}))
                    neighbour_b = self.rng.choice(list(choices_b - choices_a - {chosen_node_a}))
                else:
                    # Exchange does not occur if the degree of the chosen nodes cannot be preserved when swapping
                    # their edges. This case is increasingly unlikely as the density of the network increases.
                    # print("Exchange Not Possible.")
                    temporal_network.append(copy.deepcopy(snapshot))
                    continue

                # print("The edges between", chosen_node_a, "-", neighbour_a, "and", chosen_node_b, "-", neighbour_b, "will be swapped!")

                # Swap the edges between the chosen node and their chosen neighbour
                snapshot.remove_edge(chosen_node_a, neighbour_a)
                snapshot.add_edge(chosen_node_a, neighbour_b)

                snapshot.remove_edge(chosen_node_b, neighbour_b)
                snapshot.add_edge(chosen_node_b, neighbour_a)

            temporal_network.append(copy.deepcopy(snapshot))

            
            # This can be used to visualise the swapping of edges over the temporal network
            '''ax = plt.subplot(111)
            nx.draw(snapshot, with_labels=True, pos=nx.shell_layout(snapshot))
            plot_title = "t = "+str(t)
            plt.title(plot_title)
            plt.show()'''
            

        return temporal_network

    def _set_param(self, rho):
        if rho <= 0.0 or rho > 1.0:
            raise ValueError("rho must be in the interval (0,1].")
        else:
            self.rho = rho
