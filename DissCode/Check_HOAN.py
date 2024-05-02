import numpy as np
import networkx as nx
from statistics import mean
import copy as copy
from TNGenerator import NeighbourExchangeTN
from SpreadingModels.SBFCModel_TN.Model import FakeNewsSBFC_TN
from SpreadingModels.SBFCModel.Model import FakeNewsSBFC
from SpreadingModels.SBFCModel_TAN.Model import FakeNewsSBFC_TAN
from SpreadingModels.SBFCModel_TAN.Model_diego import FakeNewsSBFC_TAN_diego
import matplotlib.pyplot as plt
from abcpy.continuousmodels import Uniform
from abcpy.discretemodels import DiscreteUniform


from Stats import Statistics_instance, Statistics_list, Statistics_dict_list
from Dists import Distance_instance, Distance_list, Distance_dict_list

import pathpy

from pathpy.classes import temporal_network, higher_order_network, network
from pathpy.path_extraction import temporal_paths
from pathpy.algorithms import centralities

# ==============================================================================
# Initialise parameters
theta = Uniform([[0.25], [0.75]], name='theta')
gamma = Uniform([[0.25], [0.75]], name='gamma')
delta = Uniform([[0.0001], [0.005]], name='delta')
seed_node = DiscreteUniform([[0], [4]], name='seed_node')
test_time = [1,2,3,4,5]

# ==============================================================================
# Create Temporal Network
G = nx.Graph()
G.add_nodes_from([0,1,2,3])
G1 = copy.deepcopy(G)
G2 = copy.deepcopy(G)
G3 = copy.deepcopy(G)
G4 = copy.deepcopy(G)
G5 = copy.deepcopy(G)
G1.add_edges_from([])
G2.add_edges_from([(0, 2), (0, 3), (2, 3)])
G3.add_edges_from([(0, 1), (0, 3), (1, 2), (1, 3)])
G4.add_edges_from([(0, 1), (0, 2), (1, 3)])
G5.add_edges_from([(0, 2), (1, 3)])

temp_network = []
temp_network.append(G1)
temp_network.append(G2)
temp_network.append(G3)
temp_network.append(G4)
temp_network.append(G5)

'''example_SBFC_TN = FakeNewsSBFC_TN([theta, gamma, delta, seed_node], temp_network, test_time, seed= 12)
results = example_SBFC_TN.forward_simulate([.3, .7, .005, 0], 1)
res = np.array_split(results[0], len(test_time))
for i in range(len(test_time)):
    print(res[i])'''

example_SBFC_TAN = FakeNewsSBFC_TAN_diego([theta, gamma, delta, seed_node], temp_network, test_time, k_order = 2)
results = example_SBFC_TAN.forward_simulate([.3, .7, .005, 0], 1)
print(example_SBFC_TAN.get_time_aggregated_network())
tan_network = example_SBFC_TAN.get_time_aggregated_network()


print(example_SBFC_TAN.get_k_order_network())
res = np.array_split(results[0], len(test_time))
HO = []
for i in range(len(test_time)):
    #print('Infection array at time t=',i+1,':',res[i])
    HO.append(example_SBFC_TAN.colour_TAN_all(res[i]))
    #print('HOAN:', HO)


results2 = example_SBFC_TAN.forward_simulate([.7, .3, .0001, 0], 1)
res2 = np.array_split(results2[0], len(test_time))
HO2 = []
for i in range(len(test_time)):
    #print('Infection array at time t=',i+1,':',res2[i])
    HO2.append(example_SBFC_TAN.colour_TAN_all(res2[i]))
    #print('HOAN:', HO2)


#HO = example_SBFC_TAN.colour_TAN_all([1, -1, -1, 0])
#print('Infection array: [1, -1, -1, 0]')
#print('HOAN:', HO)
a = Statistics_dict_list(HO, tan_network)
b= Statistics_dict_list(HO2, tan_network)
print(type(a))
dis = Distance_dict_list(a, b, tan_network)

print(dis.distance(a,b, 1))


'''example_SBFC_TN = FakeNewsSBFC_TN([theta, gamma, delta, seed_node], temp_network, test_time, k_order = 1)
example_SBFC_TN.get_k_order_network()
example_SBFC_TN.get_time_aggregated_network()
example_SBFC_TN.get_config()
results = example_SBFC_TN.forward_simulate([.3, .7, .005, 1], 1)
print(np.array_split(results[0], len(test_time)))

example_SBFC_TN = FakeNewsSBFC_TAN([theta, gamma, delta, seed_node], temp_network, test_time, k_order = 2)
example_SBFC_TN.get_k_order_network()
example_SBFC_TN.get_time_aggregated_network()
example_SBFC_TN.get_config()
results = example_SBFC_TN.forward_simulate([.3, .7, .005, 0], 1)
print(np.array_split(results[0], len(test_time)))

example_SBFC_TN = FakeNewsSBFC_TAN([theta, gamma, delta, seed_node], temp_network, test_time, k_order = 3)
example_SBFC_TN.get_k_order_network()
example_SBFC_TN.get_time_aggregated_network()
example_SBFC_TN.get_config()
results = example_SBFC_TN.forward_simulate([.3, .7, .005, 0], 1)
print(np.array_split(results, len(test_time)))'''