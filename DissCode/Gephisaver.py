import networkx as nx
import numpy as np







case, node_no, infection_node = 'er', 100, 10
A = np.load('Networks/' + case + '_' + str(node_no) + '_network.npy')
static_network = nx.from_numpy_array(A)
nx.write_gexf(static_network, "ER.gexf")

case, node_no, infection_node = 'ba', 100, 4
A = np.load('Networks/' + case + '_' + str(node_no) + '_network.npy')
static_network = nx.from_numpy_array(A)
nx.write_gexf(static_network, "Ba.gexf")

case, node_no, infection_node = 'inrv', 354, 70  ### 2 Different Networks
A = np.load('Networks/' + case + '_' + str(node_no) + '_network.npy')
static_network = nx.from_numpy_array(A)
nx.write_gexf(static_network, "INRV.gexf")

case, node_no, infection_node = 'fb', 4039, 2000
A = np.load('Networks/' + case + '_' + str(node_no) + '_network.npy')
static_network = nx.from_numpy_array(A)
nx.write_gexf(static_network, "FB.gexf")