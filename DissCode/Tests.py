import numpy as np
import networkx as nx
from statistics import mean, median
import copy as copy
import pathpy
from datetime import datetime
from scipy.optimize import dual_annealing

from TNGenerator import NeighbourExchangeTN
from SpreadingModels.SBFCModel_TN.Model import FakeNewsSBFC_TN
from SpreadingModels.SBFCModel.Model import FakeNewsSBFC
from SpreadingModels.SIRModel.Model import FakeNewsSIR
from SpreadingModels.SIRModel_TN.Model import FakeNewsSIR_TN
from SpreadingModels.SIRModel_TAN.Model import FakeNewsSIR_TAN
from SpreadingModels.SBFCModel_TAN.Model import FakeNewsSBFC_TAN
from SpreadingModels.SBFCModel_TAN.Model_diego import FakeNewsSBFC_TAN_diego

from Stats import Statistics_instance, Statistics_list, Statistics_dict_list, Statistics_list_SIR, Statistics_dict_list_SIR
from Dists import Distance_instance, Distance_list, Distance_dict_list, Distance_list_SIR, Distance_dict_list_SIR

from abcpy.continuousmodels import Uniform
from abcpy.discretemodels import DiscreteUniform


# TEST Configurations
# ==============================================================================

# Different types of network (BA: Barabasi-Albert, ER: Erdos-Renyi, FB: Facebook Social Network,
# INRV: Indian Village contact Network) with node_no many nodes on the network. The infection_node
# is the true seed-node. (Choose one of the options)
# ==============================================================================
case, node_no, infection_node = 'ba', 100, 4
#case, node_no, infection_node = 'er', 100, 10
#case, node_no, infection_node = 'inrv', 354, 70  ### 2 Different Networks
#case, node_no, infection_node = 'fb', 4039, 2000

# ==============================================================================
# Time observed
no_snapshots = 20
start_test_time = 1
end_test_time = 151
steps = int((end_test_time - start_test_time)/no_snapshots)
test_time = np.arange(start_test_time, end_test_time, steps)
print('Test time:', test_time)

# ==============================================================================
# Time Stamp
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

# ==============================================================================
# Load network
# Pre-existing empirical network:
#A = np.load('Results/'+case+'_'+str(node_no)+'_yobs_'+str(0)+'.npy')
A = np.load('Networks/' + case + '_' + str(node_no) + '_network.npy')
#A = np.load('Networks/er_100__0_2_order_TAN.npy')

# NetworkX generator:
# random_graph = nx.generators.random_graphs.barabasi_albert_graph(15, 2)
random_graph_BA = nx.generators.random_graphs.barabasi_albert_graph(800, 5, seed=5555)
# random_graph_ER = nx.generators.random_graphs.erdos_renyi_graph(500, 20)

#example = random_graph_BA
static_network = nx.from_numpy_array(A)

# Temporal Network using Neighbour Exchange Model
# NE_TN = NeighbourExchangeTN(random_graph, 15, rho=1.0)
# BA_TN = NeighbourExchangeTN(random_graph_BA, 7, rho=1.0, seed=5555)
# ER_TN = NeighbourExchangeTN(random_graph_ER, 7, rho=1.0)
temp_net = NeighbourExchangeTN(static_network, t_max=7, rho = 1.0)

# print(example.nodes())
# temporal_net = NE_TN.get_temporal_network()
# temporal_net_BA = BA_TN.get_temporal_network()
# temporal_net_ER = ER_TN.get_temporal_network()
temporal_net = temp_net.get_temporal_network()

# ==============================================================================
# Initialise parameters
theta = Uniform([[0.25], [0.75]], name='theta')
gamma = Uniform([[0.25], [0.75]], name='gamma')
delta = Uniform([[0.0001], [0.005]], name='delta')
seed_node = DiscreteUniform([[0], [node_no]], name='seed_node')


# ==============================================================================
# TEST SCENARIOS
# ==============================================================================

# ==============================================================================
# OPTIMIZING FUNCTIONS
# ==============================================================================

def function_SIR(parameters):
        parameters[0], parameters[1] = round(parameters[0], 3), round(parameters[1], 3)
        parameters = list(parameters)
        parameters.append(1)
        global count
        count += 1
        #print(count, 'Sim: ', parameters[0], parameters[1], parameters[2])
        
        sim_results = model.forward_simulate(parameters, 1)
        split_sim_array = np.array_split(sim_results[0], len(test_time))
        stats2 = Statistics_list_SIR(split_sim_array, static_network)

        dists = Distance_list_SIR(stats1, stats2, static_network)
        distance_per_sim = []
        for snapshot_id in range(len(test_time)):
                distance_per_sim.append(dists.distance(stats1, stats2, snapshot_id))
        simulation_total_dist = np.sum(distance_per_sim)
        #print('Distances:', distance_per_sim,
        #'\nTotal calculated distance is:', simulation_total_dist)
        return(simulation_total_dist)

def function_SBFC(parameters):
        parameters[0], parameters[1], parameters[2] = round(parameters[0], 3), round(parameters[1], 3), round(parameters[2], 3)
        parameters = list(parameters)
        parameters.append(1)
        global count
        count += 1
        
        sim_results = model.forward_simulate(parameters, 1)
        split_sim_array = np.array_split(sim_results[0], len(test_time))
        #print(split_sim_array)
        split_fake_news_exposures = np.array_split(model.fake_news_exposed_observed_times, len(test_time))
        split_debunking_exposures = np.array_split(model.debunking_exposed_observed_times, len(test_time))
        stats2 = Statistics_list(split_sim_array, split_fake_news_exposures, split_debunking_exposures, static_network)

        dists = Distance_list(stats1, stats2, static_network)
        distance_per_sim = []
        for snapshot_id in range(len(test_time)):
                distance_per_sim.append(dists.distance(stats1, stats2, snapshot_id))
        simulation_total_dist = np.sum(distance_per_sim)
        #print('Distances:', distance_per_sim, '\nTotal calculated distance is:', simulation_total_dist)
        return(simulation_total_dist)

def function_SBFC_TN(parameters):
        parameters[0], parameters[1], parameters[2] = round(parameters[0], 3), round(parameters[1], 3), round(parameters[2], 3)
        parameters = list(parameters)
        parameters.append(1)
        global count
        count += 1
        #print('Sim: ', parameters[0], parameters[1], parameters[2], parameters[3])
        #print(count, 'Sim: ', parameters[0], parameters[1], parameters[2], parameters[3])
        sim_results = model.forward_simulate(parameters, 1)
        split_sim_array = np.array_split(sim_results[0], len(test_time))
        split_fake_news_exposures = np.array_split(model.fake_news_exposed_observed_times, len(test_time))
        split_debunking_exposures = np.array_split(model.debunking_exposed_observed_times, len(test_time))
        stats2 = Statistics_list(split_sim_array, split_fake_news_exposures, split_debunking_exposures, static_network)

        dists = Distance_list(stats1, stats2, static_network)
        distance_per_sim = []
        for snapshot_id in range(len(test_time)):
                distance_per_sim.append(dists.distance(stats1, stats2, snapshot_id))
        simulation_total_dist = np.sum(distance_per_sim)
        #print('Distances:', distance_per_sim,
        #'\nTotal calculated distance is:', simulation_total_dist)
        return(simulation_total_dist)

def function_SBFC_HOAN(parameters):
        parameters[0], parameters[1], parameters[2] = round(parameters[0], 3), round(parameters[1], 3), round(parameters[2], 3)
        parameters = list(parameters)
        parameters.append(1)
        global count
        count += 1

        #print('Sim: ', parameters[0], parameters[1], parameters[2], parameters[3])
        #print(count, 'Sim: ', parameters[0], parameters[1], parameters[2], parameters[3])
        sim_results = model.forward_simulate(parameters, 1)
        sim_results = np.array_split(sim_results[0], len(test_time))
        split_fake_news_exposures = np.array_split(model.fake_news_exposed_observed_times, len(test_time))
        split_debunking_exposures = np.array_split(model.debunking_exposed_observed_times, len(test_time))
        

        HO_sim_list = []
        HO_FN_exposure_list = []
        HO_D_exposure_list = []
        for i in range(len(test_time)):
                #print('Infection array at time t=',i+1,':', sim_results[i])
                HO_sim_list.append(model.colour_TAN_all(sim_results[i]))
                HO_FN_exposure_list.append(model.colour_TAN_fake_news_exposures(sim_results[i], split_fake_news_exposures[i]))
                HO_D_exposure_list.append(model.colour_TAN_debunking_exposures(sim_results[i], split_debunking_exposures[i]))
                #print('HOAN:', HO_sim_list[i])

        stats2 = Statistics_dict_list(HO_sim_list, HO_FN_exposure_list, HO_D_exposure_list, k_order_network)
        dists = Distance_dict_list(stats1, stats2, k_order_network)
        #print(dists.max_length)
        distance_per_sim = []
        for snapshot_id in range(len(test_time)):
                distance_per_sim.append(dists.distance(stats1, stats2, snapshot_id))
        simulation_total_dist = np.sum(distance_per_sim)
        #print('Distances:', distance_per_sim,
        #'\nTotal calculated distance is:', simulation_total_dist)
        return(simulation_total_dist)

def function_SIR_HOAN(parameters):
        parameters[0], parameters[1]= round(parameters[0], 3), round(parameters[1], 3)
        parameters = list(parameters)
        parameters.append(1)
        global count
        count += 1

        #print('Sim: ', parameters[0], parameters[1], parameters[2], parameters[3])
        #print(count, 'Sim: ', parameters[0], parameters[1], parameters[2], parameters[3])
        sim_results = model.forward_simulate(parameters, 1)
        sim_results = np.array_split(sim_results[0], len(test_time))
        
        HO_sim_list = []
        for i in range(len(test_time)):
                #print('Infection array at time t=',i+1,':', sim_results[i])
                HO_sim_list.append(model.colour_TAN_all(sim_results[i]))
                #print('HOAN:', HO_sim_list[i])

        stats2 = Statistics_dict_list_SIR(HO_sim_list, k_order_network)
        dists = Distance_dict_list_SIR(stats1, stats2, k_order_network)
        #print(dists.max_length)
        distance_per_sim = []
        for snapshot_id in range(len(test_time)):
                distance_per_sim.append(dists.distance(stats1, stats2, snapshot_id))
        simulation_total_dist = np.sum(distance_per_sim)
        #print('Distances:', distance_per_sim,
        #'\nTotal calculated distance is:', simulation_total_dist)
        return(simulation_total_dist)



# ==============================================================================
# INFERENCE
# ==============================================================================


# ==============================================================================
# SIR
# ==============================================================================

'''
model = FakeNewsSIR_TN([theta, gamma, seed_node], temporal_net, test_time)

obs_array = model.forward_simulate([.4, .2, 1], 1)
split_observed_array = np.array_split(obs_array[0], len(test_time))
stats1 = Statistics_list_SIR(split_observed_array, static_network)

lw, up = [0.25, 0.001 ], [0.75, 0.3]
print(list(zip(lw, up)))
print('Params: .4 0.2 1')
for i in range(10):
        # Time Stamp
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        #k_order_network = model.get_time_aggregated_network()
        count = 0
        ret = dual_annealing(function_SIR, bounds=list(zip(lw, up)), maxiter = 500, initial_temp = 1000)
        print('Count:', count, 'ret.x:', ret.x, 'ret.fun', ret.fun)'''

# ==============================================================================
# SBFC
# ==============================================================================

# model = FakeNewsSBFC([theta, gamma, delta, seed_node], static_network, test_time)
# model = FakeNewsSBFC_TN([theta, gamma, delta, seed_node], temporal_net, test_time)

'''obs_array = model.forward_simulate([.7, .25, .005, 1], 1)
split_observed_array = np.array_split(obs_array[0], len(test_time))
split_fake_news_exposures = np.array_split(model.fake_news_exposed_observed_times, len(test_time))
split_debunking_exposures = np.array_split(model.debunking_exposed_observed_times, len(test_time))
stats1 = Statistics_list(split_observed_array, split_fake_news_exposures, split_debunking_exposures, temporal_net)

lw, up = [0.25, 0.25, 0.001 ], [0.75, 0.75, 0.3]
print(list(zip(lw, up)))
print('Params: .7 .25 0.005 1')
for i in range(10):
        # Time Stamp
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        #k_order_network = model.get_time_aggregated_network()
        count = 0
        ret = dual_annealing(function_SBFC_TN, bounds=list(zip(lw, up)), maxiter = 500, initial_temp = 1000)
        print('Count:', count, 'ret.x:', ret.x, 'ret.fun', ret.fun)
'''

# ==============================================================================
# HOAN SBFC
# ==============================================================================

# model = FakeNewsSBFC_TAN_diego([theta, gamma, delta, seed_node], temporal_net, test_time, k_order = 2)
# model = FakeNewsSBFC_TAN_diego([theta, gamma, delta, seed_node], temporal_net, test_time, k_order = 3)
'''
k_order_network = model.time_aggregated_network
obs_array = model.forward_simulate([.3, .7, 0.001, 1], 1)
split_observed_array = np.array_split(obs_array[0], len(test_time))
split_fake_news_exposures = np.array_split(model.fake_news_exposed_observed_times, len(test_time))
split_debunking_exposures = np.array_split(model.debunking_exposed_observed_times, len(test_time))
        
#print(model.fake_news_exposed_observed_times)
HO_sim_list = []
HO_FN_exposure_list = []
HO_D_exposure_list = []
for i in range(len(test_time)):
        #print('Infection array at time t=',i+1,':', sim_results[i])
        HO_sim_list.append(model.colour_TAN_all(split_observed_array[i]))
        HO_FN_exposure_list.append(model.colour_TAN_fake_news_exposures(split_observed_array[i], split_fake_news_exposures[i]))
        HO_D_exposure_list.append(model.colour_TAN_debunking_exposures(split_observed_array[i], split_debunking_exposures[i]))
        #print('HOAN:', HO_sim_list[i])

stats1 = Statistics_dict_list(HO_sim_list, HO_FN_exposure_list, HO_D_exposure_list, k_order_network)
#print(stats1.subset_fake_news_exposed)

lw, up = [0.25, 0.25, 0.001 ], [0.75, 0.75, 0.3]
print(list(zip(lw, up)))

print('Params: .3 .7 0.001 1')
for i in range(10):
        # Time Stamp
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        #k_order_network = model.get_time_aggregated_network()
        count = 0
        ret = dual_annealing(function_SBFC_HOAN, bounds=list(zip(lw, up)), maxiter = 500, initial_temp = 1000)
        print('Count:', count, 'ret.x:', ret.x, 'ret.fun', ret.fun)
'''

# ==============================================================================
# HOAN SIR
# ==============================================================================

'''model = FakeNewsSIR_TAN([theta, delta, seed_node], temporal_net, test_time, k_order = 2)
# model = FakeNewsSIR_TAN([theta, gamma, delta, seed_node], temporal_net, test_time, k_order = 3)

k_order_network = model.time_aggregated_network
obs_array = model.forward_simulate([.7, .2, 1], 1)
split_observed_array = np.array_split(obs_array[0], len(test_time))
        
HO_sim_list = []
for i in range(len(test_time)):
        #print('Infection array at time t=',i+1,':', sim_results[i])
        HO_sim_list.append(model.colour_TAN_all(split_observed_array[i]))
        #print('HOAN:', HO_sim_list[i])

stats1 = Statistics_dict_list_SIR(HO_sim_list, k_order_network)

lw, up = [0.25, 0.001], [0.75, 0.3]
print(list(zip(lw, up)))

print('Params: .7 0.2 1')
for i in range(10):
        # Time Stamp
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        #k_order_network = model.get_time_aggregated_network()
        count = 0
        ret = dual_annealing(function_SIR_HOAN, bounds=list(zip(lw, up)), maxiter = 500, initial_temp = 1000)
        print('Count:', count, 'ret.x:', ret.x, 'ret.fun', ret.fun)


k_order_network = model.time_aggregated_network
obs_array = model.forward_simulate([.6, .001, 1], 1)
split_observed_array = np.array_split(obs_array[0], len(test_time))
        
HO_sim_list = []
for i in range(len(test_time)):
        #print('Infection array at time t=',i+1,':', sim_results[i])
        HO_sim_list.append(model.colour_TAN_all(split_observed_array[i]))
        #print('HOAN:', HO_sim_list[i])

stats1 = Statistics_dict_list_SIR(HO_sim_list, k_order_network)

lw, up = [0.25, 0.001], [0.75, 0.3]
print(list(zip(lw, up)))

print('Params: .6 0.001 1')
for i in range(10):
        # Time Stamp
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        #k_order_network = model.get_time_aggregated_network()
        count = 0
        ret = dual_annealing(function_SIR_HOAN, bounds=list(zip(lw, up)), maxiter = 500, initial_temp = 1000)
        print('Count:', count, 'ret.x:', ret.x, 'ret.fun', ret.fun)

'''
# ==============================================================================
# TESTING DIFFERENCES
# ==============================================================================


now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
model = FakeNewsSBFC([theta, gamma, delta, seed_node], static_network, test_time)
dist_id = []
dist_inferred = []

for i in range(100):
        obs_array = model.forward_simulate([.3, .7, .001, 1], 1)
        split_observed_array = np.array_split(obs_array[0], len(test_time))
        split_fake_news_exposures = np.array_split(model.fake_news_exposed_observed_times, len(test_time))
        split_debunking_exposures = np.array_split(model.debunking_exposed_observed_times, len(test_time))
        stats1 = Statistics_list(split_observed_array, split_fake_news_exposures, split_debunking_exposures, temporal_net)

        obs_array = model.forward_simulate([.3, .7, .001, 1], 1)
        split_observed_array = np.array_split(obs_array[0], len(test_time))
        split_fake_news_exposures = np.array_split(model.fake_news_exposed_observed_times, len(test_time))
        split_debunking_exposures = np.array_split(model.debunking_exposed_observed_times, len(test_time))
        stats2 = Statistics_list(split_observed_array, split_fake_news_exposures, split_debunking_exposures, temporal_net)
        dists = Distance_list(stats1, stats2, static_network)
        dist_sim = []
        for snapshot_id in range(len(test_time)):
                dist_sim.append(dists.distance(stats1, stats2, snapshot_id))
        dist_id.append(np.sum(dist_sim))

        obs_array = model.forward_simulate([.3925, .4945, .0035, 1], 1)
        split_observed_array = np.array_split(obs_array[0], len(test_time))
        split_fake_news_exposures = np.array_split(model.fake_news_exposed_observed_times, len(test_time))
        split_debunking_exposures = np.array_split(model.debunking_exposed_observed_times, len(test_time))
        stats2 = Statistics_list(split_observed_array, split_fake_news_exposures, split_debunking_exposures, temporal_net)
        dists = Distance_list(stats1, stats2, static_network)
        dist_sim = []
        for snapshot_id in range(len(test_time)):
                dist_sim.append(dists.distance(stats1, stats2, snapshot_id))
        dist_inferred.append(np.sum(dist_sim))

print('id', mean(dist_id), median(dist_id))
print('inferred', mean(dist_inferred), median(dist_inferred))

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


dist_id = []
dist_inferred = []

for i in range(100):
        obs_array = model.forward_simulate([.3, .7, .05, 1], 1)
        split_observed_array = np.array_split(obs_array[0], len(test_time))
        split_fake_news_exposures = np.array_split(model.fake_news_exposed_observed_times, len(test_time))
        split_debunking_exposures = np.array_split(model.debunking_exposed_observed_times, len(test_time))
        stats1 = Statistics_list(split_observed_array, split_fake_news_exposures, split_debunking_exposures, temporal_net)

        obs_array = model.forward_simulate([.3, .7, .05, 1], 1)
        split_observed_array = np.array_split(obs_array[0], len(test_time))
        split_fake_news_exposures = np.array_split(model.fake_news_exposed_observed_times, len(test_time))
        split_debunking_exposures = np.array_split(model.debunking_exposed_observed_times, len(test_time))
        stats2 = Statistics_list(split_observed_array, split_fake_news_exposures, split_debunking_exposures, temporal_net)
        dists = Distance_list(stats1, stats2, static_network)
        dist_sim = []
        for snapshot_id in range(len(test_time)):
                dist_sim.append(dists.distance(stats1, stats2, snapshot_id))
        dist_id.append(np.sum(dist_sim))

        obs_array = model.forward_simulate([.3925, .4945, .044, 1], 1)
        split_observed_array = np.array_split(obs_array[0], len(test_time))
        split_fake_news_exposures = np.array_split(model.fake_news_exposed_observed_times, len(test_time))
        split_debunking_exposures = np.array_split(model.debunking_exposed_observed_times, len(test_time))
        stats2 = Statistics_list(split_observed_array, split_fake_news_exposures, split_debunking_exposures, temporal_net)
        dists = Distance_list(stats1, stats2, static_network)
        dist_sim = []
        for snapshot_id in range(len(test_time)):
                dist_sim.append(dists.distance(stats1, stats2, snapshot_id))
        dist_inferred.append(np.sum(dist_sim))

print('id', mean(dist_id), median(dist_id))
print('inferred', mean(dist_inferred), median(dist_inferred))

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
