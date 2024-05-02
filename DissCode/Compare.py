import numpy as np
import networkx as nx
from ST955.SBFCModel_TAN.Model import FakeNewsSBFC_TAN
from Stats import Statistics_list, Statistics_instance
from Dists import Distance_list, Distance_instance
from statistics import mean

def compare(model, parameters1, parameters2, no_simulations, network, k_order, no_snapshots):
        
    total_dist_per_simulation = []
    
    result1 = model.forward_simulate(parameters1, no_simulations)
    result2 = model.forward_simulate(parameters2, no_simulations)

    for simulation in range(no_simulations):
        result1_ss = np.array_split(result1[simulation], no_snapshots)
        #print('RESULT1_SS', result1_ss)
        susceptible1_ss, believers1_ss, fact_checkers1_ss = [], [], []
        for i in range(no_snapshots):
            susceptible1, believers1, fact_checkers1 = 0, 0, 0
            #print('RESULT1_SS[',i,']', result1_ss[i])
            for result in result1_ss[i]:
                if (result == 0): susceptible1 +=1
                if (result == 1): believers1 +=1
                if (result == -1): fact_checkers1 +=1
            susceptible1_ss.append(susceptible1)
            believers1_ss.append(believers1)
            fact_checkers1_ss.append(fact_checkers1)
            #print(fact_checkers1_ss)

        '''print('\n\n+++++++++++++ PERFORMED ON', k_order, 'ORDER TAN FOR', no_simulations,'SIMULATIONS +++++++++++++')
        print("# Nodes:", network.number_of_nodes())
        print("# Edges:", network.number_of_edges())
        print("\n----------------\nResults 1:", parameters1, "\n----------------")
        print("S:", susceptible1_ss, "\nB:", believers1_ss, "\nFC:", fact_checkers1_ss)'''

        

        result2_ss = np.array_split(result2[simulation], no_snapshots)
        #print('RESULT2_SS', result2_ss)
        susceptible2_ss, believers2_ss, fact_checkers2_ss = [], [], []
        for i in range(no_snapshots):
            susceptible2, believers2, fact_checkers2 = 0, 0, 0
            #print('RESULT2_SS[',i,']', result2_ss[i])
            for result in result2_ss[i]:
                if (result == 0): susceptible2 +=1
                if (result == 1): believers2 +=1
                if (result == -1): fact_checkers2 +=1
            susceptible2_ss.append(susceptible2)
            believers2_ss.append(believers2)
            fact_checkers2_ss.append(fact_checkers2)
            #print(fact_checkers1_ss)

        '''print('\n\n+++++++++++++ PERFORMED ON', k_order, 'ORDER TAN FOR', no_simulations,'SIMULATIONS +++++++++++++')
        print("# Nodes:", network.number_of_nodes())
        print("# Edges:", network.number_of_edges())
        print("\n----------------\nResults 2:", parameters2, "\n----------------")
        print("S:", susceptible2_ss, "\nB:", believers2_ss, "\nFC:", fact_checkers2_ss)'''
        

    #==============================================================================
    # Distances
        stats1 = Statistics_list(result1_ss, network)
        stats2 = Statistics_list(result2_ss, network)

        dists = Distance_list(stats1, stats2, network)
        distance_per_sim = []
        for snapshot_id in range(no_snapshots):
            distance_per_sim.append(dists.distance(stats1, stats2, snapshot_id))
        simulation_total_dist = round(np.sum(distance_per_sim), 4)
        total_dist_per_simulation.append(simulation_total_dist)
        print('Distances:', distance_per_sim,
        '\nTotal calculated distance is:', simulation_total_dist)
    
    print('Total dist per simulation:', total_dist_per_simulation,
          '\nAverage dist:', mean(total_dist_per_simulation))
        
    '''print("\n---------------- Mean Differences ----------------")
    print('Susceptible = ', mean(mean_dists.diff_susceptible),
        '\nInfected = ', mean(mean_dists.diff_infected),
        '\nFactchecker = ', mean(mean_dists.diff_factchecker),
        '\nTotal = ', mean(mean_dists.diff_total),
        '\nElementwise diff = ', mean(mean_dists.elementwise_diff()))

    diff_clust_infected, diff_clust_susceptible, diff_clust_factchecker = mean_dists.clustering_diff()
    print("\n----------- Mean Clustering Differences ----------")
    print('Clust Susceptible = ', round(mean(diff_clust_susceptible),3),
        '\nClust Infected = ', round(mean(diff_clust_infected),3),
        '\nClust Factchecker = ', round(mean(diff_clust_factchecker),3))'''

