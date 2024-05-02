import numpy as np
import networkx as nx

class Distance_instance():
    def __init__(self, Statistics1, Statistics2, network):
        self.clusts = list(nx.clustering(network).values())
        self.S1 = Statistics1
        self.S2 = Statistics2
        self.network = network
        self.diff_infected = abs(Statistics1.no_infected - Statistics2.no_infected)
        self.diff_susceptible = abs(Statistics1.no_susceptible - Statistics2.no_susceptible)
        self.diff_factchecker = abs(Statistics1.no_factchecker - Statistics2.no_factchecker)
        self.diff_total = abs(Statistics1.total - Statistics2.total)
        self.all_length = nx.all_pairs_shortest_path_length(network)
        self.max_length = max(self.all_length)
        
        # METHOD NOT FINISHED BUT UNNECESSARY IN PRACTICE
    
class Distance_list():
    def __init__(self, Statistics1, Statistics2, network):
        self.S1 = Statistics1
        self.S2 = Statistics2
        self.no_snapshots = Statistics1.no_snapshots
        self.network = network
        self.all_length = dict(nx.all_pairs_shortest_path_length(network))
        self.max_length = max(self.all_length)
    

    def distance(self, data1, data2, snapshot_id):
        # Should compute for each time-step and sum over them
        result1 = pow((data1.no_infected[snapshot_id] - data2.no_infected[snapshot_id])/self.network.number_of_nodes(), 2)
        result2 = pow((data1.no_factchecker[snapshot_id] - data2.no_factchecker[snapshot_id])/self.network.number_of_nodes(), 2)
        result3 = self.local_distance(data1.subset_infected[snapshot_id], data2.subset_infected[snapshot_id], self.all_length)
        result4 = self.local_distance(data1.subset_factchecker[snapshot_id], data2.subset_factchecker[snapshot_id], self.all_length)
        #print('susbet fake exposed:', data1.subset_fake_news_exposed)
        result5 = self.local_distance(data1.subset_fake_news_exposed[snapshot_id], data2.subset_fake_news_exposed[snapshot_id], self.all_length)
        result6 = self.local_distance(data1.subset_debunking_exposed[snapshot_id], data2.subset_debunking_exposed[snapshot_id], self.all_length)
        result7 = pow((len(data1.subset_fake_news_exposed[snapshot_id]) - len(data2.subset_fake_news_exposed[snapshot_id]))/self.network.number_of_nodes(), 2)
        result8 = pow((len(data1.subset_debunking_exposed[snapshot_id]) - len(data2.subset_debunking_exposed[snapshot_id]))/self.network.number_of_nodes(), 2)
        

        result1 = np.sqrt(result1)
        result2 = np.sqrt(result2)
        result7 = np.sqrt(result7)
        result8 = np.sqrt(result8)
        result = result1 + result2 + result3 + result4 + result5 + result6 + result7 + result8
        #print('Dists', round(result1, 4), round(result3, 4), round(result5, 4), round(result6, 4), round(result7, 4), round(result8, 4))
        return (result)

    def local_distance(self, S1, S2, all_length):

        # Take the disjoint parts of these two subsets
        S1_dj = np.setdiff1d(S1, S2)
        #print('S1_dj:', S1_dj, '.')
        S2_dj = np.setdiff1d(S2, S1)
        #print('S2_dj:', S2_dj, '.')

        result = 0
        no_paths = 0
        for ind1 in range(len(S1_dj)):
            for ind2 in range(len(S2_dj)):
                #print('a=', S1_dj[ind1], '.')
                #print('b=', S2_dj[ind2], '.')
                a = S1_dj[ind1]
                b = S2_dj[ind2]
                #print(all_length)
                #print(all_length[a])
                #print(all_length[a][b])
                try:
                    c = all_length[a][b]
                    no_paths += 1
                except KeyError:
                    c = 0
                result += c / self.max_length
                #print('temp distance:', result)
        if len(S1_dj) != 0 and len(S2_dj) != 0 and no_paths != 0:
            #result = result / (len(S1_dj) * len(S2_dj))
            result = result / no_paths
        #print('final local distance:', result)
        elif len(S1_dj) * len(S2_dj) == 0 and len(S1_dj) + len(S2_dj) != 0:
            result = 1
        #print(all_length)
        #print('Local distance outputted', result)
        return (result)

class Distance_dict_list(Distance_list):
    def __init__(self, Statistics1, Statistics2, network):
        self.S1 = Statistics1
        self.S2 = Statistics2
        self.no_snapshots = Statistics1.no_snapshots
        self.network = network
        self.diff_infected, self.diff_susceptible, self.diff_factchecker, self.diff_total = [], [], [], []
        
        self.all_length = dict(nx.all_pairs_shortest_path_length(network))
        res = {key: max(val.values()) for key, val in self.all_length.items()}
        #print(res)
        self.max_length = max(res.values())
        #self.max_length = max(self.all_length)
    

class Distance_list_SIR():
    def __init__(self, Statistics1, Statistics2, network):
        self.S1 = Statistics1
        self.S2 = Statistics2
        self.no_snapshots = Statistics1.no_snapshots
        self.network = network
        self.all_length = dict(nx.all_pairs_shortest_path_length(network))
        self.max_length = max(self.all_length)
    
    def distance(self, data1, data2, snapshot_id):
        # Should compute for each time-step and sum over them
        result1 = pow((data1.no_infected[snapshot_id] - data2.no_infected[snapshot_id])/self.network.number_of_nodes(), 2)
        result2 = pow((data1.no_factchecker[snapshot_id] - data2.no_factchecker[snapshot_id])/self.network.number_of_nodes(), 2)
        result3 = self.local_distance(data1.subset_infected[snapshot_id], data2.subset_infected[snapshot_id], self.all_length)
        #print('susbet fake exposed:', data1.subset_fake_news_exposed)

        result1 = np.sqrt(result1)
        result2 = np.sqrt(result2)
        result = result1 + result2 + result3
        #print('Dists', round(result1, 4), round(result3, 4), round(result5, 4), round(result6, 4), round(result7, 4), round(result8, 4))
        return (result)

    def local_distance(self, S1, S2, all_length):

        # Take the disjoint parts of these two subsets
        S1_dj = np.setdiff1d(S1, S2)
        #print('S1_dj:', S1_dj, '.')
        S2_dj = np.setdiff1d(S2, S1)
        #print('S2_dj:', S2_dj, '.')

        result = 0
        no_paths = 0
        for ind1 in range(len(S1_dj)):
            for ind2 in range(len(S2_dj)):
                #print('a=', S1_dj[ind1], '.')
                #print('b=', S2_dj[ind2], '.')
                a = S1_dj[ind1]
                b = S2_dj[ind2]
                #print(all_length)
                #print(all_length[a])
                #print(all_length[a][b])
                try:
                    c = all_length[a][b]
                    no_paths += 1
                except KeyError:
                    c = 0
                result += c / self.max_length
                #print('temp distance:', result)
        if len(S1_dj) != 0 and len(S2_dj) != 0 and no_paths != 0:
            #result = result / (len(S1_dj) * len(S2_dj))
            result = result / no_paths
        #print('final local distance:', result)
        elif len(S1_dj) * len(S2_dj) == 0 and len(S1_dj) + len(S2_dj) != 0:
            result = 1
        #print(all_length)
        #print('Local distance outputted', result)
        return (result)

class Distance_dict_list_SIR(Distance_list_SIR):
    def __init__(self, Statistics1, Statistics2, network):
        self.S1 = Statistics1
        self.S2 = Statistics2
        self.no_snapshots = Statistics1.no_snapshots
        self.network = network
        self.diff_infected, self.diff_susceptible, self.diff_factchecker, self.diff_total = [], [], [], []
        
        self.all_length = dict(nx.all_pairs_shortest_path_length(network))
        res = {key: max(val.values()) for key, val in self.all_length.items()}
        #print(res)
        self.max_length = max(res.values())
        #self.max_length = max(self.all_length)