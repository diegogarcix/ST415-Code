from typing import Any
import numpy as np
import networkx as nx
from operator import countOf

class Statistics_instance():
    def __init__(self, infection_array, fake_news_exposure_array, debunking_exposure_array, network):
        self.infection_array = infection_array
        self.no_susceptible = np.sum(infection_array == 0)
        self.no_infected = np.sum(infection_array == 1)
        self.no_factchecker = np.sum(infection_array == -1)
        self.total = np.sum(infection_array)
        self.n = len(infection_array)
        self.network = network

        subset_infected, subset_susceptible, subset_factchecker, subset_fake_news_exposed, subset_debunking_exposed = [], [], [], [], []
        for i in range(self.n):
            if infection_array[0][i] == 0.0:
                subset_susceptible.append(i)
                if fake_news_exposure_array[i] == 1:
                    subset_fake_news_exposed.append(i)
                if debunking_exposure_array[i] == 1:
                    subset_debunking_exposed.append(i)
            if infection_array[0][i] == 1.0:
                subset_infected.append(i)
                if debunking_exposure_array[i] == 1:
                    subset_debunking_exposed.append(i)
            if infection_array[0][i] == -1.0:
                subset_factchecker.append(i)

        

class Statistics_list():
    def __init__(self, infection_arrays, fake_news_exposure_arrays, debunking_exposure_arrays, network):
        self.infection_arrays = infection_arrays
        self.no_snapshots = len(infection_arrays)
        self.n = len(infection_arrays[0])

        self.no_infected, self.no_susceptible, self.no_factchecker, self.total = [], [], [], []
        for i in range(len(infection_arrays)):
            self.no_susceptible.append(np.sum(infection_arrays[i] == 0))
            self.no_infected.append(np.sum(infection_arrays[i] == 1))
            self.no_factchecker.append(np.sum(infection_arrays[i] == -1))
            #self.total.append(np.sum(infection_arrays[i]))

        self.network = network

        final_subset_susceptible, final_subset_infected, final_subset_factchecker = [], [], []
        final_subset_fake_news_exposed, final_subset_debunking_exposed = [], []
    
        for ss in range(self.no_snapshots):
            subset_infected, subset_susceptible, subset_factchecker = [], [], []
            subset_fake_news_exposed, subset_debunking_exposed = [], []
            for i in range(self.n):
                if infection_arrays[ss][i] == 0:
                    subset_susceptible.append(i)
                    if fake_news_exposure_arrays[ss][i] == 1:
                        subset_fake_news_exposed.append(i)
                    if debunking_exposure_arrays[ss][i] == 1:
                        subset_debunking_exposed.append(i)
                if infection_arrays[ss][i] == 1:
                    subset_infected.append(i)
                    if debunking_exposure_arrays[ss][i] == 1:
                        subset_debunking_exposed.append(i)
                if infection_arrays[ss][i] == -1:
                    subset_factchecker.append(i)

            '''print('Subset susceptible:', subset_susceptible,
                  '\nSubset infected:', subset_infected,
                  '\nSubset factchecker:', subset_factchecker)'''
            
            final_subset_susceptible.append(subset_susceptible)
            final_subset_infected.append(subset_infected)
            final_subset_factchecker.append(subset_factchecker)
            final_subset_fake_news_exposed.append(subset_fake_news_exposed)
            final_subset_debunking_exposed.append(subset_debunking_exposed)
        self.subset_susceptible, self.subset_infected, self.subset_factchecker = final_subset_susceptible, final_subset_infected, final_subset_factchecker
        self.subset_fake_news_exposed, self.subset_debunking_exposed = final_subset_fake_news_exposed, final_subset_debunking_exposed

    def statistics(self, data):
        return data

class Statistics_dict_list():
    def __init__(self, infection_dicts, fake_news_exposure_arrays, debunking_exposure_arrays, network):
        #print('fake news exp arrays:', fake_news_exposure_arrays)
        self.infection_arrays = infection_dicts
        self.no_snapshots = len(infection_dicts)
        self.n = len(infection_dicts[0])

        self.no_infected, self.no_susceptible, self.no_factchecker, self.total = [], [], [], []
        for i in range(len(infection_dicts)):
            self.no_susceptible.append(countOf(infection_dicts[i].values(), 0))
            self.no_infected.append(countOf(infection_dicts[i].values(), 0))
            self.no_factchecker.append(countOf(infection_dicts[i].values(), 0))

        self.network = network

        final_subset_susceptible, final_subset_infected, final_subset_factchecker = [], [], []
        final_subset_fake_news_exposed, final_subset_debunking_exposed = [], []
    
        for ss in range(self.no_snapshots):
            subset_infected, subset_susceptible, subset_factchecker = [], [], []
            subset_fake_news_exposed, subset_debunking_exposed = [], []
            for i in infection_dicts[ss]:
                if infection_dicts[ss][i] == 0:
                    subset_susceptible.append(i)
                    if fake_news_exposure_arrays[ss][i] == 1:
                        subset_fake_news_exposed.append(i)
                    if debunking_exposure_arrays[ss][i] == 1:
                        subset_debunking_exposed.append(i)
                if infection_dicts[ss][i] == 1:
                    subset_infected.append(i)
                    if debunking_exposure_arrays[ss][i] == 1:
                        subset_debunking_exposed.append(i)
                if infection_dicts[ss][i] == -1:
                    subset_factchecker.append(i)

            '''print('Subset susceptible:', subset_susceptible,
                  '\nSubset infected:', subset_infected,
                  '\nSubset factchecker:', subset_factchecker)'''
            
            final_subset_susceptible.append(subset_susceptible)
            final_subset_infected.append(subset_infected)
            final_subset_factchecker.append(subset_factchecker)
            final_subset_fake_news_exposed.append(subset_fake_news_exposed)
            final_subset_debunking_exposed.append(subset_debunking_exposed)
        self.subset_susceptible, self.subset_infected, self.subset_factchecker = final_subset_susceptible, final_subset_infected, final_subset_factchecker
        self.subset_fake_news_exposed, self.subset_debunking_exposed = final_subset_fake_news_exposed, final_subset_debunking_exposed

    def statistics(self, data):
        return data

            

class Statistics_list_SIR():
    def __init__(self, infection_arrays, network):
        self.infection_arrays = infection_arrays
        self.no_snapshots = len(infection_arrays)
        self.n = len(infection_arrays[0])

        self.no_infected, self.no_susceptible, self.no_factchecker, self.total = [], [], [], []
        for i in range(len(infection_arrays)):
            self.no_susceptible.append(np.sum(infection_arrays[i] == 0))
            self.no_infected.append(np.sum(infection_arrays[i] == 1))
            self.no_factchecker.append(np.sum(infection_arrays[i] == -1))
            self.total.append(np.sum(infection_arrays[i]))

        self.network = network

        final_subset_susceptible, final_subset_infected, final_subset_factchecker = [], [], []
    
        for ss in range(self.no_snapshots):
            subset_infected, subset_susceptible, subset_factchecker = [], [], []
            for i in range(self.n):
                if infection_arrays[ss][i] == 0:
                    subset_susceptible.append(i)
                if infection_arrays[ss][i] == 1:
                    subset_infected.append(i)
                if infection_arrays[ss][i] == -1:
                    subset_factchecker.append(i)

            '''print('Subset susceptible:', subset_susceptible,
                  '\nSubset infected:', subset_infected,
                  '\nSubset factchecker:', subset_factchecker)'''
            
            final_subset_susceptible.append(subset_susceptible)
            final_subset_infected.append(subset_infected)
            final_subset_factchecker.append(subset_factchecker)
        self.subset_susceptible, self.subset_infected, self.subset_factchecker = final_subset_susceptible, final_subset_infected, final_subset_factchecker
        
    def statistics(self, data):
        return data
    
class Statistics_dict_list_SIR():
    def __init__(self, infection_dicts, network):
        self.infection_arrays = infection_dicts
        self.no_snapshots = len(infection_dicts)
        self.n = len(infection_dicts[0])

        self.no_infected, self.no_susceptible, self.no_factchecker, self.total = [], [], [], []
        for i in range(len(infection_dicts)):
            self.no_susceptible.append(countOf(infection_dicts[i].values(), 0))
            self.no_infected.append(countOf(infection_dicts[i].values(), 0))
            self.no_factchecker.append(countOf(infection_dicts[i].values(), 0))

        self.network = network

        final_subset_susceptible, final_subset_infected, final_subset_factchecker = [], [], []
        
        for ss in range(self.no_snapshots):
            subset_infected, subset_susceptible, subset_factchecker = [], [], []
            for i in infection_dicts[ss]:
                if infection_dicts[ss][i] == 0:
                    subset_susceptible.append(i)
                if infection_dicts[ss][i] == 1:
                    subset_infected.append(i)
                if infection_dicts[ss][i] == -1:
                    subset_factchecker.append(i)

            '''print('Subset susceptible:', subset_susceptible,
                  '\nSubset infected:', subset_infected,
                  '\nSubset factchecker:', subset_factchecker)'''
            
            final_subset_susceptible.append(subset_susceptible)
            final_subset_infected.append(subset_infected)
            final_subset_factchecker.append(subset_factchecker)
        self.subset_susceptible, self.subset_infected, self.subset_factchecker = final_subset_susceptible, final_subset_infected, final_subset_factchecker
        
    def statistics(self, data):
        return data