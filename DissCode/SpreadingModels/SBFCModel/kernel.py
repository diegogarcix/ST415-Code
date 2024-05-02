from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.special import gamma
from scipy.stats import multivariate_normal

from abcpy.probabilisticmodels import Continuous
from abcpy.perturbationkernel import PerturbationKernel, DiscreteKernel

class NetworkRandomWalkKernelInverseDegree(PerturbationKernel, DiscreteKernel):
    def __init__(self, models, network):
        """
        This class defines a kernel perturbing discrete parameters on a provided network with moves inversely proportional to degree of a node.
        Parameters
        ----------
        models: list
            List of abcpy.ProbabilisticModel objects
        network: A network
            Networkx object
        """

        self.models = models
        self.network = network

    def update(self, accepted_parameters_manager, kernel_index, row_index, rng=np.random.RandomState()):
        """
        Updates the parameter values contained in the accepted_paramters_manager.
        Parameters
        ----------
        accepted_parameters_manager: abcpy.AcceptedParametersManager object
            Defines the AcceptedParametersManager to be used.
        kernel_index: integer
            The index of the kernel in the list of kernels in the joint kernel.
        row_index: integer
            The index of the row that should be considered from the accepted_parameters_bds matrix.
        rng: random number generator
            The random number generator to be used.
        Returns
        -------
        np.ndarray
            The perturbed parameter values.
        """
        # Get all parameters relevant to this kernel
        discrete_model_values = accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index][row_index]

        if isinstance(discrete_model_values[0],
                      (np.float, np.float32, np.float64, np.int, np.int32, np.int64)):
            # Perturb
            discrete_model_values = np.array(discrete_model_values)
            perturbed_discrete_values = []
            # Implement a random walk for the discrete parameter values
            for discrete_value in discrete_model_values:
                nodes_proposed = list(self.network.neighbors(discrete_value))
                weight = np.zeros(shape=(len(nodes_proposed),))
                for ind in range(len(nodes_proposed)):
                    weight[ind] = 1 / self.network.degree(nodes_proposed[ind])
                weight = weight / sum(weight)
                perturbed_discrete_values.append(np.random.choice(nodes_proposed, 1, p=weight)[0])
            perturbed_discrete_values = np.array(perturbed_discrete_values)
        else:
            # Learn the structure
            struct = [[] for i in range(len(discrete_model_values))]
            for i in range(len(discrete_model_values)):
                struct[i] = discrete_model_values[i].shape[0]
            struct = np.array(struct).cumsum()
            discrete_model_values = np.concatenate(discrete_model_values)

            # Perturb
            discrete_model_values = np.array(discrete_model_values)
            perturbed_discrete_values = []
            # Implement a random walk for the discrete parameter values
            for discrete_value in discrete_model_values:
                nodes_proposed = list(self.network.neighbors(discrete_value))
                weight = np.zeros(shape=(len(nodes_proposed),))
                for ind in range(len(nodes_proposed)):
                    weight[ind] = 1 / self.network.degree(nodes_proposed[ind])
                weight = weight / sum(weight)
                perturbed_discrete_values.append(np.random.choice(nodes_proposed, 1, p=weight)[0])
            perturbed_discrete_values = np.array(perturbed_discrete_values)
            # Perturbed values anc split according to the structure
            perturbed_discrete_values = np.split(perturbed_discrete_values, struct)[:-1]

        return perturbed_discrete_values

    def calculate_cov(self, accepted_parameters_manager, kernel_index):
        """
        Calculates the covariance matrix of this kernel. Since there is no covariance matrix associated with this
        random walk, it returns an empty list.
        """

        return np.array([0]).reshape(-1, )

    def pmf(self, accepted_parameters_manager, kernel_index, mean, x):
        """Calculates the pdf of the kernel.
        Commonly used to calculate weights.
        Parameters
        ----------
        accepted_parameters_manager: abcpy.AcceptedParametersManager object
            The AcceptedParametersManager to be used.
        kernel_index: integer
            The index of the kernel in the list of kernels in the joint kernel.
        mean: np array, np.float or np.integer
            The reference point of the kernel
        x: The point at which the pdf should be evaluated.
        Returns
        -------
        float
            The pdf evaluated at point x.
        """
        density = 1
        if isinstance(mean[0],
                      (np.float, np.float32, np.float64, np.int, np.int32, np.int64)):
            mean = np.array(mean).astype(int)
            for ind1 in range(len(mean)):
                discrete_value = mean[ind1]
                nodes_proposed = list(self.network.neighbors(discrete_value))
                weight = np.zeros(shape=(len(nodes_proposed),))
                for ind2 in range(len(nodes_proposed)):
                    weight[ind2] = 1 / self.network.degree(nodes_proposed[ind2])
                weight = weight / sum(weight)
                if x[ind1] in nodes_proposed:
                    density = density * weight[np.where(nodes_proposed == x[ind1])[0]][0]
                else:
                    density = density * 0
            return density
        else:
            mean = np.array(np.concatenate(mean)).astype(int)
            for ind1 in range(len(mean)):
                discrete_value = mean[ind1]
                nodes_proposed = list(self.network.neighbors(discrete_value))
                weight = np.zeros(shape=(len(nodes_proposed),))
                for ind2 in range(len(nodes_proposed)):
                    weight[ind2] = 1 / self.network.degree(nodes_proposed[ind2])
                weight = weight / sum(weight)
                if x[ind1] in nodes_proposed:
                    density = density * weight[np.where(nodes_proposed == x[ind1])[0]][0]
                else:
                    density = density * 0
            return density



class NetworkRandomWalkKernelEdge(PerturbationKernel, DiscreteKernel):
    def __init__(self, models, network, name_weight):
        """
        This class defines a kernel perturbing discrete parameters on a provided network with moves proportional to an attribute of the edegs of the network.
        Parameters
        ----------
        models: list
            List of abcpy.ProbabilisticModel objects
        network: A network
            Networkx object
        name_weight: string
            name of the attribute of the network to be used as probability
        """

        self.models = models
        self.network = network
        self.name_weight = name_weight

    def update(self, accepted_parameters_manager, kernel_index, row_index, rng=np.random.RandomState()):
        """
        Updates the parameter values contained in the accepted_paramters_manager.
        Parameters
        ----------
        accepted_parameters_manager: abcpy.AcceptedParametersManager object
            Defines the AcceptedParametersManager to be used.
        kernel_index: integer
            The index of the kernel in the list of kernels in the joint kernel.
        row_index: integer
            The index of the row that should be considered from the accepted_parameters_bds matrix.
        rng: random number generator
            The random number generator to be used.
        Returns
        -------
        np.ndarray
            The perturbed parameter values.
        """
        # Get all parameters relevant to this kernel
        discrete_model_values = accepted_parameters_manager.kernel_parameters_bds.value()[kernel_index][row_index]

        if isinstance(discrete_model_values[0],
                      (np.float, np.float32, np.float64, np.int, np.int32, np.int64)):
            # Perturb
            discrete_model_values = np.array(discrete_model_values)
            perturbed_discrete_values = []
            # Implement a random walk for the discrete parameter values
            for discrete_value in discrete_model_values:
                nodes_proposed = list(self.network.neighbors(discrete_value))
                weight = np.zeros(shape=(len(nodes_proposed),))
                for ind in range(len(nodes_proposed)):
                    weight[ind] = self.network[discrete_value][nodes_proposed[ind]][self.name_weight]
                weight = weight / sum(weight)
                perturbed_discrete_values.append(np.random.choice(nodes_proposed, 1, p=weight)[0])
            perturbed_discrete_values = np.array(perturbed_discrete_values)
        else:
            # Learn the structure
            struct = [[] for i in range(len(discrete_model_values))]
            for i in range(len(discrete_model_values)):
                struct[i] = discrete_model_values[i].shape[0]
            struct = np.array(struct).cumsum()
            discrete_model_values = np.concatenate(discrete_model_values)

            # Perturb
            discrete_model_values = np.array(discrete_model_values)
            perturbed_discrete_values = []
            # Implement a random walk for the discrete parameter values
            for discrete_value in discrete_model_values:
                nodes_proposed = list(self.network.neighbors(discrete_value))
                weight = np.zeros(shape=(len(nodes_proposed),))
                for ind in range(len(nodes_proposed)):
                    weight[ind] = self.network[discrete_value][nodes_proposed[ind]][self.name_weight]
                weight = weight / sum(weight)
                perturbed_discrete_values.append(np.random.choice(nodes_proposed, 1, p=weight)[0])
            perturbed_discrete_values = np.array(perturbed_discrete_values)
            # Perturbed values anc split according to the structure
            perturbed_discrete_values = np.split(perturbed_discrete_values, struct)[:-1]

        return perturbed_discrete_values

    def calculate_cov(self, accepted_parameters_manager, kernel_index):
        """
        Calculates the covariance matrix of this kernel. Since there is no covariance matrix associated with this
        random walk, it returns an empty list.
        """

        return np.array([0]).reshape(-1, )

    def pmf(self, accepted_parameters_manager, kernel_index, mean, x):
        """Calculates the pdf of the kernel.
        Commonly used to calculate weights.
        Parameters
        ----------
        accepted_parameters_manager: abcpy.AcceptedParametersManager object
            The AcceptedParametersManager to be used.
        kernel_index: integer
            The index of the kernel in the list of kernels in the joint kernel.
        mean: np array, np.float or np.integer
            The reference point of the kernel
        x: The point at which the pdf should be evaluated.
        Returns
        -------
        float
            The pdf evaluated at point x.
        """
        density = 1
        if isinstance(mean[0],
                      (np.float, np.float32, np.float64, np.int, np.int32, np.int64)):
            mean = np.array(mean).astype(int)
            for ind1 in range(len(mean)):
                discrete_value = mean[ind1]
                nodes_proposed = list(self.network.neighbors(discrete_value))
                weight = np.zeros(shape=(len(nodes_proposed),))
                for ind2 in range(len(nodes_proposed)):
                    weight[ind2] = self.network[discrete_value][nodes_proposed[ind2]][self.name_weight]
                weight = weight / sum(weight)
                if x[ind1] in nodes_proposed:
                    density = density * weight[np.where(nodes_proposed == x[ind1])[0]][0]
                else:
                    density = density * 0
            return density
        else:
            mean = np.array(np.concatenate(mean)).astype(int)
            for ind1 in range(len(mean)):
                discrete_value = mean[ind1]
                nodes_proposed = list(self.network.neighbors(discrete_value))
                weight = np.zeros(shape=(len(nodes_proposed),))
                for ind2 in range(len(nodes_proposed)):
                    weight[ind2] = self.network[discrete_value][nodes_proposed[ind2]][self.name_weight]
                weight = weight / sum(weight)
                if x[ind1] in nodes_proposed:
                    density = density * weight[np.where(nodes_proposed == x[ind1])[0]][0]
                else:
                    density = density * 0
            return density