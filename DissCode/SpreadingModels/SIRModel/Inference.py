import numpy as np
import networkx as nx
import logging
from Model import FakeNewsSIR

logging.basicConfig(level=logging.INFO)
problem = 'FakeNewsSIR'

# ==============================================================================
# Different types of network (BA: Barabasi-Albert, ER: Erdos-Renyi, FB: Facebook Social Network,
# INRV: Indian Village contact Network) with node_no many nodes on the network. The infection_node
# is the true seed-node. (Choose one of the options)
# ==============================================================================
# case, node_no, infection_node = 'ba', 100, 4
case, node_no, infection_node = 'er', 100, 10
# case, node_no, infection_node = 'inrv', 354, 70
# case, node_no, infection_node = 'fb', 4039, 2000
# ==============================================================================
# Time observed
time_observed = np.arange(20, 70 + 1)
# ==============================================================================
# Load network
# ==============================================================================
A = np.load('Networks/' + case + '_' + str(node_no) + '_network.npy')
network = nx.from_numpy_matrix(A)

# Define Graphical Model
from abcpy.continuousmodels import Uniform
from abcpy.discretemodels import DiscreteUniform

theta = Uniform([[0.25], [0.75]], name='theta')
gamma = Uniform([[0.000280], [0.00059]], name='gamma')
seed_node = DiscreteUniform([[0], [node_no]], name='seed_node')
FakeNewsSIR = FakeNewsSIR([theta, gamma, seed_node], network, time_observed, name='FakeNewsSIR')

# Example to Generate Data to check it's correct
result_fake_obs = FakeNewsSIR.forward_simulate([.3, .0004, 10], 2)
# result_fake_obs_1 = FakeNewsSIR.forward_simulate([.3, .01, 10], 3)

# Summary statistics (here we use Identity statistics)
from abcpy.statistics import Identity

stat_cal = Identity(degree=1, cross=False)
# stat_fake_obs = stat_cal.statistics(result_fake_obs)
# print(stat_fake_obs.shape)

# Define distance on the summary statistics space (use similarity at each element of the vector)
from Distance import ElementDifference

dist_calc = ElementDifference(stat_cal)
# print(dist_calc.distance(result_fake_obs, result_fake_obs_1))

# The step to infer the parameters
# First define the kernel
from abcpy.perturbationkernel import DefaultKernel

kernel = DefaultKernel([theta, gamma, seed_node])

# Second define backend, whether parallelize or not
from abcpy.backends import BackendDummy, BackendMPI

# backend = BackendMPI()
backend = BackendDummy()

# Now define the algorithm for inference
from abcpy.inferences import SABC

sampler = SABC([FakeNewsSIR], [dist_calc], backend, kernel, seed=1)
# The following are the tuning parametrs for the algorithm, please choose as appropriate
steps, epsilon, n_samples, n_samples_per_param = 4, [1e10], 5, 1

# Run inference and save the journals containing approximate posterior samples
journal_sabc = sampler.sample(observations=[result_fake_obs], steps=steps, epsilon=epsilon, n_samples=n_samples,
                              n_samples_per_param=n_samples_per_param, ar_cutoff=0.001, full_output=1)
journal_sabc.save("sabc_" + problem + '_obs.jrnl')

# Plot the posteriors
from abcpy.output import Journal

journal_sabc = Journal.fromFile("sabc_" + problem + '_obs.jrnl')
print(journal_sabc.configuration)
print(journal_sabc.posterior_mean())
journal_sabc.plot_posterior_distr(path_to_save="sabc_" + problem + '_posterior.pdf', show_samples=False)
