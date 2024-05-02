import numpy as np
import networkx as nx
import logging
from Model import FakeNewsSBFC
from abcpy.continuousmodels import Uniform
from abcpy.discretemodels import DiscreteUniform
from abcpy.statistics import Identity
from abcpy.backends import BackendDummy, BackendMPI
from abcpy.inferences import SABC
from abcpy.output import Journal
from Distance import ElementDifference
from abcpy.perturbationkernel import DefaultKernel, JointPerturbationKernel, MultivariateNormalKernel

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
A = np.load('../Networks/' + case + '_' + str(node_no) + '_network.npy')
network = nx.from_numpy_matrix(A)

# Define Graphical Model

theta = Uniform([[0.25], [0.75]], name='theta')
gamma = Uniform([[0.25], [0.75]], name='gamma')
delta = Uniform([[0.0001], [0.005]], name='delta')
seed_node = DiscreteUniform([[0], [node_no-1]], name='seed_node')
FakeNewsSBFC = FakeNewsSBFC([theta, gamma, delta, seed_node], network, time_observed, name='FakeNewsSBFC')

# Example to Generate Data to check it's correct
result_fake_obs = FakeNewsSBFC.forward_simulate([.3, .3, .0004, 10], 2)
# result_fake_obs_1 = FakeNewsSBFC.forward_simulate([.3, .01, 10], 3)

# Summary statistics (here we use Identity statistics)
summary_stat = Identity(degree=1, cross=False)
# stat_fake_obs = summary_stat.statistics(result_fake_obs)
# print(stat_fake_obs.shape)

# Define distance on the summary statistics space (use similarity at each element of the vector)

dist_calc = ElementDifference(summary_stat)
# print(dist_calc.distance(result_fake_obs, result_fake_obs_1))

# The step to infer the parameters
# First define the kernel
#kernel = DefaultKernel([theta, gamma, delta, seed_node])
from kernel import NetworkRandomWalkKernelInverseDegree
kernel_1 = MultivariateNormalKernel([theta, gamma, delta])
kernel_2 = NetworkRandomWalkKernelInverseDegree([seed_node], network)
kernel = JointPerturbationKernel([kernel_1, kernel_2])

# Second define backend, whether parallelize or not
backend = BackendMPI()
#backend = BackendDummy()

# Now define the algorithm for inference
sampler = SABC([FakeNewsSBFC], [dist_calc], backend, kernel, seed=1)
# The following are the tuning parameters for the algorithm, please choose as appropriate
steps, epsilon, n_samples, n_samples_per_param = 30, [1e10], 1000, 1

# Run inference and save the journals containing approximate posterior samples
journal_sabc = sampler.sample(observations=[result_fake_obs], steps=steps, epsilon=epsilon, n_samples=n_samples,
                              n_samples_per_param=n_samples_per_param, ar_cutoff=0.001, full_output=1)
journal_sabc.save("sabc_" + problem + '_obs.jrnl')

# Plot the posteriors

journal_sabc = Journal.fromFile("sabc_" + problem + '_obs.jrnl')
print(journal_sabc.configuration)
print(journal_sabc.posterior_mean())
journal_sabc.plot_posterior_distr(path_to_save="sabc_" + problem + '_posterior.pdf', show_samples=False, true_parameter_values=[.3, .3, .0004, 10])
