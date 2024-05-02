from abcpy.continuousmodels import ProbabilisticModel, Continuous, InputConnector
import numpy as np
import copy

"""
#### This class implements a SBFC model for the spread of fake news on a network
----------
Parameters:
----------
parameters: list
    Contains the probabilistic models and hyper-parameters from which the model derives.
            theta: float
                (Optional) The spreading parameter for the fake news (in the interval [0,1].)
            gamma: float
                (Optional) The spreading parameter for the debunking (in the interval [0,1])
            delta: float
                (Optional) The recovery parameter (in the interval [0,1])
            seed_node: integer
                The node where the infection starts.
network: NetworkX.Graph
    The fixed network structure on which news spreads.
time_observed: numpy.ndarray
    The time-points at which the observation were made.
seed: int
    seed for RNG. Default: None
----------
"""


class FakeNewsSBFC(ProbabilisticModel, Continuous):

    def __init__(self, parameters, network, time_observed, seed=None, name="FakeNewsSBFC"):

        self.node_count = network.number_of_nodes()
        self.network = network
        self.T = max(time_observed) + 1
        self.time_observed = time_observed
        self.rng = np.random.RandomState(seed)

        self.node_status = np.zeros(self.node_count)
        self.fake_news_exposures = [dict() for _ in range(self.node_count)]
        self.debunking_exposures = [dict() for _ in range(self.node_count)]
        self.fake_news_exposed_observed_times = []
        self.debunking_exposed_observed_times = []

        # We expect input of type parameters = [theta, gamma, seed_node]
        if not self._check_input(parameters):
            raise ValueError("The parameter values are out of the model parameter domain.")

        input_parameters = InputConnector.from_list(parameters)
        super().__init__(input_parameters, name)

    def forward_simulate(self, input_values, k, rng=np.random.RandomState(), mpi_comm=None):
        # Initialize local parameters
        #       theta - spreading rate for fake news
        #       gamma - spreading rate for the debunking of the fake news
        #       delta - recovery rate for believer nodes
        theta = input_values[0]
        gamma = input_values[1]
        delta = input_values[2]
        seed_node = [input_values[3]]

        self.fake_news_exposed_observed_times = []
        self.debunking_exposed_observed_times = []

        #print("----------------\nPARAMETERS:\n----------------\n"
        #       "theta:", theta, "\ngamma:", gamma, "\ndelta:", delta, "\nseed_node:", seed_node,
        #       "\n----------------")
        result = self.simulate(theta, gamma, delta, seed_node, k)

        if not self._check_output(result):
            raise RuntimeError("Result is not a 1-dimensional numpy array.")

        # # The results of forward simulate should be an 1-dimensional numpy array for each repetition. So need to think
        # how to compress the data. One option is code infected/non-infected as 0/1 and recovered/ow as 0/1 .. then
        # for each repetition you will have the same length.
        self.fake_news_exposed_observed_times = np.array(self.fake_news_exposed_observed_times)
        self.debunking_exposed_observed_times = np.array(self.debunking_exposed_observed_times)
        return result

    def simulate(self, theta, gamma, delta, seed_node, n_simulate):
        # Initialize local parameters
        diffusion_state_array = [None] * n_simulate

        for k in range(n_simulate):
            # Initialize the time-series
            tmp_diffusion_states = list()

            self.reset_node_status()
            self.reset_exposures()

            # Initialising the initially infected nodes, fact-checking nodes and their statuses
            fact_checker_nodes, believer_nodes = self.init_nodes(seed_node, theta)

            # Recording the results at t=0 if observed
            if 0 in self.time_observed:
                tmp_diffusion_states.append(copy.deepcopy(self.node_status))

            for t in range(1, self.T):
                # Run Spreading Processes
                # print("----------------\nt =", t, "\n----------------")
                new_fact_checker_nodes, new_believer_nodes = self.run_spreading_process(fact_checker_nodes,
                                                                                        believer_nodes, theta,
                                                                                        gamma, delta)

                fact_checker_nodes = copy.deepcopy(new_fact_checker_nodes)
                believer_nodes = copy.deepcopy(new_believer_nodes)

                # Record the results if t is observed
                if t in self.time_observed:
                    current_node_status = copy.deepcopy(self.node_status)
                    tmp_diffusion_states.append(current_node_status)
                    self.record_exposures()

            # add results of the kth simulation
            diffusion_state_array[k] = np.array(tmp_diffusion_states).flatten()

        return diffusion_state_array

    def init_nodes(self, seed_node, theta):
        # Initialising the initially infected nodes, fact-checking nodes and their statuses
        fact_checker_nodes = list()
        neighbours = list(self.network.neighbors(seed_node[0]))
        believer_nodes = list(self.rng.choice(neighbours, int(np.ceil(len(neighbours) * theta)), replace=False))
        believer_nodes.append(seed_node[0])
        for sn in believer_nodes:
            self.node_status[sn] = 1

        return fact_checker_nodes, believer_nodes

    def run_spreading_process(self, fact_checker_nodes, believer_nodes, theta, gamma, delta):
        # FACT CHECKING SPREADING PROCESS
        # Create a placeholder variable to track new fact-checker nodes from the debunking process
        new_fact_checker_nodes = copy.deepcopy(fact_checker_nodes)
        for i in fact_checker_nodes:
            neighbours = list(self.network.neighbors(i))
            chosen_nodes_for_debunking = self.rng.choice(neighbours, int(np.ceil(len(neighbours) * gamma)))
            for node in chosen_nodes_for_debunking:
                # Skip debunking process if node is not a believer
                if self.node_status[node] != 1:
                    continue

                success_probability = self.compute_infection_probability(node, gamma, process_type="debunking")
                if self.rng.binomial(1, success_probability) == 1:
                    new_fact_checker_nodes.append(node)
                    believer_nodes.remove(node)
                    self.node_status[node] = -1

        # FAKE NEWS SPREADING PROCESS
        # Get new believer nodes based on the ones debunked successfully during the fact checking process and
        # create a placeholder variable to track new believer nodes from the fake news spreading process
        new_believer_nodes = copy.deepcopy(believer_nodes)
        for j in believer_nodes:
            # Believer node attempts to self fact-check using Bernoulli(delta), where delta is the recovery rate
            if self.rng.binomial(1, delta) == 1:
                new_fact_checker_nodes.append(j)
                new_believer_nodes.remove(j)
                self.node_status[j] = -1
                continue

            neighbours = list(self.network.neighbors(j))
            chosen_nodes_for_infecting = self.rng.choice(neighbours, int(np.ceil(len(neighbours) * theta)))
            for node in chosen_nodes_for_infecting:
                # Skip infection process if node is not  susceptible
                if self.node_status[node] != 0:
                    continue

                success_probability = self.compute_infection_probability(node, theta, process_type="fake_news")
                if self.rng.binomial(1, success_probability) == 1:
                    new_believer_nodes.append(node)
                    self.node_status[node] = 1

        return new_fact_checker_nodes, new_believer_nodes

    def compute_infection_probability(self, node, spreading_parameter, process_type):
        # Find the neighbours of the node to be converted
        node_neighbours = list(self.network.neighbors(node))
        F_node = len(node_neighbours)

        if process_type == "fake_news":
            exposures = self.update_fake_news_exposures(node, node_neighbours)
        elif process_type == "debunking":
            exposures = self.update_debunking_exposures(node, node_neighbours)
        else:
            raise RuntimeError("Spreading process type must be either 'fake_news' or 'debunking'.")

        success_probability = self.compute_probability(exposures, F_node, spreading_parameter)
        return success_probability

    def update_debunking_exposures(self, node, node_neighbours):
        FC_neighbours_count = 0
        for adj_node in node_neighbours:
            if self.node_status[adj_node] == -1:
                FC_neighbours_count += 1

        # Increment the value for the key representing the number of fact-checking neighbours
        # by 1 exposure
        if FC_neighbours_count not in self.debunking_exposures[node].keys():
            self.debunking_exposures[node][FC_neighbours_count] = 1
        else:
            self.debunking_exposures[node][FC_neighbours_count] += 1

        return copy.deepcopy(self.debunking_exposures[node])

    def update_fake_news_exposures(self, node, node_neighbours):
        B_neighbours_count = 0
        for adj_node in node_neighbours:
            if self.node_status[adj_node] == 1:
                B_neighbours_count += 1

        # Increment the value for the key representing the number of believer neighbours
        # by 1 exposure
        if B_neighbours_count not in self.fake_news_exposures[node].keys():
            self.fake_news_exposures[node][B_neighbours_count] = 1
        else:
            self.fake_news_exposures[node][B_neighbours_count] += 1

        return copy.deepcopy(self.fake_news_exposures[node])

    def compute_probability(self, exposures, degree, spreading_param):
        # exposures here is expected to be a dictionary associated with the respective node of concern
        # M: the maximum number of neighbours that have ever been fact-checkers/believers
        probability = 1
        M = max(exposures.keys())

        for neighbour_count in exposures.keys():
            if neighbour_count == M:
                continue
            probability *= pow((1 - self.compute_pk(neighbour_count, degree, spreading_param)),
                               exposures[neighbour_count])

        probability *= pow((1 - self.compute_pk(M, degree, spreading_param)), (exposures[M] - 1))
        probability *= self.compute_pk(M, degree, spreading_param)

        return probability

    def compute_pk(self, neighbour_count, degree, spreading_param):
        epsilon_max, epsilon_min, shape = 0.25, 0.001, 1
        pk = epsilon_min + ((epsilon_max - epsilon_min) /
                            (1 + np.exp(-shape * (neighbour_count - spreading_param * degree))))
        return pk

    def _check_input(self, input_values):
        # raises exceptions if the input is of wrong type or has the wrong format.
        # returns False if the values of the input models are not compatible, True otherwise.
        if len(input_values) != 4:
            raise RuntimeError('Input parameters must be a list with 4 elements, parameters=(theta, gamma, delta, '
                               'seed_node).')

        if not isinstance(input_values, list):
            raise TypeError('Input parameters must be of type: list')

        return True

    def _check_output(self, values):
        for simulation_results in values:
            if len(simulation_results) != self.get_output_dimension():
                return False
        return True

    def get_output_dimension(self):
        return len(self.time_observed) * self.node_count

    def update_network(self, network):
        self.network = network

    def reset_node_status(self):
        # node_status encodes the status of each node as:
        #    0: susceptible, 1: believer, -1: fact-checker
        # All nodes initialised to susceptible
        self.node_status = np.zeros(self.node_count)

    def reset_exposures(self):
        # Exposure summaries for each node are stored in the elements of a list containing all nodes.
        # Within the list, each node has a corresponding dictionary which records:
        #       key: number of fact-checker/believer nodes neighbouring that node
        #       value: number of times that node has been exposed in total for the corresponding key
        # Each time a node is exposed to debunking/fake news, we will count the number of neighbours of that node
        # which are fact-checkers/believers. Then, we either create a key to represent that count with
        # a corresponding value of 1 exposure, or increment the value for that pre-existing key by 1 exposure.
        self.fake_news_exposures = [dict() for _ in range(self.node_count)]
        self.debunking_exposures = [dict() for _ in range(self.node_count)]

    def record_exposures(self):
        for node in range(self.node_count):
            if len(self.fake_news_exposures[node]) > 0:
                self.fake_news_exposed_observed_times.append(1)
            else:
                self.fake_news_exposed_observed_times.append(0)

            if len(self.debunking_exposures[node]) > 0:
                self.debunking_exposed_observed_times.append(1)
            else:
                self.debunking_exposed_observed_times.append(0)

            

