import numpy as np
import pickle
import os

import pandas as pd
import seaborn

from ExMAS.main_prob import main as exmas_algo
from NYC_tools import NYC_data_prep_functions as nyc_func
import Topology.utils_topology as utils
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import datetime

# num_list = [1] + list(range(100, 1000, 100))
#
# topological_config = utils.get_parameters('data/configs/topology_settings.json')
# utils.create_results_directory(topological_config)

os.chdir(r'C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Miscellaneous_scripts')

import pickle

with open('final_res_27-07-22.obj', 'rb') as file:
    data = list(pickle.load(file))

x = 0
xx = 00




