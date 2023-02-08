import pickle
import numpy as np
import pandas as pd
import Utils.utils_topology as utils_topology
import multiprocessing as mp
import Utils.visualising_functions as vf
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(['science', 'no-latex'])


""" Load all the topological parameters """
topological_config = utils_topology.get_parameters('data/configs/topology_settings_like_old.json')

""" Set up varying parameters (optional) """
topological_config.variable = 'shared_discount'
topological_config.values = np.round(np.arange(0.10, 0.51, 0.01), 2)

utils_topology.create_results_directory(topological_config, date="20-12-22")

path = r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\20-12-22\merged_files_20-12-22.xlsx"
df = pd.read_excel(path)
df["Shareable_travellers"] = df["Demand_size"] - df["No_isolated_pairs"]

fig, ax1 = plt.subplots(figsize=(8, 6), dpi=200)
ax2 = ax1.twinx()
sns.lineplot(data=df, x="shared_discount", y="Shareable_travellers", ax=ax1, color="red")
sns.lineplot(data=df, x="shared_discount", y="Average_clustering_group1", ax=ax2, color="blue")
ax1.set(ylabel="Greatest component")
ax2.set(ylabel="Avg. sq. clustering of trav. node")
plt.show()
plt.close()


