import pickle
import pandas as pd
import numpy as np
from Utils import utils_topology as utils
import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use(['science', 'no-latex'])

date = "19-01-23"
special_name = "_full"
sblts_exmas = "exmas"

# with open('data/results/' + date + '/rep_graphs_' + date + '.obj', 'rb') as file:
#     e = pickle.load(file)

with open('data/results/' + date + special_name + '/dotmap_list_' + date + '.obj', 'rb') as file:
    e = pickle.load(file)

# with open('data/results/' + date + '/all_graphs_list_' + date + '.obj', 'rb') as file:
#     e = pickle.load(file)

topological_config = utils.get_parameters('data/configs/topology_settings_panel.json')
# utils.create_results_directory(topological_config, date=date)
topological_config.path_results = 'data/results/' + date + special_name + '/'

# G = e[0]['pairs_matching']
# G = e['pairs_shareability']

# colours = []
# for item in G.nodes:
#     if item == 0:
#         colours.append("red")
#     else:
#         colours.append("black")
# nx.set_node_attributes(G, dict(zip(list(G.nodes), colours)), "group")
# visualize(G, config=json.load(open('data/configs/netwulf_config.json')))
#
# edge_colours = []
# for item in G.edges:
#     if item[0] == 0 or item[1] == 0:
#         edge_colours.append("red")
#     else:
#         edge_colours.append("black")
# nx.set_edge_attributes(G, dict(zip(list(G.edges), colours)), "colour")
#
# visualize(G, config=json.load(open('data/configs/netwulf_config.json')))

# nx.draw_networkx_nodes(G, pos=nx.spring_layout(G), node_size=10, node_color="black")
# nx.draw_networkx_edges(G, pos=nx.spring_layout(G), width=2, edge_color="black")
# plt.show()


# visualize(e['pairs_matching'], config=json.load(open('data/configs/netwulf_config.json')))
# draw_bipartite_graph(e['bipartite_matching'], 1000, topological_config, date=date, save=True,
#                      name="full_bi_matching", dpi=200, colour_specific_node=None,
#                      default_edge_size=0.1)

# num_list = [1, 5, 10, 100, 900]
# for num in num_list:
#     if num == 1:
#         obj = [e[1]]
#     else:
#         obj = e[:num]
#     utils.draw_bipartite_graph(utils.analyse_edge_count(obj, topological_config,
#                                                         list_types_of_graph=['bipartite_matching'],
#                                                         logger_level='WARNING')[
#                                    'bipartite_matching'],
#                                num, node_size=1, dpi=80, figsize=(10, 24), plot=False, width_power=1,
#                                config=topological_config, save=True, saving_number=num, date=date)

# fig, axes = plt.subplots(nrows=2, ncols=5, sharex='col', sharey='row')
#
# for col in range(2):
#     for row in range(5):
#         num = num_list[5*col+row]
#         img = mpimg.imread(topological_config.path_results + "temp/graph_" +
#                            str(datetime.date.today().strftime("%d-%m-%y")) + "_no_" + str(num) + ".png")
#         axes[col, row].imshow(img)
#         axes[col, row].set_title('Step ' + str(num))
#         axes[col, row].axis('off')
#
# plt.savefig(topological_config.path_results + "graph_growth" + str(datetime.date.today().strftime("%d-%m-%y")) + ".png")
# plt.show()

# topological_config.path_results = 'data/results/31-05-22/'

# num_list = [1] + list(range(100, 1000, 100))
# df = pd.DataFrame()
# for num in num_list:
#     if num == 1:
#         obj = [e[0]]
#     else:
#         obj = e[:num]
#     temp_df = utils.analysis_all_graphs(obj, topological_config, save=False, save_num=num, date='31-05-22')
#     df = pd.concat([df, temp_df])
#
# df.to_excel(topological_config.path_results + 'all_graphs_properties_' + '31-05-22' + '.xlsx')

num_list = list(range(1000))
df = pd.DataFrame()
for num in num_list:
    if num == 0:
        obj = [e[0]]
    else:
        obj = e[:num]
    temp_graph = utils.analyse_edge_count(obj, topological_config, list_types_of_graph=['pairs_matching'],
                             logger_level='WARNING')['pairs_matching']
    t = utils.graph_mini_graphstatistics(temp_graph)
    temp_df = pd.DataFrame.from_dict({'average_degree': [t.average_degree], 'max_comp': [t.proportion_max_component],
                                      'number_of_isolated': [t.number_of_isolated_pairs]})
    df = pd.concat([df, temp_df])

#
df.reset_index(inplace=True)
df.drop(columns=['index'], inplace=True)
df.to_excel(topological_config.path_results + 'frame_evolution_' + date + '.xlsx', index=False)

# str_for_end = "_198"
# ticks = 5
# multiplier = 100
#
# ax = sns.histplot([multiplier*(x[sblts_exmas].res.PassUtility_ns - x[sblts_exmas].res.PassUtility) / x[sblts_exmas].res.PassUtility_ns for x in e])
# ax.set(xlabel="Utility gain", ylabel=None)
# ax.xaxis.set_major_formatter(mtick.PercentFormatter())
# plt.locator_params(axis='x', nbins=ticks)
# plt.savefig(topological_config.path_results + "figs/relative_pass_utility" + str_for_end + ".png")
# plt.close()
#
# ax = sns.histplot([multiplier*(x[sblts_exmas].res.PassHourTrav - x[sblts_exmas].res.PassHourTrav_ns) / x[sblts_exmas].res.PassHourTrav_ns for x in e])
# ax.set(xlabel="Travel time extension", ylabel=None)
# ax.xaxis.set_major_formatter(mtick.PercentFormatter())
# plt.locator_params(axis='x', nbins=ticks)
# plt.savefig(topological_config.path_results + "figs/relative_pass_hours" + str_for_end + ".png")
# plt.close()
#
# ax = sns.histplot([multiplier*(x[sblts_exmas].res.VehHourTrav_ns - x[sblts_exmas].res.VehHourTrav) / x[sblts_exmas].res.VehHourTrav_ns for x in e])
# ax.set(xlabel='Vehicle time shortening', ylabel=None)
# ax.xaxis.set_major_formatter(mtick.PercentFormatter())
# plt.locator_params(axis='x', nbins=ticks)
# plt.savefig(topological_config.path_results + "figs/relative_veh_hours" + str_for_end + ".png")
# plt.close()


# y = [(x[sblts_exmas].res.PassUtility_ns - x[sblts_exmas].res.PassUtility) / x[sblts_exmas].res.PassUtility_ns for x in e]
# print(y)
# x = 0

# with open("C:/Users/szmat/Documents/GitHub/ExMAS_sideline/Topology/data/results/"
#           + date + "/final_matching_" + date + ".json", "r") as file:
#     matches = json.load(file)
#
# number_elements = 99
# replications = 100
#
# # match_list = [list(map(lambda x: x.replace("(", "").replace(")", ""),
# #                        list(matches.keys())[t].split(","))) for t in range(len(matches.keys()))]
# #
# # list(map(lambda x: x.pop(-1) if x[-1] == "" else x, match_list))
#
# str_for_end = "_" + str(number_elements)
# ticks = 5
#
# sharing_prob = []
#
# for j in range(number_elements):
#     sharing_prob.append((replications - matches.get("(" + str(j) + ",)", 0))*100/replications)
#
#
# ax = sns.histplot(sharing_prob)
# ax.set(xlabel='Probability of sharing', ylabel=None)
# ax.xaxis.set_major_formatter(mtick.PercentFormatter())
# plt.locator_params(axis='x', nbins=ticks)
# plt.xlim(0, 100)
# plt.savefig(topological_config.path_results + "figs/prob_sharing" + str_for_end + ".png")
# plt.close()


# probs = {"c0": np.array([0, 0]), "c1": np.array([0, 0]), "c2": np.array([0, 0]), "c3": np.array([0, 0])}
# for rep in e:
#     df = rep.prob.sampled_random_parameters.copy()
#     df["VoT"] *= 3600
#     df.set_index("new_index", inplace=True)
#     c0 = df.loc[df["class"] == 0]
#     c1 = df.loc[df["class"] == 1]
#     c2 = df.loc[df["class"] == 2]
#     c3 = df.loc[df["class"] == 3]
#
#     schedule = rep[sblts_exmas].schedule
#     indexes_sharing = schedule["indexes"].values
#     a1 = {frozenset(t) for t in indexes_sharing}
#     a2 = {next(iter(t)) for t in a1 if len(t) == 1}
#
#     probs["c0"] += np.array([len(a2.intersection(set(c0.index))), len(c0)])
#     probs["c1"] += np.array([len(a2.intersection(set(c1.index))), len(c1)])
#     probs["c2"] += np.array([len(a2.intersection(set(c2.index))), len(c2)])
#     probs["c3"] += np.array([len(a2.intersection(set(c3.index))), len(c3)])
#
# x = ("c0", "c1", "c2", "c3")
#
#
# def foo(i):
#     return (probs["c" + str(i)][1] - probs["c" + str(i)][0]) / probs["c" + str(i)][1]
#
#
# y = [foo(t) for t in range(4)]
#
# sns.barplot(data=pd.DataFrame({"names": x, "values": y}), x="names", y="values")
# plt.show()



# import pickle
# import networkx as nx
# import netwulf as nw
# import Utils.visualising_functions as vis_eff
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import matplotlib.ticker as mtick
# import sys
# import os
# import collections
# import numpy as np
# import time
#
# os.chdir(os.path.dirname(os.getcwd()))
#
# import scienceplots
#
# plt.style.use(['science', 'no-latex'])
#
# date = "25-11-22"
# name0 = "_full_n"
# name1 = "_mini_n"
# name2 = "_maxi_n"
#
# config = vis_eff.config_initialisation('Topology/data/configs/topology_settings3.json', date, "exmas")
# config.sblts_exmas = "exmas"
#
# config.path_results0 = 'Topology/data/results/' + date + name0 + '/'
# config.path_results1 = 'Topology/data/results/' + date + name1 + '/'
# config.path_results2 = 'Topology/data/results/' + date + name2 + '/'
#
# rep_graphs0, dotmap_list0, all_graphs_list0 = vis_eff.load_data(config, config.path_results0)
# rep_graphs1, dotmap_list1, all_graphs_list1 = vis_eff.load_data(config, config.path_results1)
# rep_graphs2, dotmap_list2, all_graphs_list2 = vis_eff.load_data(config, config.path_results2)
#
# for var in ['profit', 'veh', 'utility', 'pass']:
#
#     sblts_exmas = "exmas"
#     assert var in ['veh', 'utility', 'pass', 'profit'], "wrong var"
#
#     if var == 'profit':
#         config0path = "profitability_mean_147.txt"
#         config0path_num = 0
#     else:
#         config0path = "kpis_means_147.txt"
#         multiplier = 100
#
#     if var == 'veh':
#         config0path_num = 2
#         var1 = "VehHourTrav_ns"
#         var2 = "VehHourTrav"
#         var3 = var1
#
#     elif var == 'utility':
#         config0path_num = 0
#         var1 = "PassUtility_ns"
#         var2 = "PassUtility"
#         var3 = var1
#
#     elif var == 'pass':
#         config0path_num = 1
#         var1 = "PassHourTrav"
#         var2 = "PassHourTrav_ns"
#         var3 = var2
#
#     else:
#         pass
#
#     if var == 'pass':
#         _lw = 2
#     else:
#         _lw = 1.5
#
#     with open(config.path_results0 + config0path, "r") as file:
#         hline_val = file.readlines()
#
#     # h_line_value = float(hline_val[config0path_num].split()[1])
#
#
#     if var in ['veh', 'utility', 'pass']:
#         data0 = [multiplier * (x[sblts_exmas].res[var1] - x[sblts_exmas].res[var2]) / x[
#             sblts_exmas].res[var3] for x in dotmap_list0]
#         data1 = [multiplier * (x[sblts_exmas].res[var1] - x[sblts_exmas].res[var2]) / x[
#             sblts_exmas].res[var3] for x in dotmap_list1]
#         data2 = [multiplier * (x[sblts_exmas].res[var1] - x[sblts_exmas].res[var2]) / x[
#             sblts_exmas].res[var3] for x in dotmap_list2]
#     else:
#         data0 = vis_eff.analyse_profitability(dotmap_list0, config, save_results=False)
#         data1 = vis_eff.analyse_profitability(dotmap_list1, config, save_results=False)
#         data2 = vis_eff.analyse_profitability(dotmap_list2, config, save_results=False)
#
#     data = pd.concat(axis=0, ignore_index=True, objs=[
#         pd.DataFrame.from_dict({'value': data0, 'Demand size': 'baseline'}),
#         pd.DataFrame.from_dict({'value': data1, 'Demand size': 'small'}),
#         pd.DataFrame.from_dict({'value': data2, 'Demand size': 'big'})
#     ])
#
#     fig = plt.figure(figsize=(3.5, 3))
#
#     data2 = pd.DataFrame()
#     data2["value"] = data["value"]
#     data2["Demand size"] = data["Demand size"]
#
#     with open(config.path_results0 + 'temp.obj', 'wb') as file:
#         pickle.dump(data2, file)
#
#     ax = sns.kdeplot(data=data2, x='value', hue='Demand size', palette=['red', 'blue', 'green'], common_norm=False,
#                      fill=True, alpha=.5, linewidth=0)
#     ax.set(xlabel=None, ylabel=None, yticklabels=[])
#     plt.locator_params(axis='x', nbins=5)
#     plt.tight_layout()
#
#     if var != "profit":
#         ax.xaxis.set_major_formatter(mtick.PercentFormatter())
#         ax.get_legend().remove()
#     else:
#         legend = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", labels=['Baseline', 'Sparse', 'Dense'],
#                             title='Demand', borderaxespad=0)
#         fig.canvas.draw()
#         print(f'Height: {legend.get_window_extent().height / 200} in')
#         print(f'Width: {legend.get_window_extent().width / 200} in')
#
#     plt.savefig(config.path_results0 + "figs/mixed_" + var + "_v2.png", dpi=200)
#     plt.close()
#
#
# with open(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\16-01-23\all_graphs_list_16-01-23.obj", 'rb') as file:
#     graphs = pickle.load(file)
#
# bipartite_shareability_list = [e["bipartite_shareability"] for e in graphs]
#
#
# def calc(G, disc):
#     start_time = time.time()
#     partition_for_bipartite = nx.bipartite.basic.color(G)
#     for colour_key in partition_for_bipartite.keys():
#         G.nodes[colour_key]['bipartite'] = partition_for_bipartite[colour_key]
#     total_colouring = {k: v['bipartite'] for k, v in G._node.copy().items()}
#     group0_colour = {k: v for k, v in total_colouring.items() if v == 0}
#     group1_colour = {k: v for k, v in total_colouring.items() if v == 1}
#
#     part_1_time = time.time()
#     print(f"Time of execution for the 1st part: {part_1_time - start_time}")
#
#     sq_coefficient = nx.square_clustering(G)
#     part_2_time = time.time()
#     print(f"Time of execution for the 1st part: {part_2_time - part_1_time}")
#
#     group0 = {k: v for k, v in sq_coefficient.items() if k in group0_colour.keys()}
#     group1 = {k: v for k, v in sq_coefficient.items() if k in group1_colour.keys()}
#     if len(sq_coefficient) != 0:
#         average_clustering_coefficient = sum(sq_coefficient.values()) / len(sq_coefficient)
#     else:
#         average_clustering_coefficient = 0
#     average_clustering_group0 = sum(group0.values()) / len(group0)
#     average_clustering_group1 = sum(group1.values()) / len(group1)
#
#     part_3_time = time.time()
#     print(f"Time of execution for the 1st part: {part_3_time - part_2_time}")
#
#     degree_sequence = sorted([d for n, d in G.degree()], reverse=False)
#     degree_counter = collections.Counter(degree_sequence)
#     deg, cnt = zip(*degree_counter.items())
#     average_degree = np.sum(np.multiply(deg, cnt)) / G.number_of_nodes()
#     degrees = dict(G.degree())
#     group0 = {k: v for k, v in degrees.items() if k in group0_colour.keys()}
#     group1 = {k: v for k, v in degrees.items() if k in group1_colour.keys()}
#     average_degree_group0 = sum(group0.values()) / len(group0)
#     average_degree_group1 = sum(group1.values()) / len(group1)
#
#     part_4_time = time.time()
#     print(f"Time of execution for the 1st part: {part_4_time - part_3_time}")
#
#     return {"disc": disc,
#             "average_clustering_coefficient": average_clustering_coefficient,
#             "average_clustering_group0": average_clustering_group0,
#             "average_clustering_group1": average_clustering_group1,
#             "average_degree": average_degree,
#             "average_degree_group0": average_degree_group0,
#             "average_degree_group1": average_degree_group1}
#
#
# def calc2(G, disc):
#     return {"disc": disc,
#             "average_degree": np.mean([b for a, b in nx.degree(G)]),
#             "average_clustering": nx.average_clustering(G)}
#
#
# res_bip = [calc(g, d) for g, d in zip([e["bipartite_shareability"] for e in graphs], np.arange(0, 0.51, 0.01))]
#
# with open('res_bip.obj', 'wb') as file:
#     pickle.dump(res_bip, file)
#
# res_sim = [calc2(g, d) for g, d in zip([e["pairs_shareability"] for e in graphs], np.arange(0, 0.51, 0.01))]
#
# with open('res_sim.obj', 'wb') as file:
#     pickle.dump(res_sim, file)
#
#
# with open('res_bip.obj', 'rb') as file:
#     res_bip = pickle.load(file)
#
# with open('res_sim.obj', 'rb') as file:
#     res_sim = pickle.load(file)
#
# x = [e["disc"] for e in res_bip]
# y1 = [e["average_degree_group1"] for e in res_bip]
# y2 = [e["average_degree_group0"] for e in res_bip]
#
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(x, y1, 'g-')
# ax2.plot(x, y2, label='b-')
# ax1.set_xlabel('Sharing discount')
# plt.xlim(left=0, right=0.5)
# ax1.set_ylabel('Av. degree travellers', color='g')
# ax2.set_ylabel('Av. degree rides', color='b')
# plt.savefig("Topology/data/results/16-01-23/figs/degree_discount")
# plt.close()

# x = [e["disc"] for e in res_sim]
# y1 = [e["average_degree"] for e in res_sim]
# y2 = [e["average_clustering"] for e in res_sim]
#
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.plot(x, y1, 'g-')
# ax2.plot(x, y2, label='b-')
# ax1.set_xlabel('Sharing discount')
# plt.xlim(left=0, right=0.5)
# ax1.set_ylabel('Av. degree', color='g')
# ax2.set_ylabel('Av. clustering', color='b')
# plt.savefig("Topology/data/results/20-12-22/figs/simple_both_discount")
# plt.close()

# class TempClass:
#     def __init__(self, dataset: pd.DataFrame, output_path: str, output_temp: str, input_variables: list,
#                  all_graph_properties: list, kpis: list, graph_properties_to_plot: list, labels: dict,
#                  err_style: str = "band", date: str = '000'):
#         """
#         Class designed to performed analysis on merged results from shareability graph properties.
#         :param dataset: input merged datasets from replications
#         :param output_path: output for final results
#         :param output_temp: output for temporal results required in the process
#         :param input_variables: search space variables
#         :param all_graph_properties: all graph properties for heatmap/correlation analysis
#         :param kpis: final matching coefficients to take into account
#         :param graph_properties_to_plot: properties of graph to be plotted
#         :param labels: dictionary of labels
#         :param err_style: for line plots style of the error
#         """
#         if 'Replication_ID' in dataset.columns or 'Replication' in dataset.columns:
#             for x in ['Replication_ID', 'Replication']:
#                 if x in dataset.columns:
#                     self.dataset = dataset.drop(columns=[x])
#                 else:
#                     pass
#         else:
#             self.dataset = dataset
#         self.input_variables = input_variables
#         self.all_graph_properties = all_graph_properties
#         self.dataset_grouped = self.dataset.groupby(self.input_variables)
#         self.output_path = output_path
#         self.output_temp = output_temp
#         self.kpis = kpis
#         self.graph_properties_to_plot = graph_properties_to_plot
#         self.labels = labels
#         self.err_style = err_style
#         self.heatmap = None
#         self.date = date
#
#     def alternate_kpis(self):
#         if 'nP' in self.dataset.columns:
#             pass
#         else:
#             self.dataset['nP'] = self.dataset['No_nodes_group1']
#
#         self.dataset['Proportion_singles'] = self.dataset['SINGLE'] / self.dataset['nR']
#         self.dataset['Proportion_pairs'] = self.dataset['PAIRS'] / self.dataset['nR']
#         self.dataset['Proportion_triples'] = self.dataset['TRIPLES'] / self.dataset['nR']
#         self.dataset['Proportion_triples_plus'] = (self.dataset['nR'] - self.dataset['SINGLE'] -
#                                                    self.dataset['PAIRS']) / self.dataset['nR']
#         self.dataset['Proportion_quadruples'] = self.dataset['QUADRIPLES'] / self.dataset['nR']
#         self.dataset['Proportion_quintets'] = self.dataset['QUINTETS'] / self.dataset['nR']
#         self.dataset['Proportion_six_plus'] = self.dataset['PLUS5'] / self.dataset['nR']
#         self.dataset['SavedVehHours'] = (self.dataset['VehHourTrav_ns'] - self.dataset['VehHourTrav']) / \
#                                         self.dataset['VehHourTrav_ns']
#         self.dataset['AddedPasHours'] = (self.dataset['PassHourTrav'] - self.dataset['PassHourTrav_ns']) / \
#                                         self.dataset['PassHourTrav_ns']
#         self.dataset['UtilityGained'] = (self.dataset['PassUtility_ns'] - self.dataset['PassUtility']) / \
#                                         self.dataset['PassUtility_ns']
#         self.dataset['Fraction_isolated'] = self.dataset['No_isolated_pairs'] / self.dataset['nP']
#         self.dataset_grouped = self.dataset.groupby(self.input_variables)
#
#     def plot_kpis_properties(self):
#         plot_arguments = [(x, y) for x in self.graph_properties_to_plot for y in self.kpis]
#         dataset = self.dataset.copy()
#         binning = False
#         for counter, value in enumerate(self.input_variables):
#             min_val = min(self.dataset[value])
#             max_val = max(self.dataset[value])
#             if min_val == 0 and max_val == 0:
#                 binning = False
#             else:
#                 step = (max_val - min_val) / 3
#                 if min_val < 5:
#                     bins = np.round(np.append(np.arange(min_val * 0.98, max_val * 1.02, step), [max_val + step]), 3)
#                     bins = np.round(np.arange(0, 0.51, 0.1), 3)
#                 else:
#                     bins = np.round(np.append(np.arange(min_val * 0.98, max_val * 1.02, step), [max_val + step]), 0)
#                 labels = [f'{i}+' if j == np.inf else f'{i}-{j}' for i, j in
#                           zip(bins, bins[1:])]  # additional part with infinity
#                 dataset[self.labels[value] + " bin"] = pd.cut(dataset[value], bins, labels=labels)
#                 binning = True
#
#         for counter, j in enumerate(plot_arguments):
#             if not binning:
#                 fig, ax = plt.subplots()
#                 sns.scatterplot(x=j[0], y=j[1], data=dataset, palette="crest")
#                 ax.set_xlabel(self.labels[j[0]])
#                 ax.set_ylabel(self.labels[j[1]])
#                 plt.savefig(self.output_temp + 'kpis_properties_' + str(counter) + '.png')
#                 plt.close()
#             else:
#                 if len(self.input_variables) == 1:
#                     fig, ax = plt.subplots()
#                     sns.scatterplot(x=j[0], y=j[1], data=dataset,
#                                     hue=dataset[self.labels[self.input_variables[0]] + " bin"], palette="crest")
#                     ax.set_xlabel(None)
#                     ax.set_ylabel(None)
#                     plt.legend(fontsize=8, title="Sharing discount", title_fontsize=9)
#                     # ax.get_legend().remove()
#                     plt.savefig(self.output_temp + j[0] + '###' + j[1] + '.png')
#                     plt.close()
#                 elif len(self.input_variables) == 2:
#                     fix, ax = plt.subplots()
#                     sns.scatterplot(x=j[0], y=j[1], data=dataset,
#                                     hue=dataset[self.labels[self.input_variables[0]] + " bin"],
#                                     size=dataset[self.labels[self.input_variables[1]] + " bin"], palette="crest")
#                     ax.set_xlabel(self.labels[j[0]])
#                     ax.set_ylabel(self.labels[j[1]])
#                     plt.savefig(self.output_temp + 'kpis_properties_' + str(counter) + '.png')
#                     plt.close()
#                 else:
#                     fig, ax = plt.subplots()
#                     sns.scatterplot(x=j[0], y=j[1], data=dataset, palette="crest")
#                     ax.set_xlabel(self.labels[j[0]])
#                     ax.set_ylabel(self.labels[j[1]])
#                     plt.savefig(self.output_temp + 'kpis_properties_' + str(counter) + '.png')
#                     plt.close()
#     def do(self):
#         self.alternate_kpis()
#         self.plot_kpis_properties()
#
#
# variables = ['shared_discount']
# TempClass(pd.read_excel(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\18-01-23_net_fixed\merged_files_18-01-23.xlsx"),
#           topological_config.path_results,
#           topological_config.path_results + "temp/",
#           variables,
#           topological_config.graph_topological_properties,
#           topological_config.kpis,
#           topological_config.graph_properties_against_inputs,
#           topological_config.dictionary_variables).do()

# plt.style.use(['science', 'no-latex'])
# 
# df = pd.read_excel(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\19-01-23_full\frame_evolution_19-01-23.xlsx")
#
# plt, ax1 = plt.subplots(figsize=(4, 3), dpi=200)
# ax2 = ax1.twinx()
# # ax1.plot(df["average_degree"], label="Average degree", color='blue', lw=.5, marker='o', linestyle='solid', markersize=1)
# # ax2.plot(df["max_comp"], label="Greatest component", color='red', lw=.5, marker='o', linestyle='solid', markersize=1)
# ax1.plot(df["average_degree"], label="Average degree", color='blue', lw=1)
# ax2.plot(df["max_comp"], label="Greatest component", color='red', lw=1)
# ax1.spines['left'].set_color('blue')
# ax2.spines['right'].set_color('red')
# ax1.tick_params(axis='y', colors='blue', which="both")
# ax2.tick_params(axis='y', colors='red', which="both")
# plt.legend(loc=(0.5, 0.2), fontsize=7)
# plt.savefig(r"C:\Users\szmat\Documents\GitHub\ExMAS_sideline\Topology\data\results\19-01-23_full\temp.png")