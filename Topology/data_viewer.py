import pickle
import networkx as nx
import pandas as pd
from netwulf import visualize
import json
import numpy as np
from utils_topology import draw_bipartite_graph
import utils_topology as utils
import matplotlib.pyplot as plt
import datetime
import matplotlib.image as mpimg
import netwulf

# with open('data/results/06-06-22/rep_graphs_06-06-22.obj', 'rb') as file:
#     e = pickle.load(file)

with open('data/results/06-06-22/dotmap_list_06-06-22.obj', 'rb') as file:
    e = pickle.load(file)

# with open('data/results/06-06-22/all_graphs_list_06-06-22.obj', 'rb') as file:
#     e = pickle.load(file)[0]

topological_config = utils.get_parameters('data/configs/topology_settings.json')
# utils.create_results_directory(topological_config)
topological_config.path_results = 'data/results/06-06-22/'

# """ Instead of netwulf """
# G = e['pairs_matching']
#
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
# nx.draw_networkx_nodes(G, pos=nx.spring_layout(G), node_size=5, node_color="black")
# nx.draw_networkx_edges(G, pos=nx.spring_layout(G), width=5, edge_color="black")
# plt.show()
# """ """


# visualize(e['pairs_matching'], config=json.load(open('data/configs/netwulf_config.json')))
# draw_bipartite_graph(e['bipartite_shareability'], 1, topological_config, date='06-06-22', save=True,
#                      name="bipartite_shareability_single", dpi=200, colour_specific_node=0)

num_list = [1, 5, 10, 100, 900]
for num in num_list:
    if num == 1:
        obj = [e[1]]
    else:
        obj = e[:num]
    draw_bipartite_graph(utils.analyse_edge_count(obj, topological_config,
                                                  list_types_of_graph=['bipartite_matching'], logger_level='WARNING')[
                             'bipartite_matching'],
                         num, node_size=1, dpi=80, figsize=(10, 24), plot=False, width_power=1,
                         config=topological_config, save=True, saving_number=num, date='06-06-22')

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

# num_list = list(range(1000))
# df = pd.DataFrame()
# for num in num_list:
#     if num == 0:
#         obj = [e[0]]
#     else:
#         obj = e[:num]
#     temp_graph = utils.analyse_edge_count(obj, topological_config, list_types_of_graph=['pairs_matching'],
#                              logger_level='WARNING')['pairs_matching']
#     t = utils.graph_mini_graphstatistics(temp_graph)
#     temp_df = pd.DataFrame.from_dict({'average_degree': [t.average_degree], 'max_comp': [t.proportion_max_component],
#                                       'number_of_isolated': [t.number_of_isolated_pairs]})
#     df = pd.concat([df, temp_df])
#
# df.reset_index(inplace=True)
# df.drop(columns=['index'], inplace=True)
# df.to_excel(topological_config.path_results + 'frame_evolution_06-06-22.xlsx', index=False)
