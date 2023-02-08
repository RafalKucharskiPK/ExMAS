from Utils import utils_topology as ut

e = ut.read_pickle('data/results/06-06-22/rep_graphs_06-06-22.obj')
topological_config = ut.get_parameters('data/configs/topology_settings.json')

topological_config.path_results = 'data/results/06-06-22/'

# visualize(e['pairs_shareability'], config=json.load(open('data/configs/netwulf_config.json')))

G = e['pairs_matching']

""" Centrality measures """
print(ut.centrality_degree(G, False, 2))

