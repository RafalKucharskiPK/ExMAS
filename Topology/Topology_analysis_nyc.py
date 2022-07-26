import pandas as pd
import multiprocessing as mp
import datetime
from netwulf import visualize
import pickle
import networkx as nx
import json
import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import utils_topology as utils
import NYC_tools.NYC_data_prep_functions as nyc_tools
from ExMAS.main_prob import main as exmas_algo
from ExMAS.utils import make_graph as exmas_make_graph

if __name__ == "__main__":
    """ Load all the topological parameters """
    topological_config = utils.get_parameters('data/configs/topology_settings.json')

    """ Set up varying parameters (optional) """
    # topological_config.variable = 'shared_discount'
    # topological_config.values = [0.22, 0.24]

    """ Run parameters """
    topological_config.replications = 2
    topological_config.no_batches = 1

    """ Prepare folder """
    utils.create_results_directory(topological_config)

    """ Prepare data """
    dotmaps_list, params = nyc_tools.prepare_batches(topological_config.no_batches,
                                                     filter_function=lambda x: len(x.requests) < 150,
                                                     config=topological_config.initial_parameters)

    # ExMAS.utils.plot_demand(dotmaps_list[0], params)

    """ Run ExMAS """
    dotmaps_list_results, settings_list = nyc_tools.run_exmas_nyc_batches(exmas_algo, params, dotmaps_list,
                                                                          noise_generator='wiener',
                                                                          stepwise=False,
                                                                          topo_params=topological_config,
                                                                          replications=topological_config.replications,
                                                                          logger_level='INFO')
    utils.save_with_pickle(dotmaps_list_results, 'dotmap_list', topological_config)

    """ Noise analysis """
    utils.analyse_noise(dotmaps_list_results, topological_config)
    """ Edges storing & counting """
    rep_graphs = utils.analyse_edge_count(dotmaps_list_results, topological_config, list_types_of_graph='all')
    utils.save_with_pickle(rep_graphs, 'rep_graphs', topological_config)

    pool = mp.Pool(mp.cpu_count())
    all_graphs_list = [pool.apply(utils.create_graph, args=(indata, 'all')) for indata in dotmaps_list_results]
    pool.close()
    utils.save_with_pickle(all_graphs_list, 'all_graphs_list', topological_config)

    utils.analysis_all_graphs(all_graphs_list, topological_config)

    # visualize(rep_graphs['pairs_matching'])
    # visualize(utils.create_graph(dotmaps_list_results[0], 'all')['bipartite_matching'])

    """ Perform topological analysis """
    pool = mp.Pool(mp.cpu_count())
    graph_list = [pool.apply(exmas_make_graph, args=(data.sblts.requests, data.sblts.rides)) for data in
                  dotmaps_list_results]
    topological_stats = [utils.GraphStatistics(graph, "INFO") for graph in graph_list]
    topo_dataframes = pool.map(utils.worker_topological_properties, topological_stats)
    pool.close()

    """ Merge results """
    merged_results = utils.merge_results(dotmaps_list_results, topo_dataframes, settings_list)
    merged_file_path = topological_config.path_results + 'merged_files_' + \
                       str(datetime.date.today().strftime("%d-%m-%y")) + '.xlsx'
    merged_results.to_excel(merged_file_path, index=False)

    """ Compute final results """
    variables = ['Batch']
    utils.APosterioriAnalysis(pd.read_excel(merged_file_path),
                              topological_config.path_results,
                              topological_config.path_results + "temp/",
                              variables,
                              topological_config.graph_topological_properties,
                              topological_config.kpis,
                              topological_config.graph_properties_against_inputs,
                              topological_config.dictionary_variables).do_all()
