import visualising_functions as vis_eff
import networkx as nx
import matplotlib.pyplot as plt
import netwulf as nw

date = "25-11-22"
sblts_exmas = "exmas"
special_name = "_full_n" # "_full_n"

if __name__ == '__main__':
    config = vis_eff.config_initialisation('data/configs/topology_settings3.json', date, sblts_exmas)
    config.path_results = 'data/results/' + date + special_name + '/'
    config.date = date

    vis_eff.create_figs_folder(config)

    rep_graphs, dotmap_list, all_graphs_list = vis_eff.load_data(config)

    # vis_eff.graph_visualisation_with_netwulf(all_graphs_list, rep_graphs)
    # vis_eff.kpis_gain(dotmap_list, config, bins=20, y_max=155)
    # vis_eff.probability_of_pooling_classes(dotmap_list, config)
    vis_eff.classes_analysis(dotmap_list, config)
    # vis_eff.aggregated_analysis(dotmap_list, config)
    # vis_eff.analyse_profitability(dotmap_list, config, shared_all='all', bins=20, y_max=155, save_results=True)
    # vis_eff.individual_analysis(dotmap_list, config, s=1)
    # vis_eff.individual_rides_profitability(dotmap_list, config, s=3)
    #
    # vis_eff.visualize_two_shareability_graphs(g1=all_graphs_list[0]["pairs_matching"],
    #                                           g2=all_graphs_list[1]["pairs_matching"],
    #                                           config=config, spec_name="shareability", edge_width=0.7, alpha_diff=0.6,
    #                                           thicker_common=1.5, alpha_common=1, only_netwulf=True)

    e = 0