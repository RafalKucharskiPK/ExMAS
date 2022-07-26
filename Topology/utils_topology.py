import json
from dotmap import DotMap
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import collections
import logging
import sys
import seaborn as sns
import datetime
import os
import math
from scipy.stats import norm
from tqdm import tqdm
from collections import Counter
import pickle


def get_parameters(path, time_correction=False):
    with open(path) as json_file:
        data = json.load(json_file)
        config = DotMap(data)
    if time_correction:
        config['t0'] = pd.Timestamp('15:00')

    return config


def init_log(logger_level, logger=None):
    if logger_level == 'DEBUG':
        level = logging.DEBUG
    elif logger_level == 'WARNING':
        level = logging.WARNING
    elif logger_level == 'CRITICAL':
        level = logging.CRITICAL
    elif logger_level == 'INFO':
        level = logging.INFO
    else:
        raise Exception("Not accepted logger level, please choose: 'DEBUG', 'WARNING', 'CRITICAL', 'INFO'")
    if logger is None:
        logging.basicConfig(stream=sys.stdout, format='%(asctime)s-%(levelname)s-%(message)s',
                            datefmt='%H:%M:%S', level=level)

        logger = logging.getLogger()

        logger.setLevel(level)
        return logging.getLogger(__name__)
    else:
        logger.setLevel(level)
        return logger


class GraphStatistics:
    def __init__(self, graph, logging_level="INFO"):
        self.logger = init_log(logging_level)
        self.G = graph
        self.connected = nx.is_connected(self.G)
        self.bipartite = nx.is_bipartite(self.G)
        self.average_degree = None
        self.maximum_degree = None
        self.average_clustering_coefficient = None
        self.average_clustering_group0 = None
        self.average_clustering_group1 = None
        self.components = None
        self.proportion_max_component = None
        self.num_nodes_group0 = None
        self.num_nodes_group1 = None
        self.average_degree_group0 = None
        self.average_degree_group1 = None
        self.number_of_isolated_pairs = None
        self.average_clustering_group0_reduced = None
        self.average_clustering_group1_reduced = None
        # Objects to be stored rather than strict output
        self.group0_colour = None
        self.group1_colour = None
        self.reduced_graph = None
        self.group0_colour_reduced = None
        self.group1_colour_reduced = None

    def initial_analysis(self):
        self.logger.info('Graph is connected: {}'.format(self.connected))
        self.logger.info('Graph is bipartite: {}'.format(self.bipartite))
        self.logger.info('Number of nodes: {}'.format(self.G.number_of_nodes()))
        self.logger.info('Number of edges: {}'.format(self.G.number_of_edges()))
        if self.bipartite:
            self.colouring_graph()

    def colouring_graph(self):
        if self.bipartite:
            partition_for_bipartite = nx.bipartite.basic.color(self.G)
            for colour_key in partition_for_bipartite.keys():
                self.G.nodes[colour_key]['bipartite'] = partition_for_bipartite[colour_key]
            total_colouring = {k: v['bipartite'] for k, v in self.G._node.copy().items()}
            self.group0_colour = {k: v for k, v in total_colouring.items() if v == 0}
            self.group1_colour = {k: v for k, v in total_colouring.items() if v == 1}
            # Group 0 shall be longer
            if len(self.group0_colour) > len(self.group1_colour):
                pass
            else:
                self.group0_colour, self.group1_colour = self.group1_colour, self.group0_colour

            # Additional analysis removing rides of degree 1
            remove_bc_of_degree = [node for node, degree in dict(self.G.degree()).items() if degree == 1]
            remove_only_from_group0 = [node for node in remove_bc_of_degree if node in self.group0_colour.keys()]
            self.reduced_graph = self.G.copy()
            self.reduced_graph.remove_nodes_from(remove_only_from_group0)
            partition_for_bipartite = nx.bipartite.basic.color(self.reduced_graph)
            for colour_key in partition_for_bipartite.keys():
                self.reduced_graph.nodes[colour_key]['bipartite'] = partition_for_bipartite[colour_key]
            total_colouring = {k: v['bipartite'] for k, v in self.reduced_graph._node.copy().items()}
            self.group0_colour_reduced = {k: v for k, v in total_colouring.items() if v == 0}
            self.group1_colour_reduced = {k: v for k, v in total_colouring.items() if v == 1}
        else:
            pass

    def degree_distribution(self, degree_histogram=False, degree_cdf=False):
        degree_sequence = sorted([d for n, d in self.G.degree()], reverse=False)
        degree_counter = collections.Counter(degree_sequence)
        deg, cnt = zip(*degree_counter.items())
        self.average_degree = np.sum(np.multiply(deg, cnt)) / self.G.number_of_nodes()
        self.maximum_degree = max(degree_sequence)
        self.logger.info('Average degree: {}'.format(self.average_degree))
        self.logger.info('Maximum degree: {}'.format(self.maximum_degree))

        if self.bipartite:
            degrees = dict(self.G.degree())
            group0 = {k: v for k, v in degrees.items() if k in self.group0_colour.keys()}
            group1 = {k: v for k, v in degrees.items() if k in self.group1_colour.keys()}
            self.average_degree_group0 = sum(group0.values()) / len(group0)
            self.average_degree_group1 = sum(group1.values()) / len(group1)

        if degree_histogram:
            plt.bar(*np.unique(degree_sequence, return_counts=True))
            plt.title("Degree histogram")
            plt.xlabel("Degree")
            plt.ylabel("# of Nodes")
            plt.show()

        if degree_cdf:
            cs = np.cumsum(cnt)
            n = len(degree_sequence)
            plt.style.use('seaborn-whitegrid')
            plt.plot(sorted(deg), cs / n, 'bo', linestyle='-', linewidth=1.2, markersize=2.5)
            plt.title("True Cumulative Distribution plot")
            plt.axhline(y=0.9, color='r', linestyle='dotted', alpha=0.5, label='0.9')
            plt.ylabel("P(k>=Degree)")
            plt.xlabel("Degree")
            plt.xlim(0, max(degree_sequence))
            plt.ylim((cs / n)[0], 1.05)
            plt.show()

    def nodes_per_colour(self):
        if self.bipartite:
            self.num_nodes_group0 = len(self.group0_colour)
            self.num_nodes_group1 = len(self.group1_colour)
        else:
            pass

    def clustering_coefficient(self, detailed=False):
        if not self.bipartite:
            self.logger.info('The graph is not bipartite, hence the clustering coefficient is based on triangles.')
            self.average_clustering_coefficient = nx.average_clustering(self.G)
            self.logger.info(
                "Graph's average clustering coefficient is {}.".format(self.average_clustering_coefficient))
            if detailed:
                self.logger.info('Clustering coefficients per node: {}\n'.format(nx.clustering(self.G)))
                self.logger.info('Transitivity per node: {}\n'.format(nx.transitivity(self.G)))
                self.logger.info('Triangles per node: {}\n'.format(nx.triangles(self.G)))
        else:
            self.logger.info('The graph is bipartite, hence the clustering coefficient in based on squares.')
            sq_coefficient = nx.square_clustering(self.G)
            group0 = {k: v for k, v in sq_coefficient.items() if k in self.group0_colour.keys()}
            group1 = {k: v for k, v in sq_coefficient.items() if k in self.group1_colour.keys()}
            if len(sq_coefficient) != 0:
                self.average_clustering_coefficient = sum(sq_coefficient.values()) / len(sq_coefficient)
            else:
                self.average_clustering_coefficient = 0
            self.average_clustering_group0 = sum(group0.values()) / len(group0)
            self.average_clustering_group1 = sum(group1.values()) / len(group1)
            self.logger.info('Average clustering coefficient: ' + str(self.average_clustering_coefficient))
            self.logger.info('Average clustering coefficient in group 0: ' + str(self.average_clustering_group0))
            self.logger.info('Average clustering coefficient in group 1: ' + str(self.average_clustering_group1))

            # Reduced graphs by nodes in group1 whose degree is equal to 1
            sq_coefficient = nx.square_clustering(self.reduced_graph)
            group0 = {k: v for k, v in sq_coefficient.items() if k in self.group0_colour_reduced.keys()}
            group1 = {k: v for k, v in sq_coefficient.items() if k in self.group1_colour_reduced.keys()}
            if len(group0) != 0:
                self.average_clustering_group0_reduced = sum(group0.values()) / len(group0)
            else:
                self.average_clustering_group0_reduced = 0
            if len(group1) != 0:
                self.average_clustering_group1_reduced = sum(group1.values()) / len(group1)
            else:
                self.average_clustering_group1_reduced = 0

    def component_analysis(self, plot=False):
        g_components = list(nx.connected_components(self.G))
        g_components.sort(key=len, reverse=True)
        self.components = g_components
        self.logger.info('Number of connected components: {}'.format(len(self.components)))
        self.logger.info('Sizes of the components: ' + str([len(i) for i in self.components]))
        self.proportion_max_component = len(self.components[0]) / self.G.number_of_nodes()
        self.number_of_isolated_pairs = sum(1 if x == 2 else 0 for x in [len(i) for i in self.components])
        if plot:
            plt.style.use('seaborn-whitegrid')
            plt.bar(range(len(self.components)), [len(i) for i in self.components])
            plt.title("Sorted sizes of connected components")
            plt.ylabel("No. of nodes")
            plt.xlabel("Component's ID")
            plt.xticks(range(len(self.components)))
            plt.show()

    def all_analysis(self, degree_distribution=False, degree_cdf=False, detailed_clustering=False,
                     plot_components=False):
        GraphStatistics.initial_analysis(self)
        GraphStatistics.colouring_graph(self)
        GraphStatistics.nodes_per_colour(self)
        GraphStatistics.degree_distribution(self, degree_distribution, degree_cdf)
        GraphStatistics.clustering_coefficient(self, detailed_clustering)
        GraphStatistics.component_analysis(self, plot_components)


def worker_topological_properties(GraphStatObj):
    data_output = pd.DataFrame()
    GraphStatObj.all_analysis()
    if GraphStatObj.bipartite:
        data_output = data_output.append([GraphStatObj.num_nodes_group0, GraphStatObj.num_nodes_group1,
                                          GraphStatObj.average_degree, GraphStatObj.maximum_degree,
                                          GraphStatObj.average_degree_group0,
                                          GraphStatObj.average_degree_group1,
                                          GraphStatObj.average_clustering_coefficient,
                                          GraphStatObj.proportion_max_component, len(GraphStatObj.components),
                                          GraphStatObj.average_clustering_group0,
                                          GraphStatObj.average_clustering_group1,
                                          GraphStatObj.number_of_isolated_pairs,
                                          GraphStatObj.average_clustering_group0_reduced,
                                          GraphStatObj.average_clustering_group1_reduced])
        data_output.index = ['No_nodes_group0', 'No_nodes_group1', 'Average_degree',
                             'Maximum_degree', 'Average_degree_group0', 'Average_degree_group1',
                             'Avg_clustering',
                             'Proportion_max_component', 'No_components', 'Average_clustering_group0',
                             'Average_clustering_group1', 'No_isolated_pairs',
                             'Average_clustering_group0_reduced',
                             'Average_clustering_group1_reduced']
    else:
        data_output = data_output.append([GraphStatObj.average_degree, GraphStatObj.maximum_degree,
                                          GraphStatObj.average_clustering_coefficient,
                                          GraphStatObj.proportion_max_component,
                                          len(GraphStatObj.components)])
        data_output.index = ['Average_degree', 'Maximum_degree', 'Avg_clustering',
                             'Proportion_max_component', 'No. of components']

    return data_output


def alternate_kpis(dataset):
    if 'nP' in dataset.columns:
        pass
    else:
        dataset['nP'] = dataset['No_nodes_group1']

    dataset['Proportion_singles'] = dataset['SINGLE'] / dataset['nP']
    dataset['Proportion_pairs'] = dataset['PAIRS'] / dataset['nP']
    dataset['Proportion_triples'] = dataset['TRIPLES'] / dataset['nP']
    dataset['Proportion_triples_plus'] = (dataset['nP'] - dataset['SINGLE'] -
                                          dataset['PAIRS']) / dataset['nP']
    dataset['Proportion_quadruples'] = dataset['QUADRIPLES'] / dataset['nP']
    dataset['Proportion_quintets'] = dataset['QUINTETS'] / dataset['nP']
    dataset['Proportion_six_plus'] = dataset['PLUS5'] / dataset['nP']
    dataset['SavedVehHours'] = (dataset['VehHourTrav_ns'] - dataset['VehHourTrav']) / \
                               dataset['VehHourTrav_ns']
    dataset['AddedPasHours'] = (dataset['PassHourTrav'] - dataset['PassHourTrav_ns']) / \
                               dataset['PassHourTrav_ns']
    dataset['UtilityGained'] = (dataset['PassUtility'] - dataset['PassUtility_ns']) / \
                               dataset['PassUtility_ns']
    dataset['Fraction_isolated'] = dataset['No_isolated_pairs'] / dataset['nP']
    return dataset


def amend_merged_file(merged_file, alter_kpis=False, inplace=True):
    if not inplace:
        merged_file = merged_file.copy(deep=True)
    merged_file.drop(columns=['_typ', 'dtype'], inplace=True)
    merged_file.reset_index(inplace=True, drop=True)
    if alter_kpis:
        merged_file = alternate_kpis(merged_file)
    return merged_file


def merge_results(dotmaps_list_results, topo_dataframes, settings_list, logger_level="INFO"):
    logger = init_log(logger_level)
    logger.warning("Merging results")
    res = [pd.concat([z, x, y.sblts.res]) for z, x, y in
           zip([pd.Series(k) for k in settings_list], topo_dataframes, dotmaps_list_results)]
    merged_file = pd.DataFrame()
    for item in res:
        merged_file = pd.concat([merged_file, item.T])
    amend_merged_file(merged_file)
    logger.warning("Results merged")
    return merged_file


class APosterioriAnalysis:
    def __init__(self, dataset: pd.DataFrame, output_path: str, output_temp: str, input_variables: list,
                 all_graph_properties: list, kpis: list, graph_properties_to_plot: list, labels: dict,
                 err_style: str = "band", date: str = '000'):
        """
        Class designed to performed analysis on merged results from shareability graph properties.
        :param dataset: input merged datasets from replications
        :param output_path: output for final results
        :param output_temp: output for temporal results required in the process
        :param input_variables: search space variables
        :param all_graph_properties: all graph properties for heatmap/correlation analysis
        :param kpis: final matching coefficients to take into account
        :param graph_properties_to_plot: properties of graph to be plotted
        :param labels: dictionary of labels
        :param err_style: for line plots style of the error
        """
        self.dataset = dataset.drop(columns=['Replication_ID'])
        self.input_variables = input_variables
        self.all_graph_properties = all_graph_properties
        self.dataset_grouped = self.dataset.groupby(self.input_variables)
        self.output_path = output_path
        self.output_temp = output_temp
        self.kpis = kpis
        self.graph_properties_to_plot = graph_properties_to_plot
        self.labels = labels
        self.err_style = err_style
        self.heatmap = None
        self.date = date

    def alternate_kpis(self):
        if 'nP' in self.dataset.columns:
            pass
        else:
            self.dataset['nP'] = self.dataset['No_nodes_group1']

        self.dataset['Proportion_singles'] = self.dataset['SINGLE'] / self.dataset['nR']
        self.dataset['Proportion_pairs'] = self.dataset['PAIRS'] / self.dataset['nR']
        self.dataset['Proportion_triples'] = self.dataset['TRIPLES'] / self.dataset['nR']
        self.dataset['Proportion_triples_plus'] = (self.dataset['nR'] - self.dataset['SINGLE'] -
                                                   self.dataset['PAIRS']) / self.dataset['nR']
        self.dataset['Proportion_quadruples'] = self.dataset['QUADRIPLES'] / self.dataset['nR']
        self.dataset['Proportion_quintets'] = self.dataset['QUINTETS'] / self.dataset['nR']
        self.dataset['Proportion_six_plus'] = self.dataset['PLUS5'] / self.dataset['nR']
        self.dataset['SavedVehHours'] = (self.dataset['VehHourTrav_ns'] - self.dataset['VehHourTrav']) / \
                                        self.dataset['VehHourTrav_ns']
        self.dataset['AddedPasHours'] = (self.dataset['PassHourTrav'] - self.dataset['PassHourTrav_ns']) / \
                                        self.dataset['PassHourTrav_ns']
        self.dataset['UtilityGained'] = (self.dataset['PassUtility'] - self.dataset['PassUtility_ns']) / \
                                        self.dataset['PassUtility_ns']
        self.dataset['Fraction_isolated'] = self.dataset['No_isolated_pairs'] / self.dataset['nP']
        self.dataset_grouped = self.dataset.groupby(self.input_variables)

    def boxplot_inputs(self):
        for counter, y_axis in enumerate(self.all_graph_properties):
            dataset = self.dataset.copy()
            if len(self.input_variables) <= 2:
                if len(self.input_variables) == 1:
                    sns.boxplot(x=self.input_variables[0], y=y_axis, data=dataset) \
                        .set(xlabel=self.labels[self.input_variables[0]], ylabel=self.labels[y_axis])
                elif len(self.input_variables) == 2:
                    temp_dataset = dataset.copy()
                    temp_dataset[self.labels[self.input_variables[1]]] = temp_dataset[self.input_variables[1]]
                    sns.boxplot(x=self.input_variables[0], y=y_axis, data=temp_dataset,
                                hue=self.labels[self.input_variables[1]]) \
                        .set(xlabel=self.labels[self.input_variables[0]], ylabel=self.labels[y_axis])
                else:
                    break
                plt.savefig(self.output_temp + 'temp_boxplot_' + str(counter) + '.png')
                plt.close()
            else:
                raise Exception('Grouping variables number is:', len(self.input_variables), ' - too long for boxplot.')

    def line_plot_inputs(self):
        for counter, x_axis in enumerate(self.input_variables):
            if len(self.graph_properties_to_plot) <= 2:
                if len(self.input_variables) == 1:
                    sns.lineplot(x=x_axis, y=self.graph_properties_to_plot[0], data=self.dataset)
                    plt.xlabel(self.labels[x_axis])
                    plt.ylabel(self.labels[self.graph_properties_to_plot[0]], color='b')
                elif len(self.input_variables) == 2:
                    fig, ax1 = plt.subplots()
                    ax2 = ax1.twinx()
                    sns.lineplot(x=x_axis, y=self.graph_properties_to_plot[0], data=self.dataset, color='b', ax=ax1,
                                 err_style=self.err_style)
                    sns.lineplot(x=x_axis, y=self.graph_properties_to_plot[1], data=self.dataset, color='r', ax=ax2,
                                 err_style=self.err_style)
                    ax1.set_xlabel(self.labels[x_axis])
                    ax1.set_ylabel(self.labels[self.graph_properties_to_plot[0]], color='b')
                    ax2.set_ylabel(self.labels[self.graph_properties_to_plot[1]], color='r')
                else:
                    pass
                plt.savefig(self.output_temp + 'temp_lineplot_' + str(counter) + '.png')
                plt.close()
            else:
                raise Exception('Grouping variables number is:', len(self.input_variables), ' - too long for lineplot.')

    def plot_kpis_properties(self):
        plot_arguments = [(x, y) for x in self.graph_properties_to_plot for y in self.kpis]
        dataset = self.dataset.copy()
        binning = False
        for counter, value in enumerate(self.input_variables):
            min_val = min(self.dataset[value])
            max_val = max(self.dataset[value])
            if min_val == 0 and max_val == 0:
                binning = False
            else:
                step = (max_val - min_val) / 3
                if min_val < 5:
                    bins = np.round(np.append(np.arange(min_val * 0.98, max_val * 1.02, step), [max_val + step]), 3)
                else:
                    bins = np.round(np.append(np.arange(min_val * 0.98, max_val * 1.02, step), [max_val + step]), 0)
                labels = [f'{i}+' if j == np.inf else f'{i}-{j}' for i, j in
                          zip(bins, bins[1:])]  # additional part with infinity
                dataset[self.labels[value] + " bin"] = pd.cut(dataset[value], bins, labels=labels)
                binning = True

        for counter, j in enumerate(plot_arguments):
            if not binning:
                fig, ax = plt.subplots()
                sns.scatterplot(x=j[0], y=j[1], data=dataset, palette="crest")
                ax.set_xlabel(self.labels[j[0]])
                ax.set_ylabel(self.labels[j[1]])
                plt.savefig(self.output_temp + 'kpis_properties_' + str(counter) + '.png')
                plt.close()
            else:
                if len(self.input_variables) == 1:
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=j[0], y=j[1], data=dataset,
                                    hue=dataset[self.labels[self.input_variables[0]] + " bin"], palette="crest")
                    ax.set_xlabel(self.labels[j[0]])
                    ax.set_ylabel(self.labels[j[1]])
                    plt.savefig(self.output_temp + 'kpis_properties_' + str(counter) + '.png')
                    plt.close()
                elif len(self.input_variables) == 2:
                    fix, ax = plt.subplots()
                    sns.scatterplot(x=j[0], y=j[1], data=dataset,
                                    hue=dataset[self.labels[self.input_variables[0]] + " bin"],
                                    size=dataset[self.labels[self.input_variables[1]] + " bin"], palette="crest")
                    ax.set_xlabel(self.labels[j[0]])
                    ax.set_ylabel(self.labels[j[1]])
                    plt.savefig(self.output_temp + 'kpis_properties_' + str(counter) + '.png')
                    plt.close()
                else:
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=j[0], y=j[1], data=dataset, palette="crest")
                    ax.set_xlabel(self.labels[j[0]])
                    ax.set_ylabel(self.labels[j[1]])
                    plt.savefig(self.output_temp + 'kpis_properties_' + str(counter) + '.png')
                    plt.close()

    def create_heatmap(self):
        df = self.dataset[self.all_graph_properties + self.kpis]
        for column in df.columns:
            df.rename(columns={column: self.labels[column]}, inplace=True)

        corr = df.corr()
        self.heatmap = corr
        plt.figure(figsize=(10, 10))
        sns.heatmap(corr,
                    xticklabels=corr.columns,
                    yticklabels=corr.columns,
                    linewidths=.5,
                    annot=True,
                    center=0,
                    cmap='OrRd')
        plt.subplots_adjust(bottom=0.3, left=0.3)
        plt.savefig(self.output_temp + 'heatmap' + '.png')
        plt.close()
        self.heatmap = round(self.heatmap, 3).style.background_gradient(cmap='coolwarm').set_precision(2)

    def save_grouped_results(self):
        if self.date == '000':
            date = str(datetime.date.today().strftime("%d-%m-%y"))
        writer = pd.ExcelWriter(self.output_path + 'Final_results_' + '_'.join(self.input_variables) + '_' +
                                self.date + '.xlsx', engine='xlsxwriter')
        self.dataset_grouped.min().to_excel(writer, sheet_name='Min')
        self.dataset_grouped.mean().to_excel(writer, sheet_name='Mean')
        self.dataset_grouped.max().to_excel(writer, sheet_name='Max')
        workbook = writer.book

        worksheet = workbook.add_worksheet('Boxplots')
        for counter in range(len(self.all_graph_properties)):
            worksheet.insert_image('B' + str(counter * 25 + 1),
                                   self.output_temp + 'temp_boxplot_' + str(counter) + '.png')

        worksheet = workbook.add_worksheet('Lineplots')
        for counter in range(len(self.graph_properties_to_plot)):
            worksheet.insert_image('B' + str(counter * 25 + 1),
                                   self.output_temp + 'temp_lineplot_' + str(counter) + '.png')

        worksheet = workbook.add_worksheet('KpiPlots')
        for counter in range(len(self.graph_properties_to_plot) * len(self.kpis)):
            worksheet.insert_image('B' + str(counter * 25 + 1),
                                   self.output_temp + 'kpis_properties_' + str(counter) + '.png')

        self.heatmap.to_excel(writer, sheet_name='Correlation')
        worksheet = workbook.get_worksheet_by_name('Correlation')
        worksheet.insert_image('B' + str(len(self.all_graph_properties) * 2 + 5), self.output_temp + 'heatmap' + '.png')
        writer.save()

    def do_all(self):
        self.alternate_kpis()
        self.line_plot_inputs()
        self.boxplot_inputs()
        self.plot_kpis_properties()
        self.create_heatmap()
        self.save_grouped_results()


def analyse_noise(list_dotmaps, config, logger_level="INFO", date=None):
    logger = init_log(logger_level)
    logger.info("Analysing noise")
    df = pd.DataFrame()
    df['Passenger_ID'] = list(range(len(list_dotmaps[0].sblts.requests)))

    def foo(df, num):
        out_list = []
        for j in range(num):
            num = len(df[(df['i'] == j) | (df['j'] == j)])
            out_list.append(num)
        return out_list

    for num, dmap in enumerate(list_dotmaps):
        df['Noise ' + str(num)] = dmap.prob.noise
        df['Possible pairs ' + str(num)] = foo(dmap.sblts.pairs, num=len(dmap.sblts.requests))

    if date is None:
        date = str(datetime.date.today().strftime("%d-%m-%y"))
    else:
        date = str(date)

    df.to_excel(config.path_results + 'noise_analysis_' + str(datetime.date.today().strftime("%d-%m-%y")) + '.xlsx')
    logger.info("Noise analysed")
    return df


def analyse_edge_count(list_dotmaps, config, list_types_of_graph=None, logger_level="INFO", save=True):
    logger = init_log(logger_level)
    logger.info("Analysing edges")
    shareable = []
    logger.info("Counting shareability")
    pbar = tqdm(total=len(list_dotmaps))
    for indata in list_dotmaps:
        # shareable.extend(np.unique(np.array(indata.sblts.rides.indexes)))
        shareable.extend([str(x) for x in np.unique(np.array(indata.sblts.rides.indexes))])
        pbar.update(1)

    my_dict = Counter(shareable)

    # my_dict = {tuple(i): shareable.count(i) for i in shareable}
    my_dict = {tuple(eval(i[0])): i[1] for i in my_dict.items()}
    logger.info("Shareability counted")
    shareability_edges = my_dict.copy()

    if save:
        json_save = {str(key): my_dict[key] for key in my_dict.keys()}
        a_file = open(config.path_results + "shareable_" + str(datetime.date.today().strftime("%d-%m-%y")) + ".json",
                      "w")
        json.dump(json_save, a_file)
        a_file.close()

    logger.info("Counting matched")
    scheduled = []
    pbar = tqdm(total=len(list_dotmaps))
    for indata in list_dotmaps:
        # scheduled.extend(np.unique(np.array(indata.sblts.schedule.indexes)))
        scheduled.extend([str(x) for x in np.unique(np.array(indata.sblts.schedule.indexes))])
        pbar.update(1)

    my_dict = Counter(scheduled)

    # my_dict = {tuple(i): shareable.count(i) for i in shareable}
    my_dict = {tuple(eval(i[0])): i[1] for i in my_dict.items()}
    logger.info("Matched counted")
    matching_edges = my_dict.copy()

    if save:
        json_save = {str(key): my_dict[key] for key in my_dict.keys()}
        a_file = open(config.path_results + "final_matching_" +
                      str(datetime.date.today().strftime("%d-%m-%y")) + ".json", "w")
        json.dump(json_save, a_file)
        a_file.close()

    logger.info("Edges analysed")

    if list_types_of_graph is not None:
        logger.warning("Current implementation for graph creation assumes that all runs are on the same batch")
        indata = list_dotmaps[-1]
        requests = indata.sblts.requests.copy()
        graph_list = dict()
        if list_types_of_graph == 'all':
            list_types_of_graph = ['bipartite_shareability', 'bipartite_matching', 'pairs_shareability',
                                   'pairs_matching']
        pbar = tqdm(total=2 * len(list_types_of_graph))
        while len(list_types_of_graph) > 0:
            type_of_graph = list_types_of_graph[-1]
            if type_of_graph in ['bipartite_shareability', 'bipartite_matching']:
                pbar.update(1)
                bipartite_graph = nx.Graph()
                bipartite_graph.add_nodes_from(requests.index, bipartite=1)
                if type_of_graph == 'bipartite_shareability':
                    edge_dict = shareability_edges.copy()
                else:
                    edge_dict = matching_edges.copy()
                bipartite_graph.add_nodes_from([(x) for x in edge_dict.keys()], bipartite=0)
                edges = []
                for ride in edge_dict.keys():
                    for traveler in ride:
                        edges.append((traveler, ride, {'weight': edge_dict[ride]}))
                bipartite_graph.add_edges_from(edges)
                graph_list[type_of_graph] = bipartite_graph.copy()
                pbar.update(1)

            if type_of_graph in ['pairs_shareability', 'pairs_matching']:
                pbar.update(1)
                pairs_graph = nx.Graph()
                pairs_graph.add_nodes_from(requests.index)
                if type_of_graph == 'pairs_shareability':
                    edge_dict = shareability_edges.copy()
                else:
                    edge_dict = matching_edges.copy()

                edges = []
                for ride in edge_dict.keys():
                    if len(ride) == 2:
                        ride = (ride[0], ride[1], {'weight': edge_dict[ride]})
                        edges.append(ride)
                pairs_graph.add_edges_from(edges)
                graph_list[type_of_graph] = pairs_graph.copy()
                pbar.update(1)

            list_types_of_graph.pop()

        logger.info('Graph list created')
        return graph_list


def create_results_directory(topological_config):
    today = str(datetime.date.today().strftime("%d-%m-%y"))
    topological_config.path_results += today
    try:
        os.mkdir(topological_config.path_results)
    except OSError as error:
        print(error)
        print('overwriting current files in the folder')
    try:
        os.mkdir(os.path.join(topological_config.path_results, 'temp'))
    except OSError as error:
        print(error)
        print('temp folder already exists')
    topological_config.path_results += '/'


def create_graph(indata, list_types_of_graph):
    if list_types_of_graph == 'all':
        list_types_of_graph = ['bipartite_shareability', 'bipartite_matching', 'pairs_shareability',
                               'pairs_matching', 'probability_pairs']
    list_types_of_graph = list(map(lambda x: x.lower(), list_types_of_graph))
    graph_list = dict()
    requests = indata.sblts.requests.copy()
    rides = indata.sblts.rides.copy()
    schedule = indata.sblts.schedule.copy()
    while len(list_types_of_graph) > 0:
        type_of_graph = list_types_of_graph[-1]
        if type_of_graph in ['bipartite_shareability', 'bipartite_matching']:
            bipartite_graph = nx.Graph()
            shift = 10 ** (round(math.log10(requests.shape[0])) + 1)
            if type_of_graph == 'bipartite_shareability':
                _rides = rides.copy()
            else:
                _rides = schedule.copy()
            _rides.index = _rides.index + shift

            bipartite_graph.add_nodes_from(requests.index, bipartite=1)
            bipartite_graph.add_nodes_from(_rides.index, bipartite=0)

            edges = list()
            for i, row in _rides.iterrows():
                for j, pax in enumerate(row.indexes):
                    edges.append((i, pax, {'u': row.u_paxes[j], 'true_u': row.true_u_paxes[j]}))

            bipartite_graph.add_edges_from(edges)
            graph_list[type_of_graph] = bipartite_graph.copy()

        if type_of_graph in ['pairs_shareability', 'pairs_matching']:
            pairs_graph = nx.Graph()
            pairs_graph.add_nodes_from(requests.index)
            edges = list()
            if type_of_graph == 'pairs_shareability':
                _rides = rides.copy()
            else:
                _rides = schedule.copy()
            for i, row in _rides.iterrows():
                if len(row.indexes) > 1:
                    for j, pax1 in enumerate(row.indexes):
                        for k, pax2 in enumerate(row.indexes):
                            if pax1 != pax2:
                                edges.append((pax1, pax2, {'u': row.u_pax, 'true_u': row.true_u_pax,
                                                           'u_paxes': row.true_u_paxes,
                                                           'true_u_paxes': row.true_u_paxes}))

            pairs_graph.add_edges_from(edges)
            graph_list[type_of_graph] = pairs_graph.copy()

        if type_of_graph == 'probability_pairs':
            prob_graph = nx.Graph()
            prob_graph.add_nodes_from(requests.index)
            edges = list()
            _rides = rides.copy()
            for i, row in _rides.iterrows():
                if len(row.indexes) > 1:
                    for j, pax1 in enumerate(row.indexes):
                        for k, pax2 in enumerate(row.indexes):
                            if pax1 != pax2:
                                prob = norm.cdf(row.true_u_paxes[0]) * norm.cdf(row.true_u_paxes[1])
                                edges.append((pax1, pax2, {'weight': prob}))
            prob_graph.add_edges_from(edges)
            graph_list[type_of_graph] = prob_graph

        list_types_of_graph.pop()

    return graph_list


def draw_bipartite_graph(graph, max_weight, config=None, save=False, saving_number=0, width_power=1,
                         figsize=(5, 12), dpi=100, node_size=1, batch_size=147, plot=True, date=None,
                         default_edge_size=1, name=None, colour_specific_node=None):
    # G1 = nx.convert_node_labels_to_integers(graph)
    G1 = graph
    x = G1.nodes._nodes
    l = []
    r = []
    for i in x:
        j = x[i]
        if j['bipartite'] == 1:
            l.append(i)
        else:
            r.append(i)

    if nx.is_weighted(G1):
        dict_weights = {tuple(edge_data[:-1]): edge_data[-1]["weight"] for edge_data in G1.edges(data=True)}
        r_weighted = {v[-1]: dict_weights[v] for v in dict_weights.keys() if v[-1] in r}
        r_weighted_sorted = {k: v for k, v in sorted(r_weighted.items(), key=lambda x: x[1])}
        r = list(r_weighted_sorted.keys())

    colour_list = len(l) * ['g'] + len(r) * ['b']

    pos = nx.bipartite_layout(G1, l)

    new_pos = dict()
    for num, key in enumerate(pos.keys()):
        if num <= batch_size - 1:
            new_pos[key] = pos[key]
        else:
            new_pos[r[num - batch_size]] = pos[key]

    if colour_specific_node is not None:
        assert isinstance(colour_specific_node, int), "Passed node number is not an integer"
        colour_list[colour_specific_node] = "r"

    plt.figure(figsize=figsize, dpi=dpi)

    nx.draw_networkx_nodes(G1, pos=new_pos, node_color=colour_list, node_size=node_size)

    if nx.is_weighted(G1):
        for weight in range(1, max_weight + 1):
            edge_list = [(u, v) for (u, v, d) in G1.edges(data=True) if d["weight"] == weight]
            nx.draw_networkx_edges(G1, new_pos, edgelist=edge_list,
                                   width=default_edge_size * np.power(weight, width_power)
                                         / np.power(max_weight, width_power))
    else:
        if colour_specific_node is None:
            nx.draw_networkx_edges(G1, new_pos, edgelist=G1.edges, width=default_edge_size)
        else:
            assert isinstance(colour_specific_node, int), "Passed node number is not an integer"
            colour_list = []
            for item in G1.edges:
                if item[0] == colour_specific_node or item[1] == colour_specific_node:
                    colour_list.append("red")
                else:
                    colour_list.append("black")
            nx.draw_networkx_edges(G1, new_pos, edgelist=G1.edges, width=default_edge_size/5, edge_color=colour_list)


    if save:
        if date is None:
            date = str(datetime.date.today().strftime("%d-%m-%y"))
        else:
            date = str(date)
        if name is None:
            plt.savefig(config.path_results + "temp/graph_" + date + "_no_" + str(saving_number) + ".png")
        else:
            plt.savefig(config.path_results + "temp/" + name + ".png")
    if plot:
        plt.show()


def graph_mini_graphstatistics(graph):
    z = GraphStatistics(graph, logging_level='WARNING')
    z.initial_analysis()
    z.colouring_graph()
    z.nodes_per_colour()
    z.degree_distribution()
    z.component_analysis()
    return z


def concat_all_graph_list(list_of_all_graphs):
    pairs_matching = []
    pairs_shareability = []
    bipartite_matching = []
    bipartite_shareability = []

    for rep_no in list_of_all_graphs:
        graph_temp = graph_mini_graphstatistics(rep_no['bipartite_shareability'])
        bipartite_shareability.append((len(graph_temp.G.nodes), len(graph_temp.G.edges), graph_temp.average_degree,
                                       graph_temp.average_degree_group0, graph_temp.average_degree_group1,
                                       graph_temp.proportion_max_component))
        graph_temp = graph_mini_graphstatistics(rep_no['bipartite_matching'])
        bipartite_matching.append((len(graph_temp.G.nodes), len(graph_temp.G.edges), graph_temp.average_degree,
                                   graph_temp.average_degree_group0, graph_temp.average_degree_group1,
                                   graph_temp.proportion_max_component))
        graph_temp = graph_mini_graphstatistics(rep_no['pairs_shareability'])
        pairs_shareability.append((len(graph_temp.G.nodes), len(graph_temp.G.edges), graph_temp.average_degree,
                                   graph_temp.proportion_max_component))
        graph_temp = graph_mini_graphstatistics(rep_no['pairs_matching'])
        pairs_matching.append((len(graph_temp.G.nodes), len(graph_temp.G.edges), graph_temp.average_degree,
                               graph_temp.proportion_max_component))

    return {'bipartite_shareability': bipartite_shareability, 'bipartite_matching': bipartite_matching,
            'pairs_shareability': pairs_shareability, 'pairs_matching': pairs_matching}


def analyse_concatenated_all_graph_list(concatenated_list):
    def func(num, bipartite_only=False):
        if not bipartite_only:
            return (([np.mean([x[num] for x in concatenated_list['bipartite_shareability']]),
                      np.mean([x[num] for x in concatenated_list['bipartite_matching']]),
                      np.mean([x[num] for x in concatenated_list['pairs_shareability']]),
                      np.mean([x[num] for x in concatenated_list['pairs_matching']])]),
                    ([np.std([x[num] for x in concatenated_list['bipartite_shareability']]),
                      np.std([x[num] for x in concatenated_list['bipartite_matching']]),
                      np.std([x[num] for x in concatenated_list['pairs_shareability']]),
                      np.std([x[num] for x in concatenated_list['pairs_matching']])]))
        else:
            return (([np.mean([x[num] for x in concatenated_list['bipartite_shareability']]),
                      np.mean([x[num] for x in concatenated_list['bipartite_matching']]),
                      0,
                      0]),
                    ([np.std([x[num] for x in concatenated_list['bipartite_shareability']]),
                      np.std([x[num] for x in concatenated_list['bipartite_matching']]),
                      0,
                      0]))

    nodes_mean, nodes_std = func(0)
    edges_mean, edges_std = func(1)
    degree_mean, degree_std = func(2)
    degree_0_mean, degree_0_std = func(3, True)
    degree_1_mean, degree_1_std = func(4, True)
    max_component_mean, max_component_std = func(-1)
    return pd.DataFrame(list(zip(nodes_mean, nodes_std, edges_mean, edges_std, degree_mean, degree_std, degree_0_mean,
                                 degree_0_std, degree_1_mean, degree_1_std, max_component_mean, max_component_std)),
                        columns=['nodes mean', 'nodes std', 'edges mean', 'edges std', 'degree mean', 'degree std',
                                 'degree travellers mean', 'degree travellers std', 'degree rides mean',
                                 'degree rides std', 'max component mean', 'max component std'],
                        index=['bipartite shareability', 'bipartite matching', 'pairs shareability', 'pairs matching'])


def analysis_all_graphs(graph_list, config=None, save=True, logger_level='INFO', save_num='', date=None):
    logger = init_log(logger_level)
    logger.info("Analysing list of graphs")
    t = concat_all_graph_list(graph_list)
    t = analyse_concatenated_all_graph_list(t)
    if save:
        if save_num is None:
            save_num = ''
        else:
            save_num = str(save_num)
        if date is None:
            date = str(datetime.date.today().strftime("%d-%m-%y"))
        else:
            assert isinstance(date, str), "Wrong type of 'date' variable passed, should be string"
        t.to_excel(config.path_results + 'all_graphs_properties_' + str(save_num) + '_' + date + '.xlsx')
    logger.info("Analysing list of graphs finalised")
    return t


def save_with_pickle(obj, name, config, date=None):
    if date is None:
        date = str(datetime.date.today().strftime("%d-%m-%y"))
    else:
        assert isinstance(date, str), "Wrong type of 'date' variable passed, should be string"
        date = date
    with open(config.path_results + name + '_' + date + '.obj', 'wb') as file:
        pickle.dump(obj, file)


def read_pickle(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            e = pickle.load(file)
        return e

    except:
        raise Exception("Cannot read the file. Make sure the path is correct.")


def centrality_degree(graph, tuned=True, alpha=1):
    """
    Refined function given in a networkx package allowing to calculate centrality degree on weighted networks
    @param graph: networkx graph
    @param tuned: choose whether to use refined version proposed in
    "Node centrality in weighted networks: Generalizing degree and shortest paths" by Tore Opsahla, Filip Agneessensb,
    John Skvoretzc or more straightforward approach (tuned=True => article version)
    @param alpha: parameter used in refined version of the tuned model
    @return: dictionary with centrality degree per node
    """
    if len(graph) <= 1:
        return {n: 1 for n in graph}
    if not nx.is_weighted(graph):
        return nx.degree_centrality(graph)

    elif nx.is_weighted(graph) and not tuned:
        max_degree_corr = 1 / (max([i[1] for i in graph.degree(weight='weight')]) * (len(graph) - 1))
        return {n: max_degree_corr * d for n, d in graph.degree(weight='weight')}

    elif nx.is_weighted(graph) and tuned:
        degrees_strength = zip(graph.degree(), graph.degree(weight='weight'))
        degrees_strength = [(x[0][0], x[0][1], x[1][1]) for x in degrees_strength]

        def foo(s, k, alpha):
            if k == 0:
                return 0
            else:
                return k * np.power(s / k, alpha)

        return {n: foo(s, k, alpha) for (n, k, s) in degrees_strength}

    else:
        raise Exception("Invalid arguments")


class StructuralProperties:
    """
    Aggregated functions designed to calculate structural properties of the networks
    """
    def __init__(self, graph, tuned_degree_centrality=True, alpha_degree_centrality=1):
        self.G = graph
        self.tuned_degree_centrality = tuned_degree_centrality
        self.alpha_degree_centrality = alpha_degree_centrality
        self.centrality_degree = None
        self.eigenvector_centrality = None

    def centrality_measures(self, tuned_degree_centrality=True, alpha_degree_centrality=1):
        self.centrality_degree = centrality_degree(self.G, self.tuned_degree_centrality, self.alpha_degree_centrality)
        if nx.is_weighted(self.G):
            self.eigenvector_centrality = nx.eigenvector_centrality_numpy(self.G, weight='weight')
        else:
            self.eigenvector_centrality = nx.eigenvector_centrality_numpy(self.G)
