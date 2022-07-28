import pandas as pd
from dotmap import DotMap
import ExMAS.main
from ExMAS.utils import inData as inData
import json
import os
import glob
import utils


def get_initial_parameters(path, root_path=None):
    with open(path) as json_file:
        data = json.load(json_file)
        config = DotMap(data)
    return config


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
    dataset['Fraction_isolated'] = dataset['No_isolated_pairs']/dataset['nP']
    return dataset


def split_demand_structure(run_parameters):
    out_temp = []
    out = []
    for j in range(len(run_parameters.demand_variables)):
        out_temp += [[run_parameters.demand_variables[j], b] for b in run_parameters.values_of_demand_variables[j]]
    for j in range(len(out_temp)):
        out += [out_temp[j] + [rep_no] for rep_no in range(run_parameters.replications)]
    return out


def split_operator_structure(run_parameters):
    out = []
    for j in range(len(run_parameters.operator_variables)):
        out += [[run_parameters.operator_variables[j], b] for b in run_parameters.values_of_operator_variables[j]]
    return out


def clear_temp_folder(path):
    all_temp_files = glob.glob(path + '*.csv')
    all_temp_files += glob.glob(path + '*.png')
    for file in all_temp_files:
        os.remove(file)


def create_inData():
    inData = DotMap()
    inData['passengers'] = pd.DataFrame(columns=['id', 'pos', 'status'])
    inData.passengers = inData.passengers.set_index('id')
    inData['requests'] = pd.DataFrame(
        columns=['pax', 'origin', 'destination', 'treq', 'tdep', 'ttrav', 'tarr', 'tdrop']).set_index(
        'pax')  # to do - move results into simenv
    return inData


def demand_generation(parameters: DotMap, search_space_argument: [str, float, int], precise_t0_start):
    variable, value, replication_no = search_space_argument
    parameters[variable] = value
    data = create_inData()
    data.demand_generation_info = search_space_argument
    parameters = ExMAS.utils.make_paths(parameters)
    parameters['t0'] = pd.Timestamp(parameters['t0'])
    ExMAS.utils.load_G(data, parameters, stats=True)
    utils.amended_generate_demand(data, parameters)
    return data


def worker_topological_properties(GraphStatObj):
    data_output = pd.DataFrame()
    GraphStatObj.all_analysis()
    if GraphStatObj.bipartite:
        data_output = data_output.append([GraphStatObj.num_nodes_group0, GraphStatObj.num_nodes_group1,
                                          GraphStatObj.average_degree, GraphStatObj.maximum_degree,
                                          GraphStatObj.average_degree_group0,
                                          GraphStatObj.average_degree_group1, GraphStatObj.average_clustering_coefficient,
                                          GraphStatObj.proportion_max_component, len(GraphStatObj.components),
                                          GraphStatObj.average_clustering_group0, GraphStatObj.average_clustering_group1,
                                          GraphStatObj.number_of_isolated_pairs,
                                          GraphStatObj.average_clustering_group0_reduced,
                                          GraphStatObj.average_clustering_group1_reduced])
        data_output.index = ['No_nodes_group0', 'No_nodes_group1', 'Average_degree',
                             'Maximum_degree', 'Average_degree_group0', 'Average_degree_group1', 'Avg_clustering',
                             'Proportion_max_component', 'No_components', 'Average_clustering_group0',
                             'Average_clustering_group1', 'No_isolated_pairs', 'Average_clustering_group0_reduced',
                             'Average_clustering_group1_reduced']
    else:
        data_output = data_output.append([GraphStatObj.average_degree, GraphStatObj.maximum_degree,
                                          GraphStatObj.average_clustering_coefficient,
                                          GraphStatObj.proportion_max_component,
                                          len(GraphStatObj.components)])
        data_output.index = ['Average_degree', 'Maximum_degree', 'Avg_clustering',
                             'Proportion_max_component', 'No. of components']

    return data_output


def amend_ids(ids: list):
    names = ids[0][:-1][0::2] + ['Replication_ID']
    out = []
    for j in range(len(ids)):
        values = ids[j][1::2] + [ids[j][-1]]
        x = [dict(zip(names, values))]
        out.append(pd.DataFrame(x))
    return out


def amend_merged_file(merged_file, alter_kpis=False, inplace=True):
    if not inplace:
        merged_file = merged_file.copy(deep=True)
    merged_file.drop(columns=['_typ', 'dtype'], inplace=True)
    merged_file.reset_index(inplace=True, drop=True)
    if alter_kpis:
        merged_file = alternate_kpis(merged_file)
    return merged_file


def alter_kpis(dataset):
    dataset['SavedVehHours'] = (dataset['VehHourTrav_ns'] - dataset['VehHourTrav']) / \
                                    dataset['VehHourTrav_ns']
    dataset['AddedPasHours'] = (dataset['PassHourTrav'] - dataset['PassHourTrav_ns']) / \
                                    dataset['PassHourTrav_ns']
    dataset['UtilityGained'] = (dataset['PassUtility'] - dataset['PassUtility_ns']) / \
                                    dataset['PassUtility_ns']
    return dataset
