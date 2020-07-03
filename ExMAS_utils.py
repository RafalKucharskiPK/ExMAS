#!/usr/bin/env python3

"""
Functions used within ExMAS
"""

import os
import sys
# utils
import json
import random
import math
import logging
from dotmap import DotMap
# side packages
import pandas as pd
import osmnx as ox
import networkx as nx
import numpy as np
from osmnx.distance import get_nearest_node

# DataFrame skeletons
inData = DotMap()
inData['passengers'] = pd.DataFrame(columns=['id', 'pos', 'status'])
inData.passengers = inData.passengers.set_index('id')
inData['requests'] = pd.DataFrame(
    columns=['pax', 'origin', 'destination', 'treq', 'tdep', 'ttrav', 'tarr', 'tdrop']).set_index(
    'pax')  # to do - move results into simenv

# definitions for KPIs
KPIs_descriptions = ['total travel time of vehicles (with travellers only)',
                     'as above yet in non-shared scenarion ',
                     'total travel time of passengers',
                     'as above yet in non-shared scenarion ',
                     'total (dis)utility of passengers',
                     'as above yet in non-shared scenarion ',
                     'mean vehicle cost reduction (lambda) over shared rides',
                     'total fares paid by travellers sharing',
                     'as above yet in non-shared scenarion ',
                     'relative revenue reduction',
                     'number of trips',
                     'number of single rides in the solution',
                     '2nd degree rides in the solution',
                     '3rd degree rides in the solution',
                     '4th degree rides in the solution',
                     '5th degree rides in the solution',
                     'rides of degree greater than 5 in the solution',
                     'what portion of rides were shared',
                     'proxy for the fleet size (lower bound)',
                     'as above yet in non-shared scenarion ',
                     'maximal discount to be offered while profitable', 'sys', 'sys']


def make_paths(params):
    # call it whenever you change a city name, or main path
    params.paths.main = ""
    params.paths.data = os.path.join(params.paths.main, 'data')
    params.paths.params = os.path.join(params.paths.data, 'configs')
    params.paths.albatross = os.path.join(params.paths.data, 'albatross')  # albatross data
    params.paths.G = os.path.join(params.paths.data, 'graphs',
                                  params.city.split(",")[0] + ".graphml")  # graphml of a current .city
    params.paths.skim = os.path.join(params.paths.main, 'data', 'graphs', params.city.split(",")[
        0] + ".csv")  # csv with a skim between the nodes of the .city

    params.paths.postcodes = os.path.join(params.paths.data, 'postcodes',
                                          "PC4_Nederland_2015.shp")  # PCA4 codes shapefile
    return params


def get_config(path):
    # reads a .json file with MaaSSim configuration
    # use as: params = get_config(config.json)
    with open(path) as json_file:
        data = json.load(json_file)
        config = DotMap(data)
    config['t0'] = pd.Timestamp('15:00')
    return config


def save_config(_params, path=None):
    if path is None:
        path = os.path.join(_params.paths.params, _params.NAME + ".json")
    with open(path, "w") as write_file:
        json.dump(_params, write_file)


def empty_series(df, s_id=None):
    # returns empty Series from a given DataFrame, to be used for consistency of adding new rows to DataFrames
    if s_id is None:
        s_id = len(df.index) + 1
    return pd.Series(index=df.columns, name=s_id)


def rand_node(df):
    # returns a random node of a graph
    return df.loc[random.choice(df.index)].name


def generic_generator(generator, n):
    # to create multiple passengers/vehicles/etc
    return pd.concat([generator(i) for i in range(1, n + 1)], axis=1, keys=range(1, n + 1)).T


def generate_passenger(p_id=None, rand=False):
    """
    generates single passenger (database row with structure defined in DataStructures)
    index is consecutive number if dataframe
    position is random graph node
    status is IDLE
    """
    passenger = empty_series(inData.passengers, p_id)
    passenger.pos = int(rand_node(inData.nodes)) if rand else None
    passenger.status = 0
    return passenger


def download_G(inData, _params, make_skims=True):
    # uses osmnx to download the graph
    inData.G = ox.graph_from_place(_params.city, network_type='drive')
    inData.nodes = pd.DataFrame.from_dict(dict(inData.G.nodes(data=True)), orient='index')
    if make_skims:
        inData.skim_generator = nx.all_pairs_dijkstra_path_length(inData.G,
                                                                  weight='length')
        inData.skim_dict = dict(inData.skim_generator)  # filled dict is more usable
        inData.skim = pd.DataFrame(inData.skim_dict).fillna(_params.dist_threshold).T.astype(
            int)  # and dataframe is more intuitive
    return inData


def save_G(inData, _params, path=None):
    # saves graph and skims to files
    ox.save_graphml(inData.G, filepath=_params.paths.G)
    inData.skim.to_csv(_params.paths.skim, chunksize=20000000)


def load_G(inData, _params=None, stats=False, set_t=True):
    # loads graph and skim from a file

    inData.G = ox.load_graphml(filepath=_params.paths.G)
    inData.nodes = pd.DataFrame.from_dict(dict(inData.G.nodes(data=True)), orient='index')
    skim = pd.read_csv(_params.paths.skim, index_col='Unnamed: 0')
    skim.columns = [int(c) for c in skim.columns]
    inData.skim = skim
    if stats:
        inData.stats = networkstats(inData)  # calculate center of network, radius and central node
    return inData


def merge_csvs(params, path, to_numeric=True):
    """ merges csvs in one folder into a single DF"""
    import glob

    if path is None:
        path = params.paths.sblt
    # merge csvs in a single folder with unit results
    # returns a DataFrame
    all_files = glob.glob(path + "/*.csv")
    l = list()
    for file_ in all_files:
        df = pd.read_csv(file_, index_col=0)
        l.append(df.T)

    res = pd.concat(l)
    if to_numeric:
        res = res.apply(pd.to_numeric, errors='coerce')

    return res


def networkstats(inData):
    """
    for a given network calculates it center of gravity (avg of node coordinates),
    gets nearest node and network radius (75th percentile of lengths from the center)
    returns a dictionary with center and radius
    """
    center_x = pd.DataFrame((inData.G.nodes(data='x')))[1].mean()
    center_y = pd.DataFrame((inData.G.nodes(data='y')))[1].mean()

    nearest = get_nearest_node(inData.G, (center_y, center_x))
    ret = DotMap({'center': nearest, 'radius': inData.skim[nearest].quantile(0.75)})
    return ret


def load_albatross_csv(_inData, _params, sample=True):
    # loads the full csv of albatross for a given city
    # changes date for today
    # filters for simulation time (t0 hour + simTime)
    # samples the n
    df = pd.read_csv(os.path.join(_params.paths.albatross,
                                  _params.city.split(",")[0] + "_requests.csv"),
                     index_col='Unnamed: 0')
    df['treq'] = pd.to_datetime(df['treq'])
    df.treq = df.treq + (_params.t0.date() - df.treq.iloc[0].date())
    df['tarr'] = pd.to_datetime(df['tarr'])
    df.tarr = df.tarr + (_params.t0.date() - df.tarr.iloc[0].date())
    # sample within simulation time
    df = df[df.treq.dt.hour >= _params.t0.hour]
    df = df[df.treq.dt.hour <= (_params.t0.hour + _params.simTime)]

    df['dist'] = df.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)
    df = df[df.dist < _params.dist_threshold]

    if sample:
        df = df.sample(_params.nP)

    df['ttrav_alb'] = pd.to_timedelta(df.ttrav)

    df['ttrav'] = df.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
    _inData.requests = df
    _inData.passengers = generic_generator(generate_passenger, _params.nP).reindex(_inData.requests.index)
    _inData.passengers.pos = _inData.requests.origin

    return _inData


def generate_demand(_inData, _params=None, avg_speed=False):
    # generates nP requests with a given temporal and spatial distribution of origins and destinations
    # returns _inData.requests dataframe populated with nodes and times.
    df = pd.DataFrame(index=np.arange(1, _params.nP + 1), columns=inData.passengers.columns)
    df.status = 0
    df.pos = df.apply(lambda x: rand_node(_inData.nodes), axis=1)
    _inData.passengers = df
    requests = pd.DataFrame(index=df.index, columns=_inData.requests.columns)

    distances = _inData.skim[_inData.stats['center']].to_frame().dropna()  # compute distances from center
    distances.columns = ['distance']
    distances = distances[distances['distance'] < _params.dist_threshold]
    distances['p_origin'] = distances['distance'].apply(lambda x:
                                                        math.exp(
                                                            _params.demand_structure.origins_dispertion * x))  # apply negative exponential distributions
    distances['p_destination'] = distances['distance'].apply(
        lambda x: math.exp(_params.demand_structure.destinations_dispertion * x))
    if _params.demand_structure.temporal_distribution == 'uniform':
        treq = np.random.uniform(-_params.simTime * 60 * 60 / 2, _params.simTime * 60 * 60 / 2,
                                 _params.nP)  # apply uniform distribution on request times
    elif _params.demand_structure.temporal_distribution == 'normal':
        treq = np.random.normal(_params.simTime * 60 * 60 / 2,
                                _params.demand_structure.temporal_dispertion * _params.simTime * 60 * 60 / 2,
                                _params.nP)  # apply normal distribution on request times
    requests.treq = [_params.t0 + pd.Timedelta(int(_), 's') for _ in treq]
    requests.origin = list(
        distances.sample(_params.nP, weights='p_origin', replace=True).index)  # sample origin nodes from a distribution
    requests.destination = list(distances.sample(_params.nP, weights='p_destination',
                                                 replace=True).index)  # sample destination nodes from a distribution

    requests['dist'] = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)
    while len(requests[requests.dist >= _params.dist_threshold]) > 0:
        requests.origin = requests.apply(lambda request: (distances.sample(1, weights='p_origin').index[0]
                                                          if request.dist >= _params.dist_threshold else request.origin),
                                         axis=1)
        requests.destination = requests.apply(lambda request: (distances.sample(1, weights='p_destination').index[0]
                                                               if request.dist >= _params.dist_threshold else request.destination),
                                              axis=1)
        requests.dist = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)
    requests['ttrav'] = requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
    # requests.ttrav = pd.to_timedelta(requests.ttrav)
    if avg_speed:
        requests.ttrav = (pd.to_timedelta(requests.ttrav) / _params.speeds.ride).dt.floor('1s')
    requests.tarr = [request.treq + request.ttrav for _, request in requests.iterrows()]
    requests = requests.sort_values('treq')
    requests['pax_id'] = requests.index.copy()
    _inData.requests = requests
    _inData.passengers.pos = _inData.requests.origin
    return _inData


def mode_choices(_inData, sp):
    """
    Compete with Transit.
    Parameters needed:
    params.shareability.PT_discount = 0.66
    params.shareability.PT_beta = 1.3
    params.shareability.PT_speed = 1/2
    params.shareability.beta_prob = -0.5
    params.shareability.probabilistic = False
    inData.requests['timePT']
    :param _inData:
    :param sp:
    :return:
    """
    rides = _inData.sblts.rides
    r = _inData.sblts.requests

    def get_probs(row, col='u_paxes'):
        prob_SR = list()
        for pax in range(len(row.indexes)):
            denom = math.exp(sp.beta_prob * row.u_paxes[pax]) + math.exp(sp.beta_prob * row.uPTs[pax]) + math.exp(
                sp.beta_prob * row.uns[pax])
            prob_SR.append(math.exp(sp.beta_prob * row[col][pax]) / denom)
        return prob_SR

    rides['uPTs'] = rides.apply(lambda x: [r.loc[_].u_PT for _ in x.indexes], axis=1)  # utilities of PT alternatives
    rides['uns'] = rides.apply(lambda x: [r.loc[_].u for _ in x.indexes], axis=1)  # utilities of non shared alternative

    rides['prob_PT'] = rides.apply(lambda x: get_probs(x, col='uPTs'), axis=1)  # MNL probabilities to go for PT
    rides['prob'] = rides.apply(lambda x: get_probs(x), axis=1)  # MNL probabilities to go for shared
    rides['max_prob_PT'] = rides.apply(lambda x: 0 if len(x.indexes) == 1 else max(x.prob_PT), axis=1)
    rides = rides[rides.max_prob_PT < 0.4]  # cut off function - dummy
    _inData.sblts.rides = rides
    return _inData



def init_log(sp, logger=None):
    level = sp.get('logger_level', "INFO")
    if level == 'INFO':
        level == logging.INFO
    elif level == 'WARNING':
        level == logging.WARNING
    elif level == 'CRITICAL':
        level = logging.CRITICAL
    if logger is None:
        logging.basicConfig(stream=sys.stdout, format='%(asctime)s-%(levelname)s-%(message)s',
                            datefmt='%d-%m-%y %H:%M:%S', level=level)

        logger = logging.getLogger()

        logger.setLevel(level)
        return logging.getLogger(__name__)
    else:
        logger.setLevel(level)
        return logger
