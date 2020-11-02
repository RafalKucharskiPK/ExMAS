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
import matplotlib.pyplot as plt

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
                     'maximal discount to be offered while profitable',
                     'sys',
                     'sys']

# definitions for KPIs
rides_DESCRIPTIONS = ['travellers indexes',
                      'total (dis)utility of travellers (eq. 4 from the paper)',
                      'total (dis)utility of vehicle (ride time) [s]',
                      'type of a trip (1 - single rides, 2* rides of 2nd degree, 3* rd degree, etc. second digit denotes FIFO/LIFO',
                      '(dis) utilities of consecutive travellers',
                      'sequence of times [first pickup (seconds of simulation) and consecutive time deltasalong the schedule',
                      'indexes of consequently picked up travellers',
                      'indexes of consequenctly droped off travellers',
                      'degree (number of travellers)',
                      'id',
                      'savings in vehicle hours due to sharing',
                      'total travel times without sharing',
                      'row from incidence matrix used for ILP matching',
                      'binary flag whether rides is selected as part of the solution']


# definitions for KPIs
trips_DESCRIPTIONS = ['id',
                      'origin node (OSM index)',
                      'destination node (OSM index)',
                      'desired departure time',
                      '[deprecated, from MaaSSim]',
                      'travel time [s] (shortest)',
                      'desired arrival time (pd.datetime)',
                      '[deprecated, from MaaSSim]',
                      'distance in meters (shortest)',
                      'id',
                      'value of time',
                      'maximal acceptable delay (eq. 6 from the paper)',
                      '(dis)utility of a non-shared ride',
                      'utility of PT alternative (if computed)',
                      'selected ride id (from *.schedule DataFrame)',
                      'travel time in the solution (possibly shared)',
                      '(dis) utility of selected ride',
                      'type of selected ride',
                      'index of being picked up in the  ride']




def make_paths(params, relative = False):
    # call it whenever you change a city name, or main path
    params.paths.main = "ExMAS"

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


def get_config(path, root_path = None):
    # reads a .json file with MaaSSim configuration
    # use as: params = get_config(config.json)
    with open(path) as json_file:
        data = json.load(json_file)
        config = DotMap(data)
    config['t0'] = pd.Timestamp('15:00')

    if root_path is not None:
        config.paths.G = os.path.join(root_path, config.paths.G)  # graphml of a current .city
        config.paths.skim = os.path.join(root_path,config.paths.skim)  # csv with a skim between the nodes of the .city

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

###########
# RESULTS #
###########

def merge_csvs(params = None, path = None, to_numeric=True, read_columns_from_filename = False):
    """ merges csvs in one folder into a single DF"""
    import glob

    if path is None:
        path = params.paths.sblt
    # merge csvs in a single folder with unit results
    # returns a DataFrame
    all_files = glob.glob(path)
    l = list()
    for file_ in all_files:
        df = pd.read_csv(file_, index_col=0).T
        if read_columns_from_filename:
            fields = file_.replace(path[:-5],'')[:-4].split("_")
            df[fields[0]] = fields[1]  # n centers
            df['gammdist_shape'] = fields[4]
            df['gamma_imp_shape'] = fields[8]
        l.append(df)

    res = pd.concat(l)
    if to_numeric:
        res = res.apply(pd.to_numeric, errors='coerce')

    return res

def make_KPIs(df, params):
    df['$U_q$'] = -df.PassUtility_ns
    df['$U_r$'] = -df.PassUtility
    df['$T_q$'] = -df.VehHourTrav
    df['$T_r$'] = -df.PassHourTrav
    df['$\Delta T_r$'] = (df.VehHourTrav - df.VehHourTrav_ns)/df.VehHourTrav_ns
    df['$\Delta T_q$'] = (df.PassHourTrav - df.PassHourTrav_ns)/df.PassHourTrav_ns
    df['$\Delta U_r$'] = -(df.PassUtility - df.PassUtility_ns)/df.PassUtility_ns
    df['$\Delta F$'] = -(df.fleet_size_nonshared- df.fleet_size_shared)/df.fleet_size_nonshared
    df['$\Lambda_r$'] = df.lambda_shared
    df['$\lambda$'] = df.shared_discount
    df['$R$'] =(df.SINGLE + df.PAIRS + df.TRIPLES + df.QUADRIPLES)
    df['$Q$'] = df.nP
    df['$T$'] = df.horizon.apply(lambda x: 3600 if x == -1 else x)
    df['occupancy'] = df.PassHourTrav/df.VehHourTrav
    df['revenue_ns'] = df.VehHourTrav_ns*params.price
    df['revenue_s'] = ((1-df.shared_ratio)*df.VehHourTrav_ns*params.price +
                       df.shared_ratio*df.VehHourTrav_ns*params.price*(1-params.shared_discount))
    df['$\Delta I$'] = (df['revenue_s'] - df['revenue_ns']) / df['revenue_ns']
    return df


def plot_paper(tp, figname='res.svg', x='nP', y='shared_ratio', y_label=None, y_lim=None, legend=True,
               groupby=False, kind='line', stacked=False, ax=None, diag=False, path='../res/figs/'):
    # plots x as a function of y grouped by groupby
    save = False
    tp = tp.sort_values(x)
    # tp = tp.astype(float,errors = 'ignore')
    tp = tp.set_index(x)
    if groupby:
        tp = tp.groupby(groupby)
    if ax is None:
        fig, ax = plt.subplots()
        save = True
    tp[y].plot(legend=legend, ax=ax, kind=kind, stacked=stacked)

    ax.set_ylabel(y if y_label is None else y_label)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_xlabel(x)
    if diag:
        lims = [ax.get_xlim(), ax.get_ylim()]

        lims = min(lims[0][0], lims[1][0]), max(lims[0][1], lims[1][1])
        ax.plot((lims[0], lims[1]), (lims[0], lims[1]), ls="--", c=".5", lw=1)
    if groupby:
        plt.legend(title=groupby)
    if save:
        plt.savefig(path + figname)
    return ax


def plot_paper_multi(tp, figname='res.svg', x='nP', ys=['$U_q$', '$U_r$'], path='', y_label='rel. diff.'):
    fig, ax = plt.subplots()
    for y in ys:
        ax = plot_paper(tp, x=x, y=y, ax=ax)
        ax.set_ylabel(y_label)
    plt.savefig(path + figname)
    plt.show()
    return fig


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


def requests_to_geopandas(inData, filename = None):
    """
    creates a geopandas dataframe with cooridinates Point(x,y) of origin and destination of request
    :param inData: uses inData.requests and inData.nodes (with coordinates)
    :param filename: path to save csv
    :return: geopandas dataframe with geo requests
    """
    # those imports are local only to this functions
    from shapely.geometry import Point
    import geopandas as gpd

    geo_requests = inData.requests
    geo_requests['Point_o'] = geo_requests.apply(lambda r: Point(inData.nodes.loc[r.origin].x, inData.nodes.loc[r.origin].y),
                                         axis=1)
    geo_requests['origin_x'] = geo_requests.apply(lambda r: inData.nodes.loc[r.origin].x, axis=1)
    geo_requests['origin_y'] = geo_requests.apply(lambda r: inData.nodes.loc[r.origin].y, axis=1)
    geo_requests['Point_d'] = geo_requests.apply(
        lambda r: Point(inData.nodes.loc[r.destination].x, inData.nodes.loc[r.destination].y), axis=1)
    geo_requests['destination_x'] = geo_requests.apply(lambda r: inData.nodes.loc[r.destination].x, axis=1)
    geo_requests['destination_y'] = geo_requests.apply(lambda r: inData.nodes.loc[r.destination].y, axis=1)
    geo_requests = gpd.GeoDataFrame(geo_requests, geometry='Point_o')
    if filename is not None:
        geo_requests.to_csv(filename)
    return geo_requests


def synthetic_demand_poly(_inData, _params=None):
    from random import seed
    from scipy.stats import gamma as gamma_random
    print(os.getcwd())

    origins_albatross = pd.read_csv('ExMAS/spinoffs/potential/AM_origins.csv', index_col=0)
    origins_albatross = origins_albatross[
        origins_albatross['origin'].isin(inData.skim.index)]  # droping those origins not in skim

    # Sample_origins = origins_albatross.sample(params.nP)
    # Sample_origins.to_csv('Sample_origins.csv')
    Sample_origins = pd.read_csv('ExMAS/spinoffs/potential/Sample_origins.csv', index_col=0).sample(_params.nP)
    # 1. create a passenger DataFrame

    df = pd.DataFrame(index=np.arange(1, _params.nP + 1), columns=_inData.passengers.columns)
    df.status = 0

    # initialize with a completely random nodes (no distribution)
    df.pos = df.apply(lambda x: rand_node(_inData.nodes), axis=1)
    _inData.passengers = df

    print('1', pd.Timestamp.now())

    # 1. create a Centers and potential destinations
    Centers = pd.DataFrame(data={'name': ['Dam Square', 'Station Zuid', 'Concertgebouw', 'Sloterdijk'],
                                 'x': [4.8909126, 4.871887, 4.8790061, 4.8351158],
                                 'y': [52.373095, 52.338948, 52.356275, 52.3888349]})
    Centers['node'] = Centers.apply(lambda center: get_nearest_node(_inData.G, (center.y, center.x)), axis=1)

    # drop not considered centers
    Centers = Centers[Centers.index.isin(range(_params.nCenters))]

    Dist = []

    rdist = gamma_random(a=_params.gammdist_shape, scale=_params.gammdist_scale)

    for i in range(_params.nCenters):
        # 2. distances and probabiltiies for each node #get distances from centre for each node
        _distances = _inData.skim.transpose()[Centers['node'][i]].to_frame().dropna()  # compute distances from center
        _distances.columns = ['distance']
        _distances = _distances[
            _distances['distance'] < _params.dist_threshold]  # drop disconnected points from choice set

        # apply distributions to obtain probability of destination
        _distances['p_destination'] = _distances['distance'].apply(lambda x: rdist.pdf(x))

        _distances['dist_range'] = _distances['distance'].apply(lambda x: math.trunc(x / 500) + 1)
        _distances_sum = _distances.groupby(by=['dist_range']).count()
        _distances_sum = _distances_sum.reset_index()
        _distances_sum = _distances_sum.drop(columns='p_destination')
        _distances_sum.columns = ['dist_range', 'cant']
        total = _distances_sum['cant'].mean()
        index_id = _distances.index
        _distances = _distances.merge(_distances_sum, on='dist_range', how='left')
        _distances.index = index_id
        _distances['fact'] = total / _distances['cant']
        _distances['p_destination_fixed'] = _distances['p_destination'] * _distances['fact']

        Dist.append(_distances)

    print('2', pd.Timestamp.now())

    randomdestinations = []

    # we can generate more potential destinations if desired
    for i in range(_params.nCenters):
        for j in range(_params.nP * 3):
            randomdestinations.append(Dist[i].sample(1, weights='p_destination_fixed', replace=True).index.values[0])

    print('3', pd.Timestamp.now())

    _inData.RandomDestinations = randomdestinations

    # 3. create requests DataFrame
    requests = pd.DataFrame(index=df.index, columns=_inData.requests.columns)

    # 3.a. draw random departure time

    # apply uniform distribution on request times
    treq = np.random.uniform(-_params.simTime * 60 * 60 / 2, _params.simTime * 60 * 60 / 2, _params.nP)

    requests.treq = [_params.t0 + pd.Timedelta(int(_), 's') for _ in treq]

    # 3.b. draw origin sampling albatross origins
    requests.origin = Sample_origins.values

    # 3.c. draw destination with a probability
    # seed fixed or not?
    seed(1)

    destination = []
    dist = []

    print('4', pd.Timestamp.now())

    rdist = gamma_random(a=_params.gamma_imp_shape, scale=_params.gamma_imp_scale)

    for i in range(_params.nP):

        _distances = pd.DataFrame(randomdestinations)
        _distances.columns = ["id"]
        _distances['distance'] = _distances.apply(lambda request: _inData.skim.loc[requests.origin[i + 1], request.id],
                                                  axis=1)
        _distances = _distances[_distances['distance'] < _params.dist_threshold]

        while (sum(_distances['distance']) == 0):
            requests.origin[i + 1] = origins_albatross['origin'].sample(1, replace=True).values[0]
            _distances = pd.DataFrame(randomdestinations)
            _distances.columns = ["id"]
            _distances['distance'] = _distances.apply(
                lambda request: _inData.skim.loc[requests.origin[i + 1], request.id], axis=1)
            _distances = _distances[_distances['distance'] < _params.dist_threshold]
            # print('loop gravity',requests.origin[i+1],sum(_distances['distance']) ,pd.Timestamp.now())

        # apply distributions to obtain probability of destination

        _distances['p_destination'] = _distances['distance'].apply(lambda x: rdist.pdf(x))

        _distances['dist_range'] = _distances['distance'].apply(lambda x: math.trunc(x / 500) + 1)
        _distances_sum = _distances.groupby(by=['dist_range']).count()
        _distances_sum = _distances_sum.reset_index()
        _distances_sum = _distances_sum.drop(columns=['id', 'p_destination'])
        _distances_sum.columns = ['dist_range', 'cant']
        _distances_sum

        total = _distances_sum['cant'].mean()
        index_id = _distances.index
        _distances = _distances.merge(_distances_sum, on='dist_range', how='left')
        _distances.index = index_id
        _distances['fact'] = total / _distances['cant']
        _distances['p_destination_fixed'] = _distances['p_destination'] * _distances['fact']

        samp_destination = _distances.sample(1, weights='p_destination_fixed', replace=True)
        destination.append(samp_destination['id'].values[0])
        dist.append(samp_destination['distance'].values[0])
        # print('iter ',i,pd.Timestamp.now())

    print('5', pd.Timestamp.now())

    requests.destination = destination
    requests.dist = dist
    requests['dist'] = dist
    # requests['dist'] = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)

    # 4 make travel times (without using speed, speed param is used inside cal_sblt)
    requests['ttrav'] = requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)

    requests.tarr = [request.treq + request.ttrav for _, request in requests.iterrows()]

    requests = requests.sort_values('treq')
    requests.index = df.index

    # output
    _inData.requests = requests
    _inData.passengers.pos = requests.origin
    _inData.probs = requests.dist
    _inData.centers = Centers

    print('6', pd.Timestamp.now())

    return _inData


def plot_demand_poly(inData, _params, t0=None, vehicles=False, s=10):
    plt.rcParams['figure.figsize'] = [12, 4]
    if t0 is None:
        t0 = inData.requests.treq.mean()

    # plot osmnx graph, its center, scattered nodes of requests origins and destinations
    # plots requests temporal distribution
    fig, ax = plt.subplots(1, 3)
    ((t0 - inData.requests.treq) / np.timedelta64(1, 'h')).plot.kde(title='Temporal distribution', ax=ax[0])
    (inData.requests.ttrav / np.timedelta64(1, 'm')).plot(kind='hist', title='Trips travel times [min]', ax=ax[1])
    inData.requests.dist.plot(kind='hist', title='Trips distance [m]', ax=ax[2])
    # (inData.requests.ttrav / np.timedelta64(1, 'm')).describe().to_frame().T
    plt.tight_layout()
    plt.show()
    fig, ax = ox.plot_graph(inData.G, node_size=0, edge_linewidth=0.5,
                            show=False, close=False,
                            edge_color='black', bgcolor = 'white')
    for _, r in inData.requests.iterrows():
        ax.scatter(inData.G.nodes[r.origin]['x'], inData.G.nodes[r.origin]['y'], c='green', s=s, marker='D')
        ax.scatter(inData.G.nodes[r.destination]['x'], inData.G.nodes[r.destination]['y'], c='orange', s=s)
    if vehicles:
        for _, r in inData.vehicles.iterrows():
            ax.scatter(inData.G.nodes[r.pos]['x'], inData.G.nodes[r.pos]['y'], c='blue', s=s, marker='x')
    ax.scatter(inData.G.nodes[inData.stats['center']]['x'], inData.G.nodes[inData.stats['center']]['y'], c='red',
               s=50 * s, marker='x')

    for i in range(_params.nCenters):
        ax.scatter(inData.G.nodes[inData.centers[i]]['x'], inData.G.nodes[inData.centers[i]]['y'], c='blue', s=50 * s,
                   marker='x')

    plt.tight_layout()
    plt.title(
        'Demand in {} with origins marked in green, destinations in orange and vehicles in blue'.format(_params.city))
    plt.show()


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


def load_requests(path):
    requests = pd.read_csv(path, index_col=0)
    requests.treq = pd.to_datetime(requests.treq)
    requests['pax_id'] = requests.index.copy()
    requests.tarr = pd.to_datetime(requests.tarr)
    requests.ttrav = pd.to_timedelta(requests.ttrav)
    #requests.origin = requests.origin.astype(int)
    #requests.destination = requests.destination.astype(int)
    return requests


def plot_demand(inData, params, t0=None, vehicles=False, s=10):
    if t0 is None:
        t0 = inData.requests.treq.mean()

    # plot osmnx graph, its center, scattered nodes of requests origins and destinations
    # plots requests temporal distribution
    fig, ax = plt.subplots(1, 3, figsize = (12,4))
    ((t0 - inData.requests.treq) / np.timedelta64(1, 'h')).plot.kde(title='Temporal distribution', ax=ax[0])
    (inData.requests.ttrav / np.timedelta64(1, 'm')).plot(kind='box', title='Trips travel times [min]', ax=ax[1])
    inData.requests.dist.plot(kind='box', title='Trips distance [m]', ax=ax[2])
    # (inData.requests.ttrav / np.timedelta64(1, 'm')).describe().to_frame().T
    plt.show()
    fig, ax = ox.plot_graph(inData.G, figsize=(10, 10), node_size=0, edge_linewidth=0.5,
                            show=False, close=False,
                            edge_color='grey', bgcolor = 'white' )
    for _, r in inData.requests.iterrows():
        ax.scatter(inData.G.nodes[r.origin]['x'], inData.G.nodes[r.origin]['y'], c='green', s=s, marker='D')
        ax.scatter(inData.G.nodes[r.destination]['x'], inData.G.nodes[r.destination]['y'], c='orange', s=s)
    if vehicles:
        for _, r in inData.vehicles.iterrows():
            ax.scatter(inData.G.nodes[r.pos]['x'], inData.G.nodes[r.pos]['y'], c='blue', s=s, marker='x')
    ax.scatter(inData.G.nodes[inData.stats['center']]['x'], inData.G.nodes[inData.stats['center']]['y'], c='red',
               s=10 * s, marker='+')
    plt.title(
        'Demand in {} with origins marked in green, destinations in orange'.format(params.city))
    plt.show()


def make_shareability_graph(requests, rides):
    # nodes are travellers, linked if the share
    G = nx.Graph()
    G.add_nodes_from(requests.index)
    edges = list()
    for i, row in rides.iterrows():
        if len(row.indexes)>1:
            for j, pax1 in enumerate(row.indexes):
                for k, pax2 in enumerate(row.indexes):
                    if pax1 != pax2:
                        edges.append((pax1, pax2))

    G.add_edges_from(edges)
    return G

def make_graph(requests, rides,
               mu=0.5, plot=False,
               figname=None, node_size=0.5,
               ax=None, probabilities=False):
    """
    Creates a nextworkx shareability graph and plots it if needed
    :param requests: inData.requests
    :param rides: inData.sblts.rides or inData.sblts.schedule - depending if we want results of matching or potential shareability
    :param mu: parameters for probabilites, not used
    :param plot: flag to plot the graph (may take time for big networks)
    :param figname: name to save fig
    :param node_size:
    :param ax:
    :param probabilities: flag to cal probabilities
    :return: networkX graph
    """
    if figname is not None:
        _, ax = plt.subplots()
    """ creates a graph from requests and rides, computes the probabilities"""
    G = nx.Graph()
    shift = 10 ** (round(math.log10(requests.shape[0])) + 1)
    _rides = rides.copy()
    _rides.index = _rides.index + shift
    # two kinds of nodes
    G.add_nodes_from(requests.index)
    G.add_nodes_from(_rides.index)
    # edges
    edges = list()
    for i, row in _rides.iterrows():
        for j, pax in enumerate(row.indexes):
            edges.append((i, pax, {'u': row.u_paxes[j]}))
    # probabilities
    df = pd.DataFrame(edges)
    df.columns = ['ride', 'request', 'u']
    if probabilities:
        df['u'] = df.u.apply(lambda x: x['u'])
        df['expu'] = df.u.apply(lambda x: math.exp(-mu * x))
        requests['sum_u'] = df.groupby('request').expu.sum()
        df['prob'] = df.apply(lambda x: x.expu / requests.loc[x.request].sum_u, axis=1)
    edges = list()
    for i, row in df.iterrows():
        # edges.append((row.ride,row.request, {'prob':row.prob}))
        edges.append((row.ride, row.request))
    # edges
    G.add_edges_from(edges)
    if plot:
        color_map = list()
        for node in G:
            if node < shift:
                color_map.append('grey')
            else:
                color_map.append('peru')
        pos = nx.spring_layout(G)
        # Drawing the graph
        nx.draw_networkx_nodes(G, pos, with_labels=False, node_size=node_size, font_size=7, node_color=color_map,
                               label='Travellers', ax=ax)
        nx.draw_networkx_edges(G, pos, with_labels=False, width=.3, label='Shareability', ax=ax)
        if figname is not None:
            plt.savefig(figname)
    return G

def prepare_PoA(inData):

    m = np.vstack(inData.sblts.rides['row'].values).T  # creates a numpy array for the constrains
    m = pd.DataFrame(m).astype(int)
    plt.rcParams['figure.figsize'] = [12, int(12 * inData.sblts.rides.shape[0] / inData.requests.shape[0])]
    fig, ax = plt.subplots()
    ax.imshow(m, cmap='Greys', interpolation='Nearest')
    ax.set_ylabel('rides')
    _ = ax.set_xlabel('trips')
    m.index.name = 'trips'
    m.columns.name = 'rides'
    m

    m_user_costs = m.copy()
    for col in m.columns:
        new_col = [0] * inData.sblts.rides.shape[0]
        indexes = inData.sblts.rides.loc[col]['indexes']
        u_paxes = inData.sblts.rides.loc[col].u_paxes
        for l, i in enumerate(indexes):
            new_col[i] = u_paxes[l]
        m_user_costs[col] = new_col
    m_user_costs = m_user_costs.round(1)
    m_user_costs = m_user_costs.replace(0, np.nan)
    ranking = m_user_costs.replace(0, np.nan).rank(1, ascending=True, method='first')


    inData.sblts.rides['rankings'] = inData.sblts.rides.apply(
        lambda ride: [int(ranking.loc[traveller][ride.name]) for traveller in ride.indexes], axis=1)
    inData.sblts.rides['mean_ranking'] = inData.sblts.rides.apply(lambda ride: sum(ride.rankings) / ride.degree, axis=1)
    rel_ranking = ranking.div(ranking.max(axis=1), axis=0)
    inData.sblts.rides['rel_rankings'] = inData.sblts.rides.apply(
        lambda ride: [rel_ranking.loc[traveller][ride.name] for traveller in ride.indexes], axis=1)
    inData.sblts.rides['mean_rel_ranking'] = inData.sblts.rides.apply(lambda ride: sum(ride.rel_rankings) / ride.degree,
                                                                      axis=1)
    inData.sblts.rides['PoA'] = inData.sblts.rides.apply(
        lambda ride: [m_user_costs.loc[traveller][ride.name] - m_user_costs.loc[traveller].min() for traveller in
                      ride.indexes], axis=1)
    inData.sblts.rides['mean_PoA'] = inData.sblts.rides.apply(lambda ride: sum(ride.PoA) / ride.degree, axis=1)
    inData.sblts.rides['total_PoA'] = inData.sblts.rides.apply(lambda ride: sum(ride.PoA) / ride.degree, axis=1)
    return inData




def calc_solution_PoA(inData):
    indexes = dict()
    utilities = dict()
    for _ in inData.sblts.requests.index:
        indexes[_] = list()
        utilities[_] = list()
    for i,row in inData.sblts.rides.iterrows():
        for j,traveller in enumerate(row.indexes):
            indexes[traveller] += [i]
            utilities[traveller] += [row.u_paxes[j]]

    inData.sblts.requests['ride_indexes'] = pd.Series(indexes)
    inData.sblts.requests['ride_utilities'] = pd.Series(utilities)
    inData.sblts.requests['min_utility'] = inData.sblts.requests['ride_utilities'].apply(lambda x: min(x))
    inData.sblts.requests['PoA'] = inData.sblts.requests['u_sh'] - inData.sblts.requests['min_utility']
    inData.sblts.requests['PoA_relative'] = (inData.sblts.requests['u_sh'] - inData.sblts.requests['min_utility'])/inData.sblts.requests['min_utility']
    #inData.sblts.requests['ranking'] = inData.sblts.requests.apply(lambda x: int(ranking.loc[x.name][x.ride_id]),axis =1)
    return inData
