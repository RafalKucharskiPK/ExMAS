from matplotlib.collections import LineCollection
from ExMAS.utils import plot_map_rides, read_csv_lists
import osmnx as ox
from dotmap import DotMap
import os
import pandas as pd
import networkx as nx
import seaborn as sns
import json


def process_transitize(inData,params):
    rm = inData.transitize.rm.join(inData.transitize.requests1[['VoT', 'origin', 'destination', 'treq']], on='traveller')
    rm = rm.join(inData.transitize.rides[['kind']], on='ride')

    inData.transitize.rm = rm

    inData = process_rm_multistop(inData, params)
    inData = assign_solutions(inData)
    inData = determine_fares(inData, params)

    inData.transitize.rm = inData.transitize.rm.join(
        inData.transitize.rides[['solution_0', 'solution_1', 'solution_2', 'solution_3']], on='ride')

    return inData


def prep_results(PATH, inData=None, params = None):
    inData.transitize = DotMap(_dynamic = False)

    # reads directory with csv files and stores them into inData.transitize container
    # reads lists from csv as lists
    if inData is None:
        from ExMAS.utils import inData as inData
    for file in os.listdir(PATH):
        if file.endswith(".csv"):
            inData.transitize[file.split("_")[1][:-4]] = read_csv_lists(
                pd.read_csv(os.path.join(PATH, file), index_col=0))
            if "row" in inData.transitize[file.split("_")[1][:-4]].columns:
                del inData.transitize[file.split("_")[1][:-4]]['row']

    rm = inData.transitize.rm.join(inData.transitize.requests1[['VoT', 'origin', 'destination', 'treq']], on='traveller')
    rm = rm.join(inData.transitize.rides[['kind']], on='ride')


    inData.transitize.rm = rm


    inData = process_rm_multistop(inData, params)
    inData = assign_solutions(inData)
    inData = determine_fares(inData, params)

    inData.transitize.rm = inData.transitize.rm.join(inData.transitize.rides[['solution_0', 'solution_1', 'solution_2', 'solution_3']], on='ride')



    return inData



def assign_solutions(inData):

    rides = inData.transitize.rides
    requests = inData.transitize.requests
    for level in [0,1,2,3]:
        ret = dict()
        rides = inData.transitize.rides[inData.transitize.rides['solution_{}'.format(level)]==1]
        for i, request in requests.iterrows():
            ret[request.name] = rides[rides.apply(lambda ride: request.name in ride.indexes, axis = 1)].index[0]
        requests['ride_solution_{}'.format(level)] = pd.Series(ret)
    inData.transitize.requests = requests
    return inData


def process_rm_multistop(inData, params):

    rm = inData.transitize.rm
    def calc_deps(r):
        deps = [r.times[0]]
        for d in r.times[1:r.degree]:
            deps.append(deps[-1] + d)  # departure times
        return deps
    ret = list()



    for i,ms in inData.transitize.rides[inData.transitize.rides.kind=='ms'].iterrows():
        ms.degree = len(ms.indexes)
        ms['deps'] = calc_deps(ms)
        #ms['indexes_orig'] = json.loads(ms['indexes_orig'])
        #ms['indexes_dest'] = json.loads(ms['indexes_dest'])
        #ms['high_level_indexes'] = json.loads(ms['high_level_indexes'])
        df = rm[rm.ride.isin(ms.high_level_indexes)].copy()


        df['s2s_reference'] = df.ride.astype(int).copy()
        df['ride'] = ms.name
        df['degree'] = len(ms.indexes)
        df['ttrav'] = df.apply(lambda x: ms.deps[ms.indexes_dest.index(x.s2s_reference)+len(ms.high_level_indexes)] - ms.deps[ms.indexes_orig.index(x.s2s_reference)], axis = 1)
        df['door_departure'] = df.apply(lambda x: ms.deps[ms.indexes_orig.index(x.s2s_reference)] - x.orig_walk_time, axis = 1)
        df['delay'] = df.apply(lambda x: ms.deps[ms.indexes_orig.index(x.s2s_reference)] - x.orig_walk_time - x.treq, axis = 1)
        ret.append(df)
    df = pd.concat(ret)

    def utility_s2s(traveller):
        # utility of shared trip i for all the travellers
        return (params.price * (1 - params.multistop_discount) * traveller.dist / 1000 +
                traveller.VoT * params.WtS * (traveller.ttrav + params.delay_value * traveller.delay) +
                traveller.VoT * params.walk_discomfort * (traveller.orig_walk_time + traveller.dest_walk_time))
    df['u'] = df.apply(utility_s2s,axis = 1)


    inData.transitize.rm = pd.concat([inData.transitize.rm, df])
    return inData

def determine_fares(inData, params):


    def fares(row):

        if row.kind == 'p':
            f = params.price
        elif row.kind == 'd2d':
            f = params.price * params.shared_discount
        elif row.kind == 's2s':
            f = params.price * params.s2s_discount
        elif row.kind == 'ms':
            f = params.price * params.multistop_discount
        else:
            f = params.price # shall raise exception
        return f*row.dist/1000

    inData.transitize.rm['fare'] = inData.transitize.rm.apply(fares,axis = 1)




    return inData




def make_report(inData):
    ret = dict()
    KPIs = ['u_veh', 'u_pax', 'ttrav', 'orig_walk_time', 'dest_walk_time']
    kinds = ['p','d2d','s2s','ms']
    for i in [0, 1, 2, 3]:
        ret[i] = inData.transitize.rides[inData.transitize.rides['solution_{}'.format(i)] == 1][KPIs].sum()
        ret[i] = pd.concat([ret[i],
                            inData.transitize.rides[inData.transitize.rides['solution_{}'.format(i)] == 1].groupby(
                                'kind').size()])
        inds = inData.transitize.rides[inData.transitize.rides['solution_{}'.format(i)] == 1].indexes
        ret[i]['test'] = len(sum(inds, [])) == inData.transitize.requests.shape[0]
        ret[i]['nRides'] = (0 if i == 0 else ret[i-1]['nRides']) + inData.transitize.rides[inData.transitize.rides.kind == kinds[i]].shape[0]

    inData.transitize.report = pd.DataFrame(ret)


    return inData