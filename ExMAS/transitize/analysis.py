"""
# ExMAS - TRANSITIZE
> Exact Matching of Attractive Shared rides (ExMAS) for system-wide strategic evaluations
> Module to pool requests to stop-to-stop and multi-stop rides, aka TRANSITIZE
---

Analyse the results



----
Rafał Kucharski, TU Delft,GMUM UJ  2021 rafal.kucharski@uj.edu.pl
"""




from ExMAS.utils import read_csv_lists

from dotmap import DotMap
import os
import pandas as pd



def process_transitize(inData, params):
    rm = inData.transitize.rm.join(inData.transitize.requests1[['VoT', 'origin', 'destination', 'treq']],
                                   on='traveller')
    rm = rm.join(inData.transitize.rides[['kind']], on='ride')

    inData.transitize.rm = rm

    inData = process_rm_multistop(inData, params)
    inData = assign_solutions(inData)
    inData = determine_fares(inData, params)
    #inData.transitize.requests = PT_utility(inData.transitize.requests, params)

    inData.transitize.rm = inData.transitize.rm.join(
        inData.transitize.rides[['solution_0', 'solution_1', 'solution_2', 'solution_3']], on='ride')

    return inData


def load_results(PATH, EXP_NAME = None, inData=None):
    if inData is None:
        from ExMAS.utils import inData as inData
    inData.transitize = DotMap(_dynamic=False)

    for file in os.listdir(PATH):
        if file.endswith(".csv"):
            if EXP_NAME is None or file.startswith(EXP_NAME):
                inData.transitize[file.split("_")[-1][:-4]] = read_csv_lists(
                    pd.read_csv(os.path.join(PATH, file), index_col=0))
                if "row" in inData.transitize[file.split("_")[-1][:-4]].columns:
                    del inData.transitize[file.split("_")[-1][:-4]]['row']
    return inData


def prep_results(PATH, inData=None, params=None):
    # reads directory with csv files and stores them into inData.transitize container
    # reads lists from csv as lists
    if inData is None:
        from ExMAS.utils import inData as inData
    inData.transitize = DotMap(_dynamic=False)
    for file in os.listdir(PATH):
        if file.endswith(".csv"):
            inData.transitize[file.split("_")[1][:-4]] = read_csv_lists(
                pd.read_csv(os.path.join(PATH, file), index_col=0))
            if "row" in inData.transitize[file.split("_")[1][:-4]].columns:
                del inData.transitize[file.split("_")[1][:-4]]['row']

    rm = inData.transitize.rm.join(inData.transitize.requests[['VoT', 'origin', 'destination', 'treq']],
                                   on='traveller')
    rm = rm.join(inData.transitize.rides[['kind']], on='ride')

    inData.transitize.rm = rm

    inData = process_rm_multistop(inData, params)
    inData = assign_solutions(inData)
    inData = determine_fares(inData, params)
    inData.transitize.requests = PT_utility(inData.transitize.requests, params)

    inData.transitize.rm = inData.transitize.rm.join(
        inData.transitize.rides[['solution_0', 'solution_1', 'solution_2', 'solution_3']], on='ride')

    return inData


def assign_solutions(inData):
    requests = inData.transitize.requests1
    for level in [0, 1, 2, 3]:
        ret = dict()
        rides = inData.transitize.rides[inData.transitize.rides['solution_{}'.format(level)] == 1]
        for i, request in requests.iterrows():
            ret[request.name] = rides[rides.apply(lambda ride: request.name in ride.indexes, axis=1)].index[0]
        requests['ride_solution_{}'.format(level)] = pd.Series(ret)
    inData.transitize.requests1 = requests
    return inData


def process_rm_multistop(inData, params):
    rm = inData.transitize.rm

    def calc_deps(r):
        deps = [r.times[0]]
        for d in r.times[1:r.degree]:
            deps.append(deps[-1] + d)  # departure times
        return deps

    ret = list()

    for i, ms in inData.transitize.rides[inData.transitize.rides.kind == 'ms'].iterrows():
        ms.degree = len(ms.indexes)
        ms['deps'] = calc_deps(ms)
        # ms['indexes_orig'] = json.loads(ms['indexes_orig'])
        # ms['indexes_dest'] = json.loads(ms['indexes_dest'])
        # ms['high_level_indexes'] = json.loads(ms['high_level_indexes'])
        df = rm[rm.ride.isin(ms.high_level_indexes)].copy()

        df['s2s_reference'] = df.ride.astype(int).copy()
        df['ride'] = ms.name
        df['degree'] = len(ms.indexes)
        df['kind'] = 'ms'
        df['ttrav'] = df.apply(
            lambda x: ms.deps[ms.indexes_dest.index(x.s2s_reference) + len(ms.high_level_indexes)] - ms.deps[
                ms.indexes_orig.index(x.s2s_reference)], axis=1)
        df['door_departure'] = df.apply(lambda x: ms.deps[ms.indexes_orig.index(x.s2s_reference)] - x.orig_walk_time,
                                        axis=1)
        df['delay'] = df.apply(lambda x: ms.deps[ms.indexes_orig.index(x.s2s_reference)] - x.orig_walk_time - x.treq,
                               axis=1)
        ret.append(df)
    df = pd.concat(ret)

    def utility_s2s(traveller):
        # utility of shared trip i for all the travellers
        return (params.price * (1 - params.multistop_discount) * traveller.dist / 1000 +
                traveller.VoT * params.WtS * (traveller.ttrav + params.delay_value * traveller.delay) +
                traveller.VoT * params.walk_discomfort * (traveller.orig_walk_time + traveller.dest_walk_time))

    df['u'] = df.apply(utility_s2s, axis=1)

    inData.transitize.rm = pd.concat([inData.transitize.rm, df])
    return inData

def PT_utility(requests,params):
    if 'walkDistance' in requests.columns:
        requests = requests
        walk_factor = 2
        wait_factor = 2
        transfer_penalty = 500
        requests['PT_fare'] = 1 + requests.transitTime * params.avg_speed/1000 * 0.175
        requests['u_PT'] = requests['PT_fare'] + \
                           requests.VoT * (walk_factor * requests.walkDistance / params.speeds.walk +
                                           wait_factor * requests.waitingTime +
                                           transfer_penalty * requests.transfers + requests.transitTime)
    return requests






def determine_fares(inData, params):
    def fares(row):

        if row.kind == 'p':
            f = params.price
        elif row.kind == 'd2d':
            f = params.price * (1-params.shared_discount)
        elif row.kind == 's2s':
            f = params.price * (1-params.s2s_discount)
        elif row.kind == 'ms':
            f = params.price * (1-params.multistop_discount)
        else:
            f = params.price  # shall raise exception
        return f * row.dist / 1000

    inData.transitize.rm['fare'] = inData.transitize.rm.apply(fares, axis=1)
    inData.transitize.rides['fare'] = inData.transitize.rides.apply(lambda x:
                                                                    inData.transitize.rm[
                                                                        inData.transitize.rm.ride == x.name].fare.sum(),
                                                                    axis=1)

    return inData


def make_report(inData):
    ret = dict()

    KPIs = ['u_veh', 'u_pax', 'ttrav', 'orig_walk_time', 'dest_walk_time', 'fare']
    kinds = ['p', 'd2d', 's2s', 'ms']
    for i in [0, 1, 2, 3]:
        ret[i] = inData.transitize.rides[inData.transitize.rides['solution_{}'.format(i)] == 1][KPIs].sum()
        #ret[i] = pd.concat([ret[i],
        #                    inData.transitize.rides[inData.transitize.rides['solution_{}'.format(i)] == 1].groupby(
        #                        'kind').size()])
        ret[i] = pd.concat([ret[i],
                            inData.transitize.rm[inData.transitize.rm['solution_{}'.format(i)] == 1].groupby(
                                'kind').size()])
        inds = inData.transitize.rides[inData.transitize.rides['solution_{}'.format(i)] == 1].indexes
        if 'requests1' in inData.transitize:
            ret[i]['test'] = len(sum(inds, [])) == inData.transitize.requests1.shape[0]
        else:
            ret[i]['test'] = len(sum(inds, [])) == inData.transitize.requests.shape[0]
        ret[i]['nRides'] = (0 if i == 0 else ret[i - 1]['nRides']) + \
                           inData.transitize.rides[inData.transitize.rides.kind == kinds[i]].shape[0]


    inData.transitize.report = pd.DataFrame(ret).T
    inData.transitize.report['efficiency'] = inData.transitize.report['fare'] / inData.transitize.report['u_veh']*3600
    inData.transitize.report['occupancy'] = inData.transitize.requests.ttrav.sum() / inData.transitize.report['u_veh']
    inData.transitize.report = inData.transitize.report.T
    return inData
