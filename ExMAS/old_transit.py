import os
import numpy as np
import swifter
import osmnx as ox
import networkx as nx
from matplotlib.collections import LineCollection
import pandas as pd
import math
from dotmap import DotMap
import ExMAS.utils
from ExMAS.main import matching


def transitize(inData, ride, trace=False):
    """
    See if pooled door-to-door can be served stop-to-stop
    It assumes two stops per ride: pickup and dropoff
    :param inData: complete input data
    :param ride: this particular ride (result of ExMAS)
    :param plot: to generate plots (for debugging only)
    :param trace: to report full convergence
    :return: dictionary with the row for each transitized ride, with fields:
        - origin (node)
        - destination (node)
        - treq (departure time)
        - ttrav (travel time from origin to destination)
        - efficient (boolean flag if U_s2s>U_d2d for each traveller)
        - transitizable (boolean flag True if there is common accesible pickup and origin)
        - df (extra data for each traveller-ride pair:
            -
    """

    def utility_s2s(traveller):
        # utility of shared trip i for all the travellers
        return (params.price * (1 - params.s2s_discount) * traveller.dist / 1000 +
                traveller.VoT * params.WtS * (traveller.s2s_ttrav + params.delay_value * traveller.delay) +
                traveller.VoT * params.walk_discomfort * (traveller.orig_walk_time + traveller.dest_walk_time))

    # default return
    ret = pd.Series({'indexes': None,
                     'origin': None,
                     'destination': None,
                     'treq': None,
                     'df': None,
                     'transitizable': False,
                     'ttrav': None,
                     'u_veh': None,
                     'dist': None,
                     'orig_walk_time': None,
                     'dest_walk_time': None,
                     'u_pax': None,
                     'efficient': False,
                     })

    # only pooled rides
    try:
        if ride['degree'] == 1:
            return ret  # not applicable for single rides
    except:
        return ret  # not applicable for single rides

    inData.logger.warn('Transitization of pooled ride: {} of degree: {}'.format(ride.name, ride.degree))

    reqs = inData.sblts.requests.loc[ride.indexes]  # requests
    rm = inData.transitize.rm1.loc[ride.name, :][['ride', 'exp_u_private', 'exp_u_d2d', 'sum_exp', 'u', 'u_sh']].join(
        reqs[['origin', 'destination', 'dist', 'VoT']])  #
    params = inData.params

    # see if there is a common pickup point
    orig_catchments = inData.skims.walk.loc[ride.origins]  # distances
    orig_common_catchment = orig_catchments.loc[:, orig_catchments.sum() < np.inf]  # we already set dist above
    # threshold to np.inf
    if orig_common_catchment.shape[1] == 0:
        inData.logger.info('no common origin pick-up point')
        return ret  # early exit

    # see if there is a common dropoff point
    dest_catchments = inData.skims.walk.loc[ride.destinations]
    dest_common_catchment = dest_catchments.loc[:, dest_catchments.sum() < np.inf]  # filter to accessible ones only
    if dest_common_catchment.shape[1] == 0:
        inData.logger.info('no common destination pick-up point')
        return ret  # early exit

    # now we have at least one common pickup and drop off point so we can explore
    # explore
    origs_list = orig_common_catchment.columns.to_list()
    dests_list = dest_common_catchment.columns.to_list()
    treqs = reqs.set_index('origin').treq
    early = treqs.min()  # first departure time (no earlier than first request
    late = treqs.max() + params.walk_threshold  # here we need to add some slack (to let everyone access)

    inData.logger.info('Transitizing ride: {} \t Common orig points:{},  dest points: {}'.format(ride.name,
                                                                                                 len(origs_list),
                                                                                                 len(dests_list)))

    best_logsum = -np.inf  # we optimize logsum
    ret = dict()  # fresh dict to return (needed?)
    if trace:
        trace = list()  # if we want to have full report (for debuggin and visualization only)
    for orig in origs_list:  # first loop - origins
        inData.logger.info('\t ride: {} \t exploring {}-th origin: {}'.format(ride.name,
                                                                              origs_list.index(orig),
                                                                              orig))
        orig_walks = orig_common_catchment[orig]  # walking distance to currently explore origin
        orig_walks.name = 'orig_walk_time'
        df_o = rm.join(orig_walks, on='origin')  # add to rm column orig_walks (this is the return dataframe)
        min_delay = np.inf
        opt_dep = early
        for dep in range(early, late):  # optimize delay for this pick-up point (this is heuristic?)
            delays = abs((dep - orig_walks - treqs) ** 2)  # see the square of delays (to avoid imbalance)
            if delays.sum() < min_delay:  # if improves
                min_delay = delays.sum()  # overwrite the best one
                opt_dep = dep  # this is the optimal departure
        delays = abs(opt_dep - orig_walks - treqs)  # adjust the row with delays
        delays.name = 'delay'
        inData.logger.info \
            ('\t ride: {} \t Best departure time {} in range [{},{}]'.format(ride.name, opt_dep, early, late))

        df_o_t = df_o.join(delays, on='origin')  # add delays to the dataframe
        for dest in dests_list:  # loop of destinations
            # inData.logger.warn('exploring {}-th destination: {}'.format(dests_list.index(dest), dest))
            dest_walks = dest_common_catchment[dest]  # walking times from drop off
            dest_walks.name = 'dest_walk_time'
            df_o_t_d = df_o_t.join(dest_walks, on='destination')  # add to data frame
            df_o_t_d['s2s_ttrav'] = inData.skims.ride.loc[orig, dest]  # set travel times
            df_o_t_d['u_s2s'] = df_o_t_d.apply(utility_s2s, axis=1)  # compute utility for each traveller,
            # as a function of walking times, travel times, dely and fare.
            df_o_t_d['exp_u_s2s'] = (df_o_t_d.u_s2s * params.mode_choice_beta).apply(math.exp)  # exponent it
            df_o_t_d['sum_exp'] = df_o_t_d['exp_u_s2s'] + df_o_t_d['sum_exp']  # adjust denominator of MNL
            df_o_t_d['prob_s2s'] = df_o_t_d['exp_u_s2s'] / df_o_t_d['sum_exp']  # calculate log probability
            df_o_t_d['prob_s2s'] = df_o_t_d['prob_s2s'].apply(math.log)
            obj_fun = df_o_t_d['prob_s2s'].sum()  # sum of log probabilities (can be log sum, but that works fine)
            if trace:  # full report
                trace.append([orig, dest, opt_dep, obj_fun, orig_walks, dest_walks, delays, df_o_t_d['u_s2s'].sum()])
            if obj_fun > best_logsum:  # improvement
                inData.logger.info('\t Best solution {:.4f} improved to {:.4f}. '
                                   '\n \t orig:{} dep:{} dest:{}'.format(best_logsum,
                                                                         obj_fun,
                                                                         orig, dep, dest))
                # output
                ret = {'indexes': ride.indexes,
                       'origin': orig,
                       'destination': dest,
                       'treq': dep,
                       'df': df_o_t_d,
                       "transitizable": True
                       }
                best_logsum = obj_fun

    # check efficiency
    # ret['df']['efficient'] = ret['df']['u_s2s'] <= ret['df']['u'] # see if utilities of s2s are greater than private
    ret['df']['efficient'] = ret['df']['u_s2s'] <= ret['df']['u_sh']  # see if costs of s2s are lower than d2d
    ret['efficient'] = ret['df']['efficient'].eq(True).all()  # for all the travellers
    if ret['efficient']:
        ret['ttrav'] = ret['df'].s2s_ttrav.max()
        ret['u_veh'] = ret['ttrav']
        ret['times'] = [ret['treq'],ret['df'].s2s_ttrav.max()]
        ret['dist'] = inData.skims.dist.loc[ret['origin'], ret['destination']]
        ret['orig_walk_time'] = ret['df'].orig_walk_time.sum()
        ret['dest_walk_time'] = ret['df'].dest_walk_time.sum()
        ret['u_pax'] = ret['df'].u_s2s.sum()
        ret['u_paxes'] = ret['df'].u_s2s.values


    inData.logger.warn('ride: {} \t Efficiency check: {} \t {}/{} efficient'.format(ride.name,
                                                                                    ret['efficient'],
                                                                                    ret['df']['efficient'].sum(),
                                                                                    ride.degree))
    return pd.Series(ret)


def level1(inData, params):
    inData = ExMAS.main(inData, params, plot=False)  # compute door-to-door pooled rides
    inData.logger.warn('ExMAS generated {} rides'.format(inData.sblts.rides.shape[0]))
    inData = process_level1(inData, params)  # prepare data structures to transitize pooled rides
    return inData


def process_level1(inData, params):
    # computes skim materices, prepares data structures to transitize ExMAS results
    inData.sblts.rides['degree'] = inData.sblts.rides.apply(lambda x: len(x.indexes), axis=1)
    inData.sblts.rides.to_csv('{}_ExMASrides1.csv'.format(EXPERIMENT_NAME))

    inData.skims = DotMap(_dynamic=False)  # skim matrices of the network
    inData.skims.dist = inData.skim.copy()  # distance (meters)
    inData.skims.ride = inData.skims.dist.divide(params.speeds.ride).astype(int).T  # travel time (seconds)
    inData.skims.walk = inData.skims.dist.divide(params.speeds.walk).astype(int).T  # walking time (seconds)
    inData.skims.walk = inData.skims.walk.mask(inData.skims.walk > params.walk_threshold, np.inf)  # inf if above max

    inData.sblts.rides['origins'] = inData.sblts.rides.apply(
        lambda ride: list(inData.sblts.requests.loc[ride.indexes_orig].origin), axis=1)  # list orig nodes for each ride
    inData.sblts.rides['destinations'] = inData.sblts.rides.apply(
        lambda ride: list(inData.sblts.requests.loc[ride.indexes_dest].destination),
        axis=1)  # list dest nodes for each ride
    inData.sblts.rides['deps'] = inData.sblts.rides.apply(
        lambda ride: list(inData.sblts.requests.loc[ride.indexes_orig].treq),
        axis=1)  # list departure times for each traveller
    inData.sblts.rides['dep_deltas'] = inData.sblts.rides.apply(lambda ride: list(
        inData.sblts.requests.loc[ride.indexes_orig].treq - inData.sblts.requests.loc[ride.indexes_orig].treq.mean()),
                                                                axis=1)  # list depatrure delay for each traveller

    inData.transitize.rm1 = ExMAS.utils.make_traveller_ride_matrix(
        inData)  # data frame with two indices: ride - traveller
    inData.transitize.rm1['exp_u_private'] = (inData.transitize.rm1.u * params.mode_choice_beta).apply(
        math.exp)  # utilities
    inData.transitize.rm1['exp_u_d2d'] = (inData.transitize.rm1.u_sh * params.mode_choice_beta).apply(
        math.exp)  # exp sum
    inData.transitize.rm1['sum_exp'] = inData.transitize.rm1['exp_u_d2d'] + inData.transitize.rm1[
        'exp_u_private']  # to speed up calculations


    inData.transitize.rm = inData.transitize.rm1.copy()[['ride', 'traveller', 'degree', 'dist', 'ttrav', 'delay', 'u_sh']]
    inData.transitize.rm['u'] = inData.transitize.rm['u_sh']
    del inData.transitize.rm['u_sh']

    # store d2d results
    inData.transitize.requests1 = inData.sblts.requests.copy()  # store requests

    inData.transitize.requests1.to_csv('{}_requests.csv'.format(EXPERIMENT_NAME))
    inData.transitize.rides1 = inData.sblts.rides.copy()  # store d2d rides
    inData.transitize.solution1 = inData.transitize.rides1[inData.transitize.rides1.selected == True].copy()
    inData.transitize.rm1.to_csv('{}_rm1.csv'.format(EXPERIMENT_NAME))  # store ExMAS solution

    inData.transitize.rides = inData.transitize.rides1.copy()

    inData.transitize.rides['solution_0'] = inData.transitize.rides.apply(lambda x: 1 if x.kind == 1 else 0, axis=1)
    inData.transitize.rides.kind = inData.transitize.rides.apply(lambda x: 'p' if x.kind == 1 else 'd2d', axis=1)
    inData.transitize.rides['solution_1'] = inData.transitize.rides.apply(lambda x: x.selected, axis=1)

    inData.transitize.rides['orig_walk_time'] = 0
    inData.transitize.rides['dest_walk_time'] = 0
    inData.transitize.rides['ttrav'] = inData.transitize.rides.apply(lambda x: sum(x.ttravs), axis=1)
    inData.transitize.rides = inData.transitize.rides[
        ['indexes', 'indexes_orig','indexes_dest',
         'u_pax', 'u_veh', 'kind',
         'ttrav', 'times', 'u_paxes',
         'orig_walk_time', 'dest_walk_time',
         'solution_0', 'solution_1']]

    return inData


def level_2(inData, params):
    to_swifter = inData.sblts.rides[inData.sblts.rides.degree != 1]
    if params.get('swifter', False):
        from_swifter = to_swifter.swifter.apply(lambda x: transitize(inData, x), axis=1)
    else:
        from_swifter = to_swifter.apply(lambda x: transitize(inData, x), axis=1)  # main

    inData.transitize.requests2 = from_swifter

    if inData.transitize.requests2[inData.transitize.requests2.transitizable].shape[0] > 0:
        inData.transitize.rm2 = pd.concat(
            inData.transitize.requests2[inData.transitize.requests2.transitizable].df.values)
    inData.transitize.rm2['pax_id'] = inData.transitize.rm2.index.copy()
    inData.transitize.rm2['traveller'] = inData.transitize.rm2.index.copy()

    inData.transitize.requests2 = inData.transitize.requests2[inData.transitize.requests2['efficient']]
    inData.transitize.requests2 = inData.transitize.requests2.apply(pd.to_numeric, errors='ignore')

    if inData.transitize.requests2.shape[0] == 0:
        inData.logger.warn('No transitable rides')
        inData.transitize.rides.to_csv('rides.csv')

    else:
        inData.logger.warn('Transitizing: \t{} rides '
                           '\t{} transitizable '
                           '\t{} efficient'.format(inData.transitize.rides1.shape[0],
                                                   inData.transitize.requests2[
                                                       inData.transitize.requests2.transitizable].shape[0],
                                                   inData.transitize.requests2.shape[0]))
        inData.transitize.rm2.to_csv('{}_rm2.csv'.format(EXPERIMENT_NAME))


        inData.transitize.requests2['indexes_set'] = inData.transitize.requests2.apply(lambda x: set(x.indexes), axis=1)

        # set the indexes of first level rides in the second level rides
        inData.transitize.requests2['low_level_indexes'] = inData.transitize.requests2.apply(
            lambda x: inData.transitize.rm1[inData.transitize.rm1.ride == x.name].traveller.to_list(),
            axis=1)
        inData.transitize.requests2['low_level_indexes_set'] = inData.transitize.requests2.low_level_indexes.apply(set)
        inData.transitize.requests2['low_level_indexes'] = inData.transitize.requests2.low_level_indexes.apply(list)
        inData.transitize.requests2['pax_id'] = inData.transitize.requests2.index.copy()

        to_concat = inData.transitize.requests2
        to_concat['solution_0'] = 0
        to_concat['solution_1'] = 0
        to_concat['d2d_reference'] = to_concat.pax_id
        to_concat['kind'] = 's2s'
        to_concat = to_concat[
            ['indexes', 'u_pax', 'u_veh', 'kind', 'ttrav',
             'orig_walk_time', 'dest_walk_time', 'times', 'u_paxes',
             'solution_0', 'solution_1', 'low_level_indexes_set', 'origin', 'destination', 'd2d_reference']]
        inData.transitize.rides = pd.concat([inData.transitize.rides, to_concat]).reset_index()

        to_concat = inData.transitize.rm2

        to_concat['ttrav'] = to_concat['s2s_ttrav']

        inData.transitize.rm = pd.contat([inData.transitize.rm,to_concat[['ride', 'traveller', 'dist',
                                                                 'ttrav', 'delay', 'u_sh',
                                                                 'origin_walk_time', 'dest_walk_time', 'delay']]])

        inData.transitize.requests2.index = inData.transitize.rides[inData.transitize.rides.kind == 's2s'].index.values

        #inData.transitize.requests2.to_csv('{}_requests2.csv'.format(EXPERIMENT_NAME))

        inData.sblts.rides = inData.transitize.rides
        params.process_matching = False

        inData = matching(inData, params)  # solution with s2s
        inData.transitize.rides['solution_2'] = inData.sblts.rides.selected.values
        #inData.transitize.rides.to_csv('{}_rides2.csv'.format(EXPERIMENT_NAME))

        return inData, params


def prepare_level3(inData, params):
    params.shared_discount = params.second_level_shared_discount
    params.WtS = params.multi_stop_WtS

    params.reset_ttrav = False  # so that travel times are not divided by avg_speed again
    params.VoT_std = False  #
    params.make_assertion_matching = False
    params.make_assertion_extension = False
    params.make_assertion_pairs = False
    params.without_matching = True

    inData.requests = inData.transitize.requests2  # set the new requests for ExMAS
    params.nP = inData.requests.shape[0]

    return inData, params


def level_3(inData, params):
    # set parameters for the second level of ExMAS
    inData, params = prepare_level3(inData, params)
    inData = list_unmergables(inData)
    inData_copy = ExMAS.main(inData, params, plot=False)  # pooling of 2nd level rides (s2s)
    inData.transitize.rides3 = inData_copy.sblts.rides.copy()

    inData_copy.sblts.rides.to_csv('{}_ridesExMAS2.csv'.format(EXPERIMENT_NAME))
    inData = process_level3(inData)

    inData = matching(inData, params)  # final solution
    #inData.sblts.requests.to_csv('{}_requestsExMAS2.csv'.format(EXPERIMENT_NAME))

    inData.transitize.rides['solution_3'] = inData.sblts.rides.selected.values

    inData.transitize.rides.to_csv('{}_rides.csv'.format(EXPERIMENT_NAME))

    return inData, params


def process_level3(inData):
    # update indexes looking at travellers in the first level rides
    for col in ['orig_walk_time', 'dest_walk_time', 'ttrav']:
        inData.transitize.rides3[col] = \
            inData.transitize.rides3.indexes.apply(lambda x: inData.transitize.requests2.loc[x][col].sum())
    inData.transitize.rides3['high_level_indexes'] = inData.transitize.rides3['indexes'].copy()
    inData.transitize.rides3['indexes'] = inData.transitize.rides3.apply(
        lambda x: sum(inData.transitize.requests2.loc[x.indexes].low_level_indexes.to_list(), []), axis=1)
    inData.transitize.rides3['degree'] = inData.transitize.rides3.apply(lambda x: len(x.high_level_indexes), axis=1)
    to_concat = inData.transitize.rides3[inData.transitize.rides3.degree>1]
    to_concat['solution_0'] = 0
    to_concat['solution_1'] = 0
    to_concat['solution_2'] = 0
    to_concat['kind'] = 'ms'
    to_concat = to_concat[
        ['indexes', 'indexes_orig','indexes_dest', 'high_level_indexes',
         'u_pax', 'u_veh', 'ttrav', 'kind', 'times', 'u_paxes',
         'orig_walk_time', 'dest_walk_time',
         'solution_0', 'solution_1', 'solution_2']]
    inData.transitize.rides = pd.concat([inData.transitize.rides, to_concat]).reset_index()

    #inData.transitize.rides.to_csv('{}_rides3.csv'.format(EXPERIMENT_NAME))

    inData.sblts.rides = inData.transitize.rides
    inData.sblts.requests = inData.transitize.requests1
    inData.requests = inData.transitize.requests1

    return inData


def list_unmergables(inData):
    """
    lists pairs of rides that cannot be merged together (they contain the same requests
    they cannot be used in the formation of 2nd degree ExMAS rides (stop-to-stop)
    they are then used in ExMAS to filter during explorations."""
    df = inData.transitize.rides[inData.transitize.rides.kind == 's2s']

    # df = inData.transitize.requests2

    def unmergables(row):
        # returns list of all the subgroup indiced contained in a ride
        return df[df.low_level_indexes_set.apply(
            lambda x: len(x.intersection(row.low_level_indexes_set))) > 0].index.to_list()

    df['unmergables'] = df.apply(unmergables, axis=1)

    unmergables = list()
    for i, row in df.iterrows():
        for _ in row.unmergables:
            if row.name != _:
                if not {row.name, _} in unmergables:
                    unmergables.append({row.name, _})

    inData['unmergables'] = unmergables
    inData.logger.warn('{} unmergable pairs from ExMAS'.format(len(inData.unmergables)))
    inData.logger.warn(inData.unmergables)
    return inData


def pipeline():
    from ExMAS.utils import inData as inData

    # BIGGER
    if DEBUG:
        params = ExMAS.utils.get_config('ExMAS/data/configs/transit_debug.json')  # load the default
        # small configuration
        params.nP = 100
        params.pax_delay = 0
        params.simTime = 0.2
        params.speeds.ride = 8
        params.mode_choice_beta = -0.5  # only to estimate utilities
        params.speeds.walk = 1.2
        params.walk_discomfort = 1
        params.delay_value = 1.5
        params.walk_threshold = 300
        params.shared_discount = 0.2
        params.s2s_discount = 0.95
        params.VoT_std = params.VoT / 8
        params.multi_stop_WtS = 1
        params.multistop_discount = 0.98
        params.second_level_shared_discount = ((1 - params.s2s_discount) - (1 - params.multistop_discount)) / (
                1 - params.s2s_discount)
        params.without_matching = False
    else:
        params = ExMAS.utils.get_config('ExMAS/data/configs/transit.json')  # load the default
        params.nP = 700  # number of trips
        params.simTime = 0.25  # per simTime hours
        params.mode_choice_beta = -0.3  # only to estimate utilities of pickup points
        params.VoT = 0.0035  # value of time (eur/second)
        params.VoT_std = params.VoT / 8  # variance of Value of Time

        params.speeds.walk = 1.2  # speed of walking (m/s)
        params.speeds.ride = 7

        params.walk_discomfort = 1  # walking discomfort factor
        params.delay_value = 1.2  # delay discomfort factor
        params.pax_delay = 0  # extra seconds for each pickup and drop off
        params.walk_threshold = 300  # maximal walking distance (per origin or destination)

        params.price = 1.5  # per kilometer fare
        params.shared_discount = 0.25 # discount for door to door pooling
        params.s2s_discount = 0.66  # discount for stop to stop pooling
        params.multistop_discount = 0.8  # discount for multi-stop

        params.multi_stop_WtS = 1  # willingness to share in multi-stop pooling (now lowe)

        params.second_level_shared_discount = ((1 - params.s2s_discount) - (1 - params.multistop_discount)) / (
                1 - params.s2s_discount)
        # how much we reduce multi-stop trip related to stop-to-stop
        params.without_matching = False  # we do not do matching now

    inData.params = params  # store params internally

    inData = ExMAS.utils.load_G(inData, params, stats=True)  # download the graph

    if DEBUG:
        inData = ExMAS.utils.generate_demand(inData, params)  # generate demand of requests
    else:
        inData = ExMAS.utils.load_albatross_csv(inData, params)

    inData = level1(inData, params)  # I ExMAS d2d

    inData, params = level_2(inData, params)  # II ExMAS s2s
    if inData.transitize.requests2.shape[0] == 0:
        return 0  # early exit

    inData, params = level_3(inData, params)

    return inData.transitize


if __name__ == "__main__":
    DEBUG = False
    EXPERIMENT_NAME = 'times'

    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, '..'))
    pipeline()
