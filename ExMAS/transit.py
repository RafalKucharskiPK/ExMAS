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


# def transitize(inData, ride, plot=False):
#     # loops over all the shared rides generated with ExMAS
#     # and identifies such rides which can be transitized, i.e. single pick-up and single drop-off point
#
#
#     transitizable = False #
#     ret = {"transitizable": False,
#            'orig': None,
#            'dest': None,
#            "orig_walk_times": list(),
#            'orig_walk_times': list()}
#
#     # see if there is a common pickup point
#     if ride.degree > 1:
#         orig_catchments = inData.skims.walk.loc[ride.origins]
#         # orig_catchments = orig_catchments.mask(orig_catchments> params.walk_threshold, np.inf)
#         orig_common_catchment = orig_catchments.loc[:, orig_catchments.sum() < np.inf]
#         if orig_common_catchment.shape[1] > 0:
#             dest_catchments = inData.skims.walk.loc[ride.destinations]
#             # dest_catchments = dest_catchments.mask(dest_catchments> params.walk_threshold, np.inf)
#             dest_common_catchment = dest_catchments.loc[:, dest_catchments.sum() < np.inf]
#             if dest_common_catchment.shape[1] > 0:
#                 transitizable = True
#     if transitizable:
#         ret['transitizable'] = True
#         ret['orig'] = (orig_common_catchment ** 2).sum().idxmin()
#         ret['dest'] = (dest_common_catchment ** 2).sum().idxmin()
#         ret['orig_walk_times'] = orig_common_catchment[ret['orig']].values
#         ret['dest_walk_times'] = dest_common_catchment[ret['dest']].values
#         ret['arrivals'] = ride.deps + ret['orig_walk_times']
#
#     if plot and transitizable:
#         d = dict()
#         for node in inData.G.nodes:
#             d[node] = (orig_catchments ** 2)[node].sum() / 15000 if (orig_catchments[node].sum() < np.inf) else 0
#             d[node] = (dest_catchments ** 2)[node].sum() / 15000 if (dest_catchments[node].sum() < np.inf) else d[node]
#         ride['orig'] = ret['orig']
#         ride['dest'] = ret['dest']
#         plot_ride(inData, ride, sizes=pd.Series(d))
#
#     return ret
#
#
# def gimme(inData):
#     trans = inData.sblts.rides.apply(lambda x: transitize(inData,x, plot=False), axis = 1)
#     return trans
# #     for r,ride in inData.sblts.rides[inData.sblts.rides.degree>1].iterrows():
# #         transitize(inData,ride, plot = True):


def add_route(G, ax, route, color='grey', lw=2, alpha=0.5):
    # plots route on the graph alrready plotted on ax
    edge_nodes = list(zip(route[:-1], route[1:]))
    lines = []
    for u, v in edge_nodes:
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(), key=lambda x: x['length'])
        # if it has a geometry attribute (ie, a list of line segments)
        if 'geometry' in data:
            # add them to the list of lines to plot
            xs, ys = data['geometry'].xy
            lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)
    lc = LineCollection(lines, colors=color, linewidths=lw, alpha=alpha, zorder=3)
    ax.add_collection(lc)


def plot_ride(inData, ride, sizes=0, fig = None, ax = None, label_offset = 0.0005):
    G = inData.G
    routes = list()
    if fig is None:
        fig, ax = ox.plot_graph(G, figsize=(25, 25), node_size=sizes, node_color='black', edge_color='grey',
                                bgcolor='white', edge_linewidth=0.3,
                                show=False, close=False)
    for i, origin in enumerate(ride.origins):
        ax.scatter(G.nodes[origin]['x'], G.nodes[origin]['y'], s=10, c='red', marker='o')
        ax.annotate('{}'.format(ride.indexes[i]),
                    (G.nodes[origin]['x'] + label_offset, G.nodes[origin]['y'] + label_offset)
                    , bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'), fontsize=8)

        routes.append(nx.shortest_path(G, ride.origins[i], ride.origin, weight='length'))
        routes.append(nx.shortest_path(G, ride.destination, ride.destinations[i], weight='length'))
    for i, origin in enumerate(ride.destinations):
        ax.scatter(G.nodes[origin]['x'], G.nodes[origin]['y'], s=10, c='blue', marker='o')
        ax.annotate('{}'.format(ride.indexes[i]), (G.nodes[origin]['x'] + label_offset, G.nodes[origin]['y'] + label_offset),
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), fontsize=8)

    transit_route = nx.shortest_path(G, ride.origin, ride.destination, weight='length')
    add_route(G, ax, transit_route, color='red', lw=4, alpha=0.5)

    for route in routes:
        add_route(G, ax, route, color='green', lw=4, alpha=0.5)
    ax.scatter(G.nodes[ride.origin]['x'], G.nodes[ride.origin]['y'], s=50, c='red', marker='o')
    ax.scatter(G.nodes[ride.destination]['x'], G.nodes[ride.destination]['y'], s=50, c='red', marker='o')
    return fig, ax


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
    ret = pd.Series({'indexes': ride.indexes,
           'origin': None,
           'destination': None,
           'treq': None,
           'ttrav': None,
           'df': None,
           'efficient': False,
           'transitizable': False,
           })

    # only pooled rides
    if ride.degree == 1:
        return ret  # not applicable for single rides

    inData.logger.warn('Transitization of pooled ride: {} of degree: {}'.format(ride.name, ride.degree))

    reqs = inData.sblts.requests.loc[ride.indexes]  # requests
    rm = inData.sblts.rm.loc[ride.name, :][['ride', 'exp_u_private', 'exp_u_d2d', 'sum_exp', 'u', 'u_sh']].join(
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
        for dep in range(early, late): # optimize delay for this pick-up point (this is heuristic?)
            delays = abs((dep - orig_walks - treqs) ** 2) # see the square of delays (to avoid imbalance)
            if delays.sum() < min_delay: # if improves
                min_delay = delays.sum() # overwrite the best one
                opt_dep = dep # this is the optimal departure
        delays = abs(opt_dep - orig_walks - treqs)  # adjust the row with delays
        delays.name = 'delay'
        inData.logger.info \
            ('\t ride: {} \t Best departure time {} in range [{},{}]'.format(ride.name, opt_dep, early, late))

        df_o_t = df_o.join(delays, on='origin')  # add delays to the dataframe
        for dest in dests_list:  # loop of destinations
            # inData.logger.warn('exploring {}-th destination: {}'.format(dests_list.index(dest), dest))
            dest_walks = dest_common_catchment[dest]  # walking times from drop off
            dest_walks.name = 'dest_walk_time'
            df_o_t_d = df_o_t.join(dest_walks, on='destination') # add to data frame
            df_o_t_d['s2s_ttrav'] = inData.skims.ride.loc[orig, dest]  # set travel times
            df_o_t_d['u_s2s'] = df_o_t_d.apply(utility_s2s, axis=1)  # compute utility for each traveller,
            # as a function of walking times, travel times, dely and fare.
            df_o_t_d['exp_u_s2s'] = (df_o_t_d.u_s2s * params.mode_choice_beta).apply(math.exp)  # exponent it
            df_o_t_d['sum_exp'] = df_o_t_d['exp_u_s2s'] + df_o_t_d['sum_exp']  # adjust denominator of MNL
            df_o_t_d['prob_s2s'] = df_o_t_d['exp_u_s2s'] / df_o_t_d['sum_exp'] # calculate probability
            obj_fun = df_o_t_d['prob_s2s'].sum()  # sum of probabilities (can be log sum, but that works fine)
            if trace:  # full report
                trace.append([orig, dest, opt_dep, obj_fun, orig_walks, dest_walks, delays, df_o_t_d['u_s2s'].sum()])
            if obj_fun > best_logsum: # improvement
                inData.logger.info('\t Best solution {:.4f} improved to {:.4f}. '
                                   '\n \t orig:{} dep:{} dest:{}'.format(best_logsum,
                                                                         obj_fun,
                                                                         orig, dep, dest))
                # output
                ret = {'indexes': ride.indexes,
                       'origin': orig,
                       'destination': dest,
                       'treq': dep,
                       'ttrav': inData.skims.ride.loc[orig, dest],
                       'dist': inData.skims.dist.loc[orig, dest],
                       'df': df_o_t_d,
                       "transitizable": True,
                       }
                best_logsum = obj_fun

    # check efficiency
    # ret['df']['efficient'] = ret['df']['u_s2s'] <= ret['df']['u'] # see if utilities of s2s are greater than private
    ret['df']['efficient'] = ret['df']['u_s2s'] <= ret['df']['u_sh']  # see if costs of s2s are lower than d2d
    ret['efficient'] = ret['df']['efficient'].eq(True).all() # for all the travellers

    inData.logger.warn('ride: {} \t Efficiency check: {} \t {}/{} efficient'.format(ride.name,
                                                                                    ret['efficient'],
                                                                                    ret['df']['efficient'].sum(),
                                                                                    ride.degree))
    return pd.Series(ret)


def prepare_transitize(inData, params):
    # computes skim materices, prepares data structures to transitize ExMAS results
    inData.sblts.rides['degree'] = inData.sblts.rides.apply(lambda x: len(x.indexes), axis=1)

    inData.skims = DotMap()  # skim matrices of the network
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

    inData.sblts.rm = ExMAS.utils.make_traveller_ride_matrix(inData)  # data frame with two indices: ride - traveller
    inData.sblts.rm['exp_u_private'] = (inData.sblts.rm.u * params.mode_choice_beta).apply(math.exp)  # utilities
    inData.sblts.rm['exp_u_d2d'] = (inData.sblts.rm.u_sh * params.mode_choice_beta).apply(math.exp)  # exp sum
    inData.sblts.rm['sum_exp'] = inData.sblts.rm['exp_u_d2d'] + inData.sblts.rm[
        'exp_u_private']  # to speed up calculations
    return inData


def list_unmergables(inData):
    df = inData.transitize.second_level_requests

    def unmergables(row):
        # returns list of all the subgroup indiced contained in a ride
        return df[df.indexes_set.apply(lambda x: len(x.intersection(row.indexes_set))) > 0].index.to_list()

    df['unmergables'] = df.apply(unmergables, axis=1)
    df.unmergables = df.apply(lambda m: [x for x in m.unmergables if x != m.name], axis=1)

    unmergables = list()
    for i, row in inData.transitize.second_level_requests.iterrows():
        [unmergables.append(set([row.name, _])) for _ in row.unmergables]
    inData['unmergables'] = unmergables
    inData.logger.warn('{} unmergable pairs from ExMAS'.format(len(inData.unmergables)))
    inData.logger.warn(inData.unmergables)
    return inData



if __name__ == "__main__":
    DEBUG = True
    EXPERIMENT_NAME = 'TEST'

    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, '..'))

    params = ExMAS.utils.get_config('ExMAS/data/configs/default.json')  # load the default

    from ExMAS.utils import inData as inData

    # BIGGER
    if DEBUG:
        # small configuration
        params.nP = 100
        params.pax_delay = 0
        params.simTime = 0.2
        params.speeds.ride = 8
        params.mode_choice_beta = -0.5  # only to estimate utilities
        params.speeds.walk = 1.2
        params.walk_discomfort = 1
        params.delay_value = 1.5
        params.walk_threshold = 500
        params.shared_discount = 0.2
        params.s2s_discount = 0.95
        params.VoT_std = params.VoT / 8
        params.multi_stop_WtS = 1
        params.multistop_discount = 0.98
        params.second_level_shared_discount = ((1 - params.s2s_discount) - (1 - params.multistop_discount)) / (
                    1 - params.s2s_discount)
        params.without_matching = False
    else:
        params.nP = 300  # number of trips
        params.simTime = 0.5  # per simTime hours
        params.mode_choice_beta = -0.5  # only to estimate utilities of pickup points
        params.VoT = 0.0035  # value of time (eur/second)
        params.VoT_std = params.VoT / 8  # variance of Value of Time
        params.speeds.walk = 1.2  # speed of walking (m/s)
        params.walk_discomfort = 1  # walking discomfort factor
        params.delay_value = 1.5  # delay discomfort factor
        params.pax_delay = 0  # extra seconds for each pickup and drop off
        params.walk_threshold = 400  # maximal walking distance (per origin or destination)
        params.price = 1.5  # per kilometer fare
        params.shared_discount = 0.2  # discount for door to door pooling
        params.s2s_discount = 0.66  # discount for stop to stop pooling

        params.speeds.ride = 8

        params.multi_stop_WtS = 1  # willingness to share in multi-stop pooling (now lowe)
        params.multistop_discount = 0.85  # discount for multi-stop
        params.second_level_shared_discount = ((1 - params.s2s_discount) - (1 - params.multistop_discount)) / (
                    1 - params.s2s_discount)
        # how much we reduce multi-stop trip related to stop-to-stop
        params.without_matching = False  # we do not do matching now

    inData.params = params  # store params internally

    inData = ExMAS.utils.load_G(inData, params, stats=True)  # download the graph

    inData = ExMAS.utils.generate_demand(inData, params)  # generate demand of requests

    inData = ExMAS.main(inData, params, plot=False)  # compute door-to-door pooled rides
    inData.logger.warn('ExMAS generated {} rides'.format(inData.sblts.rides.shape[0]))

    inData = prepare_transitize(inData, params)  # prepare data structures to transitize pooled rides

    # main loop s2s
    inData.transitize.first_level_requests = inData.sblts.requests.copy()  # store first level requests
    inData.transitize.first_level_rides = inData.sblts.rides.copy()  # store private and d2d rides

    # main loop
    to_swifter = inData.sblts.rides[inData.sblts.rides.degree != 1]
    from_swifter = to_swifter.swifter.apply(lambda x: transitize(inData, x), axis = 1)
    inData.transitize.second_level_requests = from_swifter

    inData.transitize.rm = pd.concat(
        inData.transitize.second_level_requests[inData.transitize.second_level_requests.transitizable].df.values)
    inData.transitize.rm['pax_id'] = inData.transitize.rm.index.copy()
    # for i, ride in inData.sblts.rides.iterrows():  # loop over rides
    #     inData.transitize.second_level_rides[ride.name] = transitize(inData, ride, plot=False)  # see if it can be tranzitizabled
    #     if inData.transitize.second_level_rides[ride.name]['transitizable']:  # if so, add resulting dataframe for rm matrix
    #         inData.transitize.rm.append(inData.transitize.second_level_rides[ride.name]['df'])

    inData.transitize.second_level_requests = inData.transitize.second_level_requests[
        inData.transitize.second_level_requests['efficient']]
    inData.transitize.second_level_requests = inData.transitize.second_level_requests.apply(pd.to_numeric,
                                                                                            errors='ignore')
    # results of s2s
    inData.sblts.requests.to_csv('requests.csv')
    inData.transitize.second_level_requests.to_csv('second_level_requests.csv')
    inData.transitize.rm.to_csv('transitized_rm.csv')
    inData.transitize.first_level_rides.to_csv('first_level_rides.csv')


    inData.logger.warn('Transitizing: \t{} rides '
                       '\t{} transitizable '
                       '\t{} efficient'.format(inData.transitize.first_level_rides.shape[0],
                                               inData.transitize.second_level_requests[inData.transitize.second_level_requests.transitizable].shape[0],
                                               inData.transitize.second_level_requests.shape[0]))

    inData.transitize.second_level_requests['indexes_set'] = inData.transitize.second_level_requests.apply(lambda x: set(x.indexes),axis=1)

    inData = list_unmergables(inData) # list which second level requests cannot be pooled (have the same travellers)

    inData.transitize.second_level_requests['pax_id'] = inData.transitize.second_level_requests.index.copy()

    # set the indexes of first level rides in the second level rides
    inData.transitize.second_level_requests['low_level_indexes'] = inData.transitize.second_level_requests.apply(lambda x: inData.sblts.rm[inData.sblts.rm.ride == x.name].traveller.to_list(),
                                       axis=1)

    inData.requests = inData.transitize.second_level_requests # set the new requests for ExMAS

    # set parameters for the second level of ExMAS
    params.shared_discount = params.second_level_shared_discount
    params.WtS = params.multi_stop_WtS

    params.reset_ttrav = False  # so that travel times are not divided by avg_speed again
    params.VoT_std = False #
    params.make_assertion_matching = False
    params.make_assertion_pairs = False
    params.process_matching = False
    params.without_matching = True

    inData.transitize.second_level_requests.to_csv('second_level_requests.csv')

    second_level_inData = ExMAS.main(inData, params, plot=False)

    #update indexes looking at travellers in the first level rides
    second_level_inData.sblts.rides['indexes'] = second_level_inData.sblts.rides.apply(
        lambda x: sum(inData.transitize.second_level_requests.loc[x.indexes].low_level_indexes.to_list(), []), axis=1)

    inData.sblts.rides = pd.concat([inData.transitize.first_level_rides, inData.sblts.rides])
    inData.sblts.requests = inData.transitize.first_level_requests
    inData.sblts.rides.to_csv('both_level_rides.csv')


    second_level_inData.sblts.rides.to_csv('second_level_rides.csv')

    inData = matching(inData, params)
    inData.sblts.schedule.to_csv('solution.csv')
