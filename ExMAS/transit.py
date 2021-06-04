
import os
import numpy as np
import osmnx as ox
import networkx as nx
from matplotlib.collections import LineCollection
import pandas as pd
import math
from dotmap import DotMap
import ExMAS.utils


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


def plot_ride(inData, ride, sizes=3):
    G = inData.G
    routes = list()
    fig, ax = ox.plot_graph(G, figsize=(15, 15), node_size=sizes, node_color='black', edge_color='grey',
                            bgcolor='white', edge_linewidth=0.3,
                            show=False, close=False)
    for i, origin in enumerate(ride.origins):
        ax.scatter(G.nodes[origin]['x'], G.nodes[origin]['y'], s=10, c='red', marker='o')
        ax.annotate('{}'.format(ride.indexes_orig[i], ride.dep_deltas[i]),
                    (G.nodes[origin]['x'] + 0.0001, G.nodes[origin]['y'] + 0.0001)
                    , bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'), fontsize=8)

        routes.append(nx.shortest_path(G, ride.origins[i], ride.orig, weight='length'))
        routes.append(nx.shortest_path(G, ride.dest, ride.destinations[i], weight='length'))
    for i, origin in enumerate(ride.destinations):
        ax.scatter(G.nodes[origin]['x'], G.nodes[origin]['y'], s=10, c='blue', marker='o')
        ax.annotate('{}'.format(ride.indexes_dest[i]), (G.nodes[origin]['x'] + 0.0001, G.nodes[origin]['y'] + 0.0001),
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), fontsize=8)

    transit_route = nx.shortest_path(G, ride.orig, ride.dest, weight='length')
    add_route(G, ax, transit_route, color='red', lw=4, alpha=1)

    for route in routes:
        add_route(G, ax, route, color='green', lw=4, alpha=1)
    ax.scatter(G.nodes[ride.orig]['x'], G.nodes[ride.orig]['y'], s=500, c='red', marker='o')
    ax.scatter(G.nodes[ride.dest]['x'], G.nodes[ride.dest]['y'], s=500, c='red', marker='o')


def transitize(inData, ride, plot=False, trace = False):

    def utility_s2s(ride):
        # utility of shared trip i for all the travellers
        return (params.price * (1 - params.s2s_discount) * ride.dist / 1000 +
                ride.VoT * params.WtS * (ride.s2s_ttrav + params.delay_value * ride.delay) +
                ride.VoT * params.walk_discomfort * (ride.orig_walk_time + ride.dest_walk_time))

    reqs = inData.sblts.requests.loc[ride.indexes]
    rm = inData.sblts.rm.loc[ride.name, :][['ride' ,'exp_u_private', 'exp_u_d2d', 'sum_exp', 'u', 'u_sh']].join(
        reqs[['origin', 'destination', 'dist', 'VoT']])
    assert rm.shape[0] == reqs.shape[0]
    params = inData.params

    ret = {'origin':      None,
           'destination':      None,
           'treq':       None,
           'ttrav':     None,
           'df':        None,
           'efficient': False,
           'transitizable': False,
           }

    # only pooled rides
    if ride.degree == 1:
        return ret

    inData.logger.warn('Transitization of pooled ride: {} of degree: {}'.format(ride.name, ride.degree))

    # see if there is a common pickup point
    orig_catchments = inData.skims.walk.loc[ride.origins]
    orig_common_catchment = orig_catchments.loc[:, orig_catchments.sum() < np.inf]
    if orig_common_catchment.shape[1] == 0:
        inData.logger.warn('no common origin pick-up point')
        return ret

    # see if there is a common dropoff point
    dest_catchments = inData.skims.walk.loc[ride.destinations]
    # dest_catchments = dest_catchments.mask(dest_catchments> params.walk_threshold, np.inf)
    dest_common_catchment = dest_catchments.loc[:, dest_catchments.sum() < np.inf]
    if dest_common_catchment.shape[1] == 0:
        inData.logger.warn('no common destination pick-up point')
        return ret

    # explore
    origs_list = orig_common_catchment.columns.to_list()
    dests_list = dest_common_catchment.columns.to_list()
    treqs = reqs.set_index('origin').treq
    early = treqs.min()
    late = treqs.max()

    inData.logger.warn('Transitizing ride: {} \t Common orig points:{},  dest points: {}'.format(ride.name,
                                                                                                 len(origs_list),
                                                                                                 len(dests_list)))

    best_logsum = -np.inf
    ret = dict()
    if trace:
        trace = list()
    for orig in origs_list:
        inData.logger.warn('\t ride: {} \t exploring {}-th origin: {}'.format(ride.name,
                                                                                        origs_list.index(orig),
                                                                                        orig))
        orig_walks = orig_common_catchment[orig]
        orig_walks.name = 'orig_walk_time'
        df_o = rm.join(orig_walks, on='origin')
        min_delay = np.inf
        opt_dep = early
        for dep in range(early, late + 600):
            delays = abs((dep - orig_walks - treqs )**2)
            if delays.sum() < min_delay:
                min_delay = delays.sum()
                opt_dep = dep
        delays = abs(opt_dep - orig_walks - treqs)
        delays.name = 'delay'
        inData.logger.warn \
            ('\t ride: {} \t Best departure time {} in range [{},{}]'.format(ride.name, opt_dep ,early ,dep))

        df_o_t = df_o.join(delays, on='origin')
        for dest in dests_list:
            # inData.logger.warn('exploring {}-th destination: {}'.format(dests_list.index(dest), dest))
            dest_walks = dest_common_catchment[dest]
            dest_walks.name = 'dest_walk_time'
            df_o_t_d = df_o_t.join(dest_walks, on='destination')
            df_o_t_d['s2s_ttrav'] = inData.skims.ride.loc[orig, dest]
            df_o_t_d['u_s2s'] = df_o_t_d.apply(utility_s2s, axis=1)
            df_o_t_d['exp_u_s2s'] = (df_o_t_d.u_s2s * params.mode_choice_beta).apply(math.exp)
            df_o_t_d['sum_exp'] = df_o_t_d['exp_u_s2s'] + df_o_t_d['sum_exp']
            df_o_t_d['prob_s2s'] = df_o_t_d['exp_u_s2s'] / df_o_t_d['sum_exp']
            logsum = df_o_t_d['prob_s2s'].sum()
            if trace:
                trace.append([orig, dest, dep, logsum, orig_walks, dest_walks, delays, df_o_t_d['u_s2s'].sum()])
            if logsum > best_logsum:
                inData.logger.warn('\t Best solution {:.4f} improved to {:.4f}. '
                                   '\n \t orig:{} dep:{} dest:{}'.format(best_logsum,
                                                                                   logsum,
                                                                                   orig ,dep, dest))
                ret = {"transitizable": True,
                       'origin': orig,
                       'destination': dest,
                       'treq': dep,
                       'ttrav': df_o_t_d['s2s_ttrav'].max(),
                       'dist': df_o_t_d['s2s_ttrav'].max()/params.avg_speed,
                       'df': df_o_t_d,
                       }
                best_logsum = logsum

    ret['df']['efficient'] = ret['df']['u_s2s'] <= ret['df']['u']
    ret['efficient'] = ret['df']['efficient'].eq(True).all()

    inData.logger.warn('ride: {} \t Efficiency check: {} \t {}/{} efficient'.format(ride.name,
                                                                                    ret['efficient'],
                                                                                    ret['df']['efficient'].sum(),
                                                                                    ride.degree))


    return ret


def prepare_transitize(inData):
    # computes skim materices, prepares data structures to transitize ExMAS results
    inData.sblts.rides['degree'] = inData.sblts.rides.apply(lambda x: len(x.indexes), axis=1)

    inData.skims = DotMap()
    inData.skims.dist = inData.skim.copy()
    inData.skims.ride = inData.skims.dist.divide(params.speeds.ride).astype(int).T
    inData.skims.walk = inData.skims.dist.divide(params.speeds.walk).astype(int).T
    inData.skims.walk = inData.skims.walk.mask(inData.skims.walk > params.walk_threshold, np.inf)

    inData.sblts.rides['origins'] = inData.sblts.rides.apply(
        lambda ride: list(inData.sblts.requests.loc[ride.indexes_orig].origin), axis=1)
    inData.sblts.rides['destinations'] = inData.sblts.rides.apply(
        lambda ride: list(inData.sblts.requests.loc[ride.indexes_dest].destination), axis=1)
    inData.sblts.rides['deps'] = inData.sblts.rides.apply(
        lambda ride: list(inData.sblts.requests.loc[ride.indexes_orig].treq), axis=1)
    inData.sblts.rides['dep_deltas'] = inData.sblts.rides.apply(lambda ride: list(
        inData.sblts.requests.loc[ride.indexes_orig].treq - inData.sblts.requests.loc[ride.indexes_orig].treq.mean()),
                                                                axis=1)

    inData.sblts.rm = ExMAS.utils.make_traveller_ride_matrix(inData)
    inData.sblts.rm['exp_u_private'] = (inData.sblts.rm.u * params.mode_choice_beta).apply(math.exp)
    inData.sblts.rm['exp_u_d2d'] = (inData.sblts.rm.u_sh * params.mode_choice_beta).apply(math.exp)
    inData.sblts.rm['sum_exp'] = inData.sblts.rm['exp_u_d2d'] + inData.sblts.rm['exp_u_private']
    return inData



if __name__ == "__main__":

    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, '..'))

    params = ExMAS.utils.get_config('ExMAS/data/configs/default.json')  # load the default

    from ExMAS.utils import inData as inData


    params.nP = 100
    params.simTime = 0.2
    params.speeds.ride = 8
    params.mode_choice_beta = -0.5 # only to estimate utilities
    params.speeds.walk = 1.2
    params.walk_discomfort = 1
    params.delay_value = 1.5
    params.walk_threshold = 600
    params.shared_discount = 0.33
    params.s2s_discount = 0.66
    params.multi_stop_WtS = 1
    params.multistop_discount = 0.8
    params.second_level_shared_discount = ((1-params.s2s_discount) - (1-params.multistop_discount)) / (1-params.s2s_discount)
    params.without_matching = True


    inData.params = params

    inData = ExMAS.utils.load_G(inData, params, stats=True)  # download the graph

    inData = ExMAS.utils.generate_demand(inData, params)

    inData = ExMAS.main(inData, params, plot=False)
    inData = prepare_transitize(inData)

    # main loop s2s
    transits = dict()
    transit_rm = list()
    for i, ride in inData.sblts.rides.iterrows():
        transits[ride.name] = transitize(inData, ride, plot=False)
        if transits[ride.name]['transitizable']:
            transit_rm.append(transits[ride.name]['df'])


    transit_rm = pd.concat(transit_rm)
    transits = pd.DataFrame(transits).T

    # results of s2s

    inData.sblts.requests.to_csv('requests.csv')
    transits.to_csv('transits.csv')
    transit_rm.to_csv('transit_rm.csv')

    transit_rm['pax_id'] = transit_rm.index.copy()

    df = transits[transits['efficient']]

    df['indexes'] = df.apply(lambda x: transit_rm[transit_rm.ride == x.name].pax_id.to_list(), axis=1)
    df['indexes_set'] = df.apply(lambda x: set(transit_rm[transit_rm.ride== x.name].pax_id.to_list()), axis =1)


    def unmergables(row):
        # returns list of all the subgroup indiced contained in a ride
        return df[df.indexes_set.apply(lambda x: len(x.intersection(row.indexes_set))) > 0].index.to_list()

    df['unmergables'] = df.apply(unmergables, axis=1)
    df.unmergables = df.apply(lambda m: [x for x in m.unmergables if x != m.name], axis=1)

    unmergables = list()
    for i, row in df.iterrows():
        [unmergables.append((row.name, _)) for _ in row.unmergables]
    inData.unmergables = unmergables

    params.shared_discount = params.second_level_shared_discount
    params.WtS = params.multi_stop_WtS

    inData.requests = df

    inData2 = ExMAS.main(inData, params, plot=False)
