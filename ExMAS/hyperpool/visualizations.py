"""
# ExMAS - TRANSITIZE
> Exact Matching of Attractive Shared rides (ExMAS) for system-wide strategic evaluations
> Module to pool requests to stop-to-stop and multi-stop rides, aka TRANSITIZE
---

Visualize the results



----
Rafał Kucharski, TU Delft, GMUM UJ  2021 rafal.kucharski@uj.edu.pl
"""

import osmnx as ox
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from .analysis import prep_results, make_report
import json

from matplotlib.collections import LineCollection


def plot_multi_ms(inData, n, figname=None, savefig=False):
    ax = None
    fig = None
    ride_indexes = inData.transitize.rm[
        (inData.transitize.rm.solution_3 == 1) & (inData.transitize.rm.kind == 'ms')].ride.unique()
    ride_indexes = np.random.choice(ride_indexes, n)
    colors = sns.color_palette("Set2", n)
    for i, ms in enumerate(ride_indexes):
        fig, ax = plot_ms(inData, ms, level=3, title=None, light=True, ax=ax, fig=fig, bbox=None, color=colors[i])
    if savefig:
        plt.savefig('maps/map_{}_{}.png'.format(n, np.random.randint(0, 1000)) if figname is None else figname, dpi=300)
    return fig




def plot_ms(inData, ride_index, fig=None, ax=None, bbox=0.1, level=0, title=None,
            fontsize=10, figsize=(20, 20), light=False, color='black'):
    colors = sns.color_palette("Set2", 6)
    rides = inData.transitize.rides
    requests = inData.transitize.requests
    G = inData.G

    ride = inData.transitize.rides.loc[ride_index]
    try:
        ride['high_level_indexes'] = json.loads(ride['high_level_indexes'])
    except:
        pass
    ride.indexes_orig = json.loads(ride.indexes_orig)
    ride.indexes_dest = json.loads(ride.indexes_dest)
    ride['origins'] = requests.loc[ride.indexes].origin.values
    ride['destinations'] = requests.loc[ride.indexes].destination.values

    private_rides = rides[rides.kind == 'p'][rides['index'].isin(ride.indexes)]

    s2s_rides = rides.loc[ride.high_level_indexes]

    d2d_rides = rides.loc[s2s_rides.d2d_reference.values]

    if ax is None:
        if bbox is not None:
            ride['origins'] = requests.loc[ride.indexes].origin.values
            ride['destinations'] = requests.loc[ride.indexes].destination.values
            X = pd.Series([inData.nodes.loc[_].x for _ in ride.origins.tolist() + ride.destinations.tolist()])
            Y = pd.Series([inData.nodes.loc[_].y for _ in ride.origins.tolist() + ride.destinations.tolist()])
            deltaX = bbox * (X.max() - X.min())
            deltaY = bbox * (Y.max() - Y.min())
            bbox = (Y.max() + deltaY, Y.min() - deltaY, X.max() + deltaX, X.min() - deltaX)
        fig, ax = ox.plot_graph(G, figsize=figsize, node_size=0, node_color='black', edge_color='grey',
                                bgcolor='white', edge_linewidth=0.1, bbox=bbox,
                                show=False, close=False)

    if level == 0:
        for i, d2d_ride in enumerate(d2d_rides.index.values):
            fig, ax = plot_d2d(inData, int(d2d_ride), light=False, fontsize=fontsize,
                               color='black', fig=fig, ax=ax, lw=5, plot_shared=False)

    if level == 2:
        for i, s2s_ride in enumerate(s2s_rides.index.values):
            plot_s2s(inData, int(s2s_ride), fig=fig, ax=ax, color=colors[i], light=False, fontsize=fontsize)

    if level == 1:
        for i, d2d_ride in enumerate(d2d_rides.index.values):
            fig, ax = plot_d2d(inData, int(d2d_ride), light=False, fontsize=fontsize,
                               color=colors[i], fig=fig, ax=ax, lw=3, plot_shared=True)
    if level == 3:
        for i, s2s_ride in enumerate(s2s_rides.index.values):
            plot_s2s(inData, int(s2s_ride), fig=fig, ax=ax, color=color, lw=0, light=False, fontsize=fontsize,
                     superlight=light)

        t = make_schedule(ride, rides)
        routes = list()  # ride segments
        o = t.node.values[0]
        for d in t.node.values[1:]:
            routes.append(nx.shortest_path(G, o, d, weight='length'))
            o = d
        for route in routes:
            add_route(G, ax, route, color=[color], lw=3, alpha=1)
    if title:
        fig.suptitle(title, size=20, fontweight='bold')
        fig.tight_layout()
    return fig, ax


def make_schedule(t, r):
    # creates a sequence of stops for a ride
    columns = ['node', 'times', 'req_id', 'od']
    degree = 2 * len(t.indexes_orig)
    df = pd.DataFrame(None, index=range(degree), columns=columns)
    x = t.indexes_orig
    s = [r.loc[i].origin for i in x] + [r.loc[i].destination for i in t.indexes_dest]
    df.node = pd.Series(s)
    df.req_id = x + t.indexes_dest
    # df.times = t.times
    df.od = pd.Series(['o'] * len(t.indexes_orig) + ['d'] * len(t.indexes_orig))
    return df


def plot_d2d(inData, ride_index, fig=None, ax=None,
             light=True, m_size=30, lw=3, fontsize=10, figsize=(25, 25), color='black',
             label_offset=0.0001, plot_shared=True):
    """
    Plots on a map (inData.G) a shared Door-to-door ride. To be called after transitize is computed.
    :param inData: needs to have:
                    .G - graph from osmnx
                    .transitize.rides
                    .transitize.requests
    :param ride_index: index in inData.transitize.rides to plot
    :param fig: optional - to pass if matplotlib figure is already created
    :param ax: optional - to pass if matplotlib figure is axis created
    :param light: scaling of line weights and node sizes and flag to make annotations
    :param m_size: size of node
    :param lw: lightweight
    :param fontsize:
    :param figsize:
    :param color:
    :param label_offset:
    :param plot_shared: add base d2d ride to the plot
    :return:
    """
    s = inData.transitize.rides  # input
    r = inData.transitize.requests  # input
    G = inData.G  # input

    ride = s.iloc[ride_index]

    ride.indexes_orig = json.loads(ride.indexes_orig)
    ride.indexes_dest = json.loads(ride.indexes_dest)

    t = make_schedule(ride, r)

    if fig is None:
        fig, ax = ox.plot_graph(G, figsize=figsize, node_size=0, edge_linewidth=0.2,
                                show=False, close=False,
                                edge_color='grey', bgcolor='white')

    deg = t.req_id.nunique()
    count = 0
    for i in t.req_id.unique():
        count += 1
        r = t[t.req_id == i]

        o = r[r.od == 'o'].iloc[0].node
        d = r[r.od == 'd'].iloc[0].node

        if not light:
            ax.annotate('o' + str(i), (G.nodes[o]['x'] + label_offset, G.nodes[o]['y'] + label_offset),
                        fontsize=fontsize,
                        bbox=dict(facecolor='white', alpha=0, edgecolor='none'))
            ax.annotate('d' + str(i), (G.nodes[d]['x'] + label_offset, G.nodes[d]['y'] + label_offset),
                        fontsize=fontsize,
                        bbox=dict(facecolor='white', alpha=0, edgecolor='none'))
        route = nx.shortest_path(G, o, d, weight='length')
        add_route(G, ax, route, color=[color], lw=lw / 2, alpha=0.3)
        ax.scatter(G.nodes[o]['x'], G.nodes[o]['y'], s=m_size, c=[color], marker='o')
        ax.scatter(G.nodes[d]['x'], G.nodes[d]['y'], s=m_size, c=[color], marker='>')

    if plot_shared:
        routes = list()  # ride segments
        o = t.node.values[0]
        for d in t.node.values[1:]:
            routes.append(nx.shortest_path(G, o, d, weight='length'))
            o = d
        for route in routes:
            add_route(G, ax, route, color=[color], lw=lw, alpha=0.5)
    return fig, ax


def plot_s2s(inData, ride_id, sizes=0, fig=None, ax=None, lw=None, light=True,
             label_offset=0, fontsize=8, color='blue', superlight=False):
    '''
    plots stop-to-stop ride including stop point (common for all origins) and walking to it
    :param inData:
    :param ride_id:
    :param sizes:
    :param fig:
    :param ax:
    :param light:
    :param label_offset:
    :param fontsize:
    :param color:
    :return: fig, ax with plotted graph and ride
    '''
    G = inData.G
    ride = inData.transitize.rides.loc[ride_id]
    requests = inData.transitize.requests
    ride['origins'] = requests.loc[ride.indexes].origin.values
    ride['destinations'] = requests.loc[ride.indexes].destination.values
    routes = list()
    if superlight:
        light = True

    if fig is None:
        fig, ax = ox.plot_graph(G, figsize=(25, 25), node_size=sizes, node_color='black', edge_color='grey',
                                bgcolor='white', edge_linewidth=0.3,
                                show=False, close=False)

    for i, origin in enumerate(ride.origins):

        ax.scatter(G.nodes[origin]['x'], G.nodes[origin]['y'], s=20, c=[color], marker='o')
        if not light:
            ax.annotate('o{}'.format(ride.indexes[i]),
                        (G.nodes[origin]['x'] + label_offset, G.nodes[origin]['y'] + label_offset)
                        , bbox=dict(facecolor='white', alpha=0, edgecolor='none'), fontsize=fontsize)

        routes.append(nx.shortest_path(G, ride.origins[i], ride.origin, weight='length'))
        routes.append(nx.shortest_path(G, ride.destination, ride.destinations[i], weight='length'))
    for i, origin in enumerate(ride.destinations):
        ax.scatter(G.nodes[origin]['x'], G.nodes[origin]['y'], s=20, c=[color], marker='o')
        if not light:
            ax.annotate('d{}'.format(ride.indexes[i]),
                        (G.nodes[origin]['x'] + label_offset, G.nodes[origin]['y'] + label_offset),
                        bbox=dict(facecolor='white', alpha=0, edgecolor='none'), fontsize=fontsize)

    transit_route = nx.shortest_path(G, ride.origin, ride.destination, weight='length')
    add_route(G, ax, transit_route, color=[color], lw=3 if lw is None else lw, alpha=1)

    for route in routes:
        add_route(G, ax, route, color=[color if superlight else 'black'], lw=1, alpha=1,
                  linestyle='solid' if superlight else 'dashed')
    ax.scatter(G.nodes[ride.origin]['x'], G.nodes[ride.origin]['y'], s=150, c=[color], marker='o')
    ax.scatter(G.nodes[ride.destination]['x'], G.nodes[ride.destination]['y'], s=150, c=[color], marker='o')
    return fig, ax


def add_route(G, ax, route, color='grey', lw=2, alpha=0.5, linestyle='solid'):
    # plots route on the graph already plotted on ax - reusable
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
    lc = LineCollection(lines, colors=color[0], linewidths=lw, alpha=alpha, zorder=3, linestyle=linestyle)
    ax.add_collection(lc)


if __name__ == "__main__":
    PATH = "../../ams"
    inData = prep_results(PATH)
    inData = make_report(inData)
    inData.transitize.report
