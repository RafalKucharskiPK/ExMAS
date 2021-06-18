from matplotlib.collections import LineCollection
from ExMAS.utils import plot_map_rides, read_csv_lists
import osmnx as ox
import os
import pandas as pd
import networkx as nx
import seaborn as sns
import json

from matplotlib.collections import LineCollection


def plot_ms(inData, ride_index, bbox=0.1, level = 0, title = None):
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

    if bbox is not None:
        ride['origins'] = requests.loc[ride.indexes].origin.values
        ride['destinations'] = requests.loc[ride.indexes].destination.values
        X = pd.Series([inData.nodes.loc[_].x for _ in ride.origins.tolist() + ride.destinations.tolist()])
        Y = pd.Series([inData.nodes.loc[_].y for _ in ride.origins.tolist() + ride.destinations.tolist()])
        deltaX = bbox * (X.max() - X.min())
        deltaY = bbox * (Y.max() - Y.min())
        bbox = (Y.max() + deltaY, Y.min() - deltaY, X.max() + deltaX, X.min() - deltaX)

    private_rides = rides[rides.kind == 'p'][rides['index'].isin(ride.indexes)]

    s2s_rides = rides.loc[ride.high_level_indexes]

    d2d_rides = rides.loc[s2s_rides.d2d_reference.values]

    fig, ax = ox.plot_graph(G, figsize=(20, 20), node_size=0, node_color='black', edge_color='grey',
                            bgcolor='white', edge_linewidth=0.2, bbox=bbox,
                            show=False, close=False)

    if level == 0:
        for i, d2d_ride in enumerate(d2d_rides.index.values):
            fig, ax = plot_d2d(inData, int(d2d_ride), light=False, fontsize=10,
                               color=colors[i], fig=fig, ax=ax, lw=5, plot_shared=False)

    if level>=2:
         for i, s2s_ride in enumerate(s2s_rides.index.values):
           plot_s2s(inData, int(s2s_ride), fig=fig, ax = ax, color = colors[i], light = False)

    if level ==1:
        for i, d2d_ride in enumerate(d2d_rides.index.values):
            fig, ax = plot_d2d(inData, int(d2d_ride), light=False, fontsize=10,
                               color=colors[i], fig=fig, ax=ax, lw=3, plot_shared=True)
    if level == 3:

        t = make_schedule(ride, rides)
        routes = list()  # ride segments
        o = t.node.values[0]
        for d in t.node.values[1:]:
            routes.append(nx.shortest_path(G, o, d, weight='length'))
            o = d
        for route in routes:
            add_route(G, ax, route, color=['black'], lw=5, alpha=1)
    if title:
        fig.suptitle(title, size = 20, fontweight='bold')
        fig.tight_layout()


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
            add_route(G, ax, route, color=[color], lw=lw, alpha=0.8)
    return fig, ax


def plot_s2s(inData, ride_id, sizes=0, fig=None, ax=None, light=True, label_offset=0, fontsize=8, color='blue'):
    G = inData.G
    ride = inData.transitize.rides.loc[ride_id]
    requests = inData.transitize.requests
    ride['origins'] = requests.loc[ride.indexes].origin.values
    ride['destinations'] = requests.loc[ride.indexes].destination.values
    routes = list()

    if fig is None:
        fig, ax = ox.plot_graph(G, figsize=(25, 25), node_size=sizes, node_color='black', edge_color='grey',
                                bgcolor='white', edge_linewidth=0.3,
                                show=False, close=False)
    for i, origin in enumerate(ride.origins):
        ax.scatter(G.nodes[origin]['x'], G.nodes[origin]['y'], s=40, c=[color], marker='o')
        if not light:
            ax.annotate('o{}'.format(ride.indexes[i]),
                        (G.nodes[origin]['x'] + label_offset, G.nodes[origin]['y'] + label_offset)
                        , bbox=dict(facecolor='white', alpha=0, edgecolor='none'), fontsize=fontsize)

        routes.append(nx.shortest_path(G, ride.origins[i], ride.origin, weight='length'))
        routes.append(nx.shortest_path(G, ride.destination, ride.destinations[i], weight='length'))
    for i, origin in enumerate(ride.destinations):
        ax.scatter(G.nodes[origin]['x'], G.nodes[origin]['y'], s=40, c=[color], marker='o')
        if not light:
            ax.annotate('d{}'.format(ride.indexes[i]),
                        (G.nodes[origin]['x'] + label_offset, G.nodes[origin]['y'] + label_offset),
                        bbox=dict(facecolor='white', alpha=0, edgecolor='none'), fontsize=fontsize)

    transit_route = nx.shortest_path(G, ride.origin, ride.destination, weight='length')
    add_route(G, ax, transit_route, color=[color], lw=3, alpha=1)

    for route in routes:
        add_route(G, ax, route, color=['black'], lw=1, alpha=1, linestyle='dashed')
    ax.scatter(G.nodes[ride.origin]['x'], G.nodes[ride.origin]['y'], s=150, c=[color], marker='o')
    ax.scatter(G.nodes[ride.destination]['x'], G.nodes[ride.destination]['y'], s=150, c=[color], marker='o')
    return fig, ax


def prep_results(PATH, inData=None):
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


def results_pipeline(PATH='ams'):
    inData = prep_results(PATH)


def add_route(G, ax, route, color='grey', lw=2, alpha=0.5, linestyle='solid'):
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
    lc = LineCollection(lines, colors=color[0], linewidths=lw, alpha=alpha, zorder=3, linestyle=linestyle)
    ax.add_collection(lc)


# def plot_ride(inData, ride, sizes=0, fig=None, ax=None, label_offset=0.0005):
#     G = inData.G
#     routes = list()
#     if fig is None:
#         fig, ax = ox.plot_graph(G, figsize=(25, 25), node_size=sizes, node_color='black', edge_color='grey',
#                                 bgcolor='white', edge_linewidth=0.3,
#                                 show=False, close=False)
#     for i, origin in enumerate(ride.origins):
#         ax.scatter(G.nodes[origin]['x'], G.nodes[origin]['y'], s=10, c='red', marker='o')
#         ax.annotate('{}'.format(ride.indexes[i]),
#                     (G.nodes[origin]['x'] + label_offset, G.nodes[origin]['y'] + label_offset)
#                     , bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'), fontsize=8)
#
#         routes.append(nx.shortest_path(G, ride.origins[i], ride.origin, weight='length'))
#         routes.append(nx.shortest_path(G, ride.destination, ride.destinations[i], weight='length'))
#     for i, origin in enumerate(ride.destinations):
#         ax.scatter(G.nodes[origin]['x'], G.nodes[origin]['y'], s=10, c='blue', marker='o')
#         ax.annotate('{}'.format(ride.indexes[i]),
#                     (G.nodes[origin]['x'] + label_offset, G.nodes[origin]['y'] + label_offset),
#                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), fontsize=8)
#
#     transit_route = nx.shortest_path(G, ride.origin, ride.destination, weight='length')
#     add_route(G, ax, transit_route, color='red', lw=4, alpha=0.5)
#
#     for route in routes:
#         add_route(G, ax, route, color='green', lw=4, alpha=0.5)
#     ax.scatter(G.nodes[ride.origin]['x'], G.nodes[ride.origin]['y'], s=50, c='red', marker='o')
#     ax.scatter(G.nodes[ride.destination]['x'], G.nodes[ride.destination]['y'], s=50, c='red', marker='o')
#     return fig, ax

if __name__ == "__main__":
    PATH = "../../ams"
    inData = prep_results(PATH)
    inData = make_report(inData)
    inData.transitize.report
