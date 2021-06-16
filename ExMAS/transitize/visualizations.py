from matplotlib.collections import LineCollection
from ExMAS.utils import plot_map_rides, read_csv_lists
import osmnx as ox
import os
import pandas as pd
import networkx as nx
import seaborn as sns

from matplotlib.collections import LineCollection


def plot_d2d(inData, ride_index, light=True, m_size=30, lw=3, fontsize = 10, figsize = (25,25)):

    def make_schedule(t, r):
        # creates a sequence of stops for a ride
        columns = ['node', 'times', 'req_id', 'od']
        degree = 2 * len(t.indexes_orig)
        df = pd.DataFrame(None, index=range(degree), columns=columns)
        x = t.indexes_orig
        s = [r.loc[i].origin for i in x] + [r.loc[i].destination for i in x]
        df.node = pd.Series(s)
        df.req_id = x + t.indexes_dest
        df.times = t.times
        df.od = pd.Series(['o'] * len(t.indexes_orig) + ['d'] * len(t.indexes_orig))
        return df


    def add_route(ax, route, color='grey', lw=2, alpha=0.5):
        # plots route on the graph (already plotted on ax)
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

    s = inData.transitize.rides  # input
    r = inData.transitize.requests  # input
    G = inData.G  # input

    ts = [make_schedule(s.iloc[ride_index],r)]




    G = inData.G
    fig, ax = ox.plot_graph(G, figsize=figsize, node_size=0, edge_linewidth=0.3,
                            show=False, close=False,
                            edge_color='grey',  bgcolor='white')

    #colors = {1: 'navy', 2: 'teal', 3: 'maroon', 4: 'black', 5: 'green', 6:'teal'}
    colors = sns.color_palette("Set2",6)


    for t in ts:

        orig_points_lats, orig_points_lons, dest_points_lats, dest_points_lons = [], [], [], []
        deg = t.req_id.nunique()
        count = 0
        for i in t.req_id.unique():
            count += 1
            r = t[t.req_id == i]

            o = r[r.od == 'o'].iloc[0].node
            d = r[r.od == 'd'].iloc[0].node

            if not light:
                ax.annotate('o' + str(i), (G.nodes[o]['x'] * 1.0001, G.nodes[o]['y'] * 1.00001), fontsize = fontsize,
                    bbox = dict(facecolor='white', alpha=0.7, edgecolor='none'))
                ax.annotate('d' + str(i), (G.nodes[d]['x'] * 1.0001, G.nodes[d]['y'] * 1.00001), fontsize = fontsize,
                    bbox = dict(facecolor='white', alpha=0.7, edgecolor='none'))
            route = nx.shortest_path(G, o, d, weight='length')
            add_route(ax, route, color='black', lw=lw / 2, alpha=0.3)
            ax.scatter(G.nodes[o]['x'], G.nodes[o]['y'], s=m_size, c=[colors[deg]], marker='o')
            ax.scatter(G.nodes[d]['x'], G.nodes[d]['y'], s=m_size, c=[colors[deg]], marker='>')

        routes = list()  # ride segments
        o = t.node.values[0]
        for d in t.node.values[1:]:
            routes.append(nx.shortest_path(G, o, d, weight='length'))
            o = d
        for route in routes:
            add_route(ax, route, color=[colors[deg]], lw=lw, alpha=0.7)
    plt.tight_layout()
    plt.savefig('map.png', dpi = 300)


def plot_s2s(inData, ride_id, sizes=0, fig=None, ax=None, label_offset=0.0005):
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
        ax.scatter(G.nodes[origin]['x'], G.nodes[origin]['y'], s=10, c='red', marker='o')
        ax.annotate('{}'.format(ride.indexes[i]),
                    (G.nodes[origin]['x'] + label_offset, G.nodes[origin]['y'] + label_offset)
                    , bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'), fontsize=8)

        routes.append(nx.shortest_path(G, ride.origins[i], ride.origin, weight='length'))
        routes.append(nx.shortest_path(G, ride.destination, ride.destinations[i], weight='length'))
    for i, origin in enumerate(ride.destinations):
        ax.scatter(G.nodes[origin]['x'], G.nodes[origin]['y'], s=10, c='blue', marker='o')
        ax.annotate('{}'.format(ride.indexes[i]),
                    (G.nodes[origin]['x'] + label_offset, G.nodes[origin]['y'] + label_offset),
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), fontsize=8)

    transit_route = nx.shortest_path(G, ride.origin, ride.destination, weight='length')
    add_route(G, ax, transit_route, color='red', lw=4, alpha=0.5)

    for route in routes:
        add_route(G, ax, route, color='green', lw=4, alpha=0.5)
    ax.scatter(G.nodes[ride.origin]['x'], G.nodes[ride.origin]['y'], s=50, c='red', marker='o')
    ax.scatter(G.nodes[ride.destination]['x'], G.nodes[ride.destination]['y'], s=50, c='red', marker='o')
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
    for i in [0, 1, 2, 3]:
        ret[i] = inData.transitize.rides[inData.transitize.rides['solution_{}'.format(i)] == 1][KPIs].sum()
        ret[i] = pd.concat([ret[i],
                            inData.transitize.rides[inData.transitize.rides['solution_{}'.format(i)] == 1].groupby(
                                'kind').size()])
        inds = inData.transitize.rides[inData.transitize.rides['solution_{}'.format(i)] == 1].indexes
        ret[i]['test'] = len(sum(inds, [])) == inData.transitize.requests.shape[0]

    inData.transitize.report = pd.DataFrame(ret)
    return inData


def results_pipeline(PATH='ams'):
    inData = prep_results(PATH)



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


def plot_ride(inData, ride, sizes=0, fig=None, ax=None, label_offset=0.0005):
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
        ax.annotate('{}'.format(ride.indexes[i]),
                    (G.nodes[origin]['x'] + label_offset, G.nodes[origin]['y'] + label_offset),
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), fontsize=8)

    transit_route = nx.shortest_path(G, ride.origin, ride.destination, weight='length')
    add_route(G, ax, transit_route, color='red', lw=4, alpha=0.5)

    for route in routes:
        add_route(G, ax, route, color='green', lw=4, alpha=0.5)
    ax.scatter(G.nodes[ride.origin]['x'], G.nodes[ride.origin]['y'], s=50, c='red', marker='o')
    ax.scatter(G.nodes[ride.destination]['x'], G.nodes[ride.destination]['y'], s=50, c='red', marker='o')
    return fig, ax

if __name__ == "__main__":

    PATH = "../../ams"
    inData = prep_results(PATH)
    inData = make_report(inData)
    inData.transitize.report