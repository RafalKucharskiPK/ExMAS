import pandas as pd
import random
import ExMAS
import ExMAS.utils
from ExMAS.utils import inData as mData
import json
from dotmap import DotMap
import matplotlib.pyplot as plt
from ExMAS.utils import get_config
from ExMAS.main import matching
import seaborn as sns
import math
import osmnx as ox
import time

def infect(inData, day, params):
    """
    for a given infected travellers and their shared rides schedule return requests with newly_infected travellers
    :param schedule: shared rides
    :param requests: travellers and their trips requests
    :param params: parameters
    :return: requests with column newly_infected
    """
    got_infected = dict()
    infected_by = dict()
    for i, ride in inData.sblts.schedule.iterrows():  # iterate over all shared rides
        travellers = inData.passengers.loc[ride.indexes]
        if travellers[travellers.state == "I"].shape[0] > 0 or travellers[~(travellers.state == "I")].shape[0] > 0:
            infected_travellers = travellers[travellers.state == "I"].index
            noninfected_travellers = travellers[~(travellers.state == "I")].index
            for infected_traveller in infected_travellers:
                for noninfected_traveller in noninfected_travellers:
                    got_infected[noninfected_traveller] = day # did_i_infect_you(t, day, params)
                    infected_by[noninfected_traveller] = infected_traveller

    inData.passengers['infection_day'].update(pd.Series(got_infected))
    inData.passengers['infected_by'].update(pd.Series(infected_by))
    inData.passengers['state'] = inData.passengers.apply(lambda x: 'I' if x.infection_day == day else x.state, axis = 1)

    return inData


def infection_map(inData):
    """
    traces infections from initial ones
    :param inData:
    :return: dictionary infection_map[intial_infector][level][list of travellers infected at this step]
    """
    infection_map = dict()

    initial_infectors = inData.passengers[inData.passengers.infection_day < 0].index

    for i in initial_infectors:
        infection_map[i] = dict()
        level = 1
        infection_map[i][level] = list(inData.passengers[inData.passengers.infected_by == i].index)
        while len(infection_map[i][level]) > 0:
            level += 1
            infection_map[i][level] = list(
                inData.passengers[inData.passengers.infected_by.isin(infection_map[i][level - 1])].index)
    return infection_map


def redo_matching(inData, params, _print):
    """
    updates which travellers are still active (not quarantined),
    removes rides composed with travellers who are quarantines (at least one)
    rematches travellers who are still active
    :param inData:
    :param params:
    :param _print:
    :return:
    """

    #1. Update travellers
    inData.passengers['active'] = inData.passengers.apply(
        lambda x: True if (x.active_today and x.state != 'Q') else False, axis=1)
    inData.requests = inData.all_requests.loc[inData.passengers.active == True]

    #2. update rides
    inactives = set(inData.all_requests.loc[inData.passengers.active == False].index)

    inData.all_rides['active'] = inData.all_rides.apply(
        lambda ride: True if len(list(set(ride.indexes) & inactives)) == 0 else False, axis=1)

    # inData.all_rides['active'] = inData.all_rides.apply(lambda ride: True if
    # inData.population.loc[ride.indexes][(inData.population.loc[ride.indexes].active == False)].shape[0] == 0 else False,
    #                                                     axis=1)

    inData.sblts.rides = inData.all_rides[inData.all_rides.active]
    # do matching
    print('Re matching, {} travellers quarantined, {} inactive today. {} out of {} rides remain feasible'.format(
        inData.passengers[inData.passengers.state == "Q"].shape[0],
        inData.passengers[inData.passengers.active_today == False].shape[0],
        inData.sblts.rides.shape[0],
        inData.all_rides.shape[0])) if _print else None



    return matching(inData, params.shareability, _print = False, make_assertion = False)



def make_population(inData, params):
    """
    generates initial population of _S_uspectible travellers and initial_share if _I_nfected given days prior
    :param params: params.corona.initial_share, params.corona.infected_prior
    :param inData
    :return: inData.population DataFrame with index from inData.requests
    """
    # init population
    def is_active_today(row):
        if row.active:
            if row.state == 'Q':
                return False
            else:
                return random.random() < params.p
        else:
            return False



    share_of_active = params.corona.participation / params.corona.p
    inData.passengers['active'] = False  # part of the simulation
    inData.passengers['state'] = 'S'
    inData.passengers.active.loc[inData.requests.sample(int(share_of_active * params.nP)).index] = True
    inData.passengers['active_today'] = inData.passengers.apply(lambda x: is_active_today(x), axis=1)
    inData.passengers['platforms'] = inData.passengers.apply(lambda x: [0] if x.active_today else [-1], axis=1)
    inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0], axis=1)
    inData.sblts.requests['platform'] = inData.requests['platform']

    #inData.population = pd.DataFrame(index=inData.requests.index)
    #inData.population['state'] = 'S'  # everyone susapectible
    # first share infected
    if params.corona.one:
        # if we want to trace a single infector
        infector = inData.requests[inData.requests.kind>1].sample(1).index[0]
        inData.passengers['state'] = inData.passengers.apply(
            lambda x: 'I' if x.name == infector else 'S', axis=1)
    else:
        # if we trace share of initially infected
        inData.passengers['state'] = inData.passengers.apply(
            lambda x: 'S' if random.random() > params.corona.initial_share else 'I', axis=1)
    inData.passengers['infection_day'] = inData.passengers.apply(
        lambda r: -1 if r.state == "I" else None, axis=1)
    inData.passengers['infected_by'] = None
    return inData


def evolve(inData, params, _print = False, _plot = False):
    """
    starts with initial share of infected population and gradually infects co-riders
    :param inData:
    :param params:
    :param _print:
    :param _plot:
    :return:
    """
    def is_active_today(row):
        if row.active:
            if row.state == 'Q':
                return False
            else:
                return random.random() < params.p
        else:
            return False

    #initialise
    day = 0
    ret = dict()
    inData.all_requests = inData.requests.copy()  # keep for reference for later updates
    inData.all_rides = inData.sblts.rides.copy()
    inData = make_population(inData, params)

    ret[day] = inData.passengers.groupby('state').size()
    print(ret[day]) if _print else None

    while "I" in ret[day].index:
        day += 1
        print('day {}'.format(day))
        # quarantine
        inData.passengers['newly_quarantined'] = inData.passengers.apply(
            lambda r: False if r.infection_day is None else day - r.infection_day == params.corona.time_to_quarantine,
            axis=1) # are there newly quarantined travellers?
        if params.corona.participation_prob < 1:
            inData.passengers['active_today'] = inData.passengers.apply(lambda x: is_active_today(x), axis=1)
            inData.passengers['platforms'] = inData.passengers.apply(lambda x: [0] if x.active_today else [-1], axis=1)
            inData.requests['platform'] = inData.requests.apply(
                lambda row: inData.passengers.loc[row.name].platforms[0], axis=1)
            inData.sblts.requests['platform'] = inData.requests['platform']
        else:
            inData.passengers['active_today'] = True
        inData.passengers.state = inData.passengers.apply(lambda r: 'Q' if r.newly_quarantined else r.state, axis=1)

        # remove quarantined requests
        inData = matching(inData, params, _print) # if so, redo matching
        print('Re matching, {} travellers quarantined, {} inactive today. {} out of {} rides remain feasible'.format(
            inData.passengers[inData.passengers.state == "Q"].shape[0],
            inData.passengers[inData.passengers.active_today == False].shape[0],
            inData.sblts.rides.shape[0],
            inData.all_rides.shape[0]))
        # infection
        inData = infect(inData, day, params)
        ret[day] = inData.passengers.groupby('state').size()
        print(ret[day]) if _print else None

    inData.report = pd.DataFrame(ret)

    if _plot:
        plot_spread(inData)

    return inData

def plot_spread(inData, MODE = 'paths'):
    """
    plots results of spreading on three plots
    1. travellers in respective states (S -> I -> Q ) over days
    2. number of new infections per day
    3. maps
    :param inData:
    :param MODE:
    :return:
    """

    plt.rcParams['figure.figsize'] = [16, 4]
    plt.rcParams["font.family"] = "Helvetica"
    plt.style.use('ggplot')
    colors = sns.color_palette("muted")

    fig, axes = plt.subplots(1, 4)
    df = inData.report.T.fillna(0)
    df.plot(kind='bar', stacked=True, color=colors[:3], ax=axes[0])
    axes[0].set_ylim([0, df.Q.max() * 1.2])

    (df.Q + df.I).diff(1).shift(-1).plot(kind='bar', color=colors[0], ax=axes[1])


    axes[0].set_title('Spreading')
    axes[0].set_xlabel('day')
    axes[1].set_xlabel('day')
    axes[1].set_title('New infections [day]')

    inf_map = infection_map(inData)
    inf_map = pd.DataFrame(inf_map).T
    lens, rets, degs = dict(), dict(), dict()

    for i, infector in inf_map.iterrows():
        j = 1
        ret = infector[1].copy()
        while len(infector[j]) > 0:
            j += 1
            ret += infector[j]

        rets[i] = ret
        lens[i] = len(ret)
        degs[i] = j
    inf_map['all_infected'] = pd.Series(rets)
    inf_map['n_infected'] = pd.Series(lens)
    inf_map['degree'] = pd.Series(degs)
    inf_map.n_infected.hist(ax = axes[2])

    inf_map.degree.hist(ax=axes[3])

    plt.show()
    if 'G' in inData.keys():
        fig, ax = ox.plot_graph(inData.G, figsize = (16, 16), node_size=0, edge_linewidth=0.1,
                                show=False, close=False,
                                edge_color='white')

        inData.all_requests['ox'] = inData.all_requests.apply(lambda r: inData.nodes.loc[r.origin].x, axis=1)
        inData.all_requests['oy'] = inData.all_requests.apply(lambda r: inData.nodes.loc[r.origin].y, axis=1)
        inData.all_requests['dx'] = inData.all_requests.apply(lambda r: inData.nodes.loc[r.destination].x, axis=1)
        inData.all_requests['dy'] = inData.all_requests.apply(lambda r: inData.nodes.loc[r.destination].y, axis=1)

        for i, pop in inData.passengers.dropna().iterrows():
            req = inData.all_requests.loc[pop.name]
            if MODE == 'paths':
                if pop.infection_day <= 0:
                    ax.plot([req.ox, req.dx], [req.oy, req.dy], color='red', alpha=1, lw=5)
                else:
                    ax.plot([req.ox, req.dx], [req.oy, req.dy], color='orange',
                            alpha=math.exp(1 / pop.infection_day) - 1, lw=5-pop.infection_day/6)
            elif MODE == "o":
                if pop.infection_day <= 0:
                    ax.scatter(req.ox, req.oy, c = 'red', s=20, alpha = 1)
                else:
                    ax.scatter(req.ox, req.oy, c = 'orange', s=20-pop.infection_day,
                               alpha = min(1,math.exp(1 / pop.infection_day) - 1))
            elif MODE == "d":
                if pop.infection_day <= 0:
                    ax.scatter(req.ox, req.oy, c = 'red', s=20, alpha = 1)
                else:
                    ax.scatter(req.ox, req.oy, c = 'orange', s=20-pop.infection_day,
                               alpha = min(1,math.exp(1 / pop.infection_day) - 1))

        plt.show()



def pipe(replication_id = None):
    """ full pipeline for corona study
    reads the data and params
    runs experiments
    saves data into csvs
    """
    from ExMAS.utils import inData as inData

    params = get_config('ExMAS/data/configs/corona.json')
    params.nP = 200
    params = ExMAS.utils.make_paths(params)

    params.t0 = pd.Timestamp.now()



    inData = ExMAS.utils.load_G(inData, params, stats=True)  # download the CITY graph

    inData = ExMAS.utils.generate_demand(inData, params)

    inData.passengers['platforms'] = inData.passengers.apply(lambda x: [0], axis=1)
    inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0],
                                                        axis=1)
    inData.sblts.requests['platform'] = inData.requests['platform']
    params.without_matching = True

    inData = ExMAS.main(inData, params)


    inData = evolve(inData, params, _print=False, _plot = False)  # <---- MAIN

    ret = inData.report.T.fillna(0)
    ret.to_csv("corona_{}_init{}_repl{}.csv".format(inData.all_requests.shape[0],
                                                    params.corona.initial_share,
                                                    replication_id))
    inData.passengers.to_csv(
        "population_{}_init_{}_repl_{}.csv".format(inData.all_requests.shape[0],
                                                                      params.corona.initial_share,
                                                                      time.time()))



### deprecated

def did_i_infect_you(time, day, params):
    # generic model to determine if you infected me while we travelled together for 'time' seconds
    if time > params.corona.time_threshold:
        return day
    else:
        return False


def time_together(ride, i, j, _print = False):
    """
    determine time that two travellers spent together on a ride
    :param ride:
    :param i:
    :param j:
    :param _print:
    :return: time in seconds
    """
    degree = len(ride.indexes)
    times = pd.Series(ride.times).cumsum()
    window_i = [times[ride.indexes_orig.index(i)], times[degree+ride.indexes_dest.index(i)]]
    window_j = [times[ride.indexes_orig.index(j)], times[degree+ride.indexes_dest.index(j)]]
    overlap = min(window_j[1],window_i[1])- max(window_i[0],window_i[0])
    if _print:
        print(ride)
        print(i, j)
        print(times)
        print(window_i, window_j, overlap)
    return overlap




if __name__ == "__main__":
    pipe()








