#!/usr/bin/env python3

"""
# ExMAS Virus spreading
> Functions and modules used to simulate virus spreading on ride-pooling networks
---

----
Rafał Kucharski, TU Delft, 2020 r.m.kucharski (at) tudelft.nl
"""

import pandas as pd
import random
import ExMAS
from ExMAS.experiments import slice_space
import networkx as nx
import ExMAS.utils
import scipy
import json
from dotmap import DotMap
from ExMAS.main import init_log
import matplotlib.pyplot as plt
from ExMAS.utils import get_config
from ExMAS.main import matching

import math
import osmnx as ox
import time


def infect(inData, day, params):
    """
    for a given infected travellers and their shared rides schedule return requests with newly_infected travellers
    :param inData: container with schedule: shared rides | requests: travellers and their trips requests
    :param day: consecutive day of simulation
    :param params: params: parameters
    :return: equests with column newly_infected
    """
    got_infected = dict()
    infected_by = dict()
    for i, ride in inData.sblts.schedule.iterrows():  # iterate over all shared rides
        travellers = inData.passengers.loc[ride.indexes]  # travellers of this shared ride
        if travellers[travellers.state == "I"].shape[0] > 0:  # is anyone infected
            infected_travellers = travellers[travellers.state == "I"].index  # infected ones
            noninfected_travellers = travellers[travellers.state == "S"].index   # susceptible ones
            for infected_traveller in infected_travellers: # let's infect: all infected
                for noninfected_traveller in noninfected_travellers:  # infect susceptible
                    got_infected[noninfected_traveller] = day  # output: when you got infected
                    infected_by[noninfected_traveller] = infected_traveller  # output: by whom

    inData.passengers['infection_day'].update(pd.Series(got_infected))  # update infection days for those infected
    inData.passengers['infected_by'].update(pd.Series(infected_by))  # update by whom infected
    # change state of those infected today
    inData.passengers['state'] = inData.passengers.apply(lambda x: 'I' if x.infection_day == day else x.state, axis=1)

    return inData


def make_population(inData, params):
    """
    generates initial population of _S_uspectible travellers and initial_share if _I_nfected given days prior
    it determines from total population, who will take part in pool used in D2D simulations (active)
    and determines who is active on first day (active_today)
    :param params: params.corona.initial_share, params.corona.infected_prior
    :param inData
    :return: inData.population DataFrame with index from inData.requests
    """
    # init population
    inData.passengers['active'] = False
    inData.passengers['state'] = 'S'
    inData.passengers['quarantine_day'] = None
    inData.passengers['infected_by'] = None

    # active D2D
    share_of_active = params.corona.participation / params.corona.p  # determine pool of travellers to draw everyday
    inData.passengers.active.loc[inData.requests.sample(int(share_of_active *
                                                            params.nP)).index] = True  # those will play in D2D simulations

    # active today
    active_ones = inData.passengers[(inData.passengers.active == True)]
    active_ones = active_ones.sample(int(active_ones.shape[0] * params.corona.p))  # those are active today
    active_ones = active_ones[active_ones.state != 'Q']  # except those quarantined
    inData.passengers['active_today'] = False
    inData.passengers['active_today'].loc[active_ones.index] = True  # those will be matched and then may be infected

    # if platform is [-1] passenger is not matched
    inData.passengers['platforms'] = inData.passengers.apply(lambda x: [0] if x.active_today else [-1], axis=1)
    inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0],
                                                        axis=1)
    inData.sblts.requests['platform'] = inData.requests['platform']

    # infect randomly initial_share of travellers
    inData.passengers['state'] = inData.passengers.apply(
            lambda x: 'S' if random.random() > params.corona.initial_share else 'I', axis=1)
    # but only those active
    inData.passengers['state'] = inData.passengers.apply(lambda x: 'S' if not x.active else x.state, axis=1)
    inData.passengers['infection_day'] = inData.passengers.apply(
        lambda r: 0 if r.state == "I" else None, axis=1)

    return inData


def evolve(inData, params, _print=False, _plot=False):
    """
    Day to Day evolution of virus spreading from initial population
    starts with initial share of infected population and gradually infects co-riders
    :param inData:
    :param params:
    :param _print:
    :param _plot:
    :return:
    """

    def recovery(x):
        # did I recover today (after quarantine)
        if x.quarantine_day == None:
            return x.state
        else:
            if x.quarantine_day + params.corona.recovery == day:
                return 'R'  # back
            else:
                return x.state

    # initialise
    day = 0
    ret = dict() # report - trace number of passengers in each state
    inData = make_population(inData, params)  # determine who is active (pool to draw everyday) who is active today
    # and initially infected
    ret[day] = inData.passengers.groupby('state').size()

    while "I" in ret[day].index: # main D2D loop, until there are still infected (we do not care about Quarantined)
        day += 1 # incerement
        inData.logger.info('day {}'.format(day))

        # quarantines
        inData.passengers['newly_quarantined'] = inData.passengers.apply(
            lambda r: False if r.infection_day is None else day - r.infection_day == params.corona.time_to_quarantine,
            axis=1)  # are there newly quarantined travellers?
        inData.passengers.quarantine_day = inData.passengers.apply(
            lambda x: day if x.newly_quarantined else x.quarantine_day, axis=1)
        inData.passengers.state = inData.passengers.apply(lambda r: 'Q' if r.newly_quarantined else r.state, axis=1)

        # recoveries
        inData.passengers.state = inData.passengers.apply(
            lambda x: recovery(x), axis=1)

        # active today
        active_ones = inData.passengers[(inData.passengers.active == True)]
        active_ones = active_ones.sample(int(active_ones.shape[0] * params.corona.p))  # those are active today
        active_ones = active_ones[active_ones.state != 'Q']  # except those quarantined
        inData.passengers['active_today'] = False
        inData.passengers['active_today'].loc[
            active_ones.index] = True  # those will be matched and then may be infected

        # if platform is [-1] passenger is not matched
        inData.passengers['platforms'] = inData.passengers.apply(lambda x: [0] if x.active_today else [-1], axis=1)
        inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0],
                                                            axis=1)
        inData.sblts.requests['platform'] = inData.requests['platform']

        # redo matching
        inData = matching(inData, params, _print)

        # and infect
        inData = infect(inData, day, params)
        ret[day] = inData.passengers.groupby('state').size()
        inData.logger.info(ret[day])
        inData.logger.info('Day: {}\t infected: {}\t quarantined: '
                           '{}\t recovered: {} \t susceptible: {}, active today: {}.'.format(day,
                                                                            inData.passengers[
                                                                                inData.passengers.state == "I"].shape[
                                                                                0],
                                                                            inData.passengers[
                                                                                inData.passengers.state == "Q"].shape[
                                                                                0],
                                                                            inData.passengers[
                                                                                inData.passengers.state == "R"].shape[
                                                                                0],
                                                                            inData.passengers[
                                                                                inData.passengers.state == "S"].shape[
                                                                                0],
                                                                            inData.passengers[
                                                                                inData.passengers.active_today == True].shape[
                                                                                0]))
        # go to next day (if still anyone is infected)

    # end of the loop
    inData.report = pd.DataFrame(ret)

    if _plot:
        plot_spread(inData)

    if params.get('report', False): # store results to csvs
        replication_id = random.randint(0, 100000)
        ret = inData.report.T.fillna(0)
        filename = "nP-{}_init-{}_p-{}_quarantine-{}_recovery-{}_repl-{}.csv".format(
            params.nP * params.corona.participation,
            params.corona.initial_share,
            params.corona.p,
            params.corona.time_to_quarantine,
            params.corona.recovery, replication_id)
        ret.to_csv("ExMAS/data/corona/corona_" + filename)  # day to day evolution of travellers in each state
        inData.passengers.to_csv("ExMAS/data/corona/population_" + filename)  # pax info (when infected and by whom)

    return inData


def pipe(inData, params):
    """ full pipeline for corona study
    reads the data and params
    runs experiments
    saves data into csvs
    """

    if not params.get('use_prep', False):
        # start from scratch with ExMAS
        params.t0 = pd.Timestamp.now()
        inData = ExMAS.utils.load_G(inData, params, stats=True)  # download the CITY graph

        inData = ExMAS.utils.generate_demand(inData, params)  # generate demand

        inData.passengers['platforms'] = inData.passengers.apply(lambda x: [0], axis=1)
        inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0],
                                                            axis=1)
        inData.sblts.requests['platform'] = inData.requests['platform']

        inData = ExMAS.main(inData, params)  # do the ExMAS

        if params.get('prep_only', False):
            # store results of ExMAS
            inData.requests.to_csv('ExMAS/data/corona/requests.csv')
            inData.sblts.requests.to_csv('ExMAS/data/corona/sblt_requests.csv')
            inData.sblts.rides.to_csv('ExMAS/data/corona/rides.csv')
            inData.passengers.to_csv('ExMAS/data/corona/passengers.csv')
    else:
        # read input demand and matching data
        inData.requests = pd.read_csv('ExMAS/data/corona/requests.csv')
        inData.sblts.requests = pd.read_csv('ExMAS/data/corona/sblt_requests.csv')
        inData.sblts.rides = pd.read_csv('ExMAS/data/corona/rides.csv')
        inData.passengers = pd.read_csv('ExMAS/data/corona/passengers.csv')
        for col in ['times', 'indexes', 'u_paxes', 'indexes_orig', 'indexes_dest']:
            inData.sblts.rides[col] = inData.sblts.rides[col].apply(lambda x: json.loads(x))
        inData.passengers.platforms = inData.passengers.platforms.apply(lambda x: json.loads(x))

    if not params.get('prep_only', False):
        inData = evolve(inData, params, _print=False, _plot=params.plot)  # <---- MAIN


def corona_exploit(one_slice, *args):
    # function to be used with optimize brute
    inData, params, search_space = args  # read static input
    _inData = inData.copy()
    _params = params.copy()
    stamp = dict()
    stamp['repl'] = time.time()
    # parameterize for this simulation
    for i, key in enumerate(search_space.keys()):
        val = search_space[key][int(one_slice[int(i)])]
        stamp[key] = val
        print(key, val)
        if key == 'nP':
            _params.nP = val
        else:
            _params.corona[key] = val
    pipe(_inData, _params)  # MAIN

    return 0

# utils

def plot_spread(inData, MODE='paths'):
    """
    plots results of spreading on three plots
    1. travellers in respective states (S -> I -> Q ) over days
    2. number of new infections per day
    3. maps
    :param inData:
    :param MODE:
    :return:
    """
    import seaborn as sns
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
    inf_map.n_infected.hist(ax=axes[2])

    inf_map.degree.hist(ax=axes[3])

    plt.show()
    if 'G' in inData.keys():
        fig, ax = ox.plot_graph(inData.G, figsize=(16, 16), node_size=0, edge_linewidth=0.1,
                                show=False, close=False,
                                edge_color='white')

        inData.passengers['ox'] = inData.requests.apply(lambda r: inData.nodes.loc[r.origin].x, axis=1)
        inData.passengers['oy'] = inData.requests.apply(lambda r: inData.nodes.loc[r.origin].y, axis=1)
        inData.passengers['dx'] = inData.requests.apply(lambda r: inData.nodes.loc[r.destination].x, axis=1)
        inData.passengers['dy'] = inData.requests.apply(lambda r: inData.nodes.loc[r.destination].y, axis=1)

        for i, pop in inData.passengers.dropna().iterrows():
            req = inData.passengers.loc[pop.name]
            if MODE == 'paths':
                if pop.infection_day <= 0:
                    ax.plot([req.ox, req.dx], [req.oy, req.dy], color='red', alpha=1, lw=5)
                else:
                    ax.plot([req.ox, req.dx], [req.oy, req.dy], color='orange',
                            alpha=math.exp(1 / pop.infection_day) - 1, lw=5 - pop.infection_day / 6)
            elif MODE == "o":
                if pop.infection_day <= 0:
                    ax.scatter(req.ox, req.oy, c='red', s=20, alpha=1)
                else:
                    ax.scatter(req.ox, req.oy, c='orange', s=20 - pop.infection_day,
                               alpha=min(1, math.exp(1 / pop.infection_day) - 1))
            elif MODE == "d":
                if pop.infection_day <= 0:
                    ax.scatter(req.ox, req.oy, c='red', s=20, alpha=1)
                else:
                    ax.scatter(req.ox, req.oy, c='orange', s=20 - pop.infection_day,
                               alpha=min(1, math.exp(1 / pop.infection_day) - 1))

        plt.show()


def did_i_infect_you(time, day, params):
    # generic model to determine if you infected me while we travelled together for 'time' seconds
    # can be extended to cover for non-detemrnistic infection
    if time > params.corona.time_threshold:
        return day
    else:
        return False


def infection_map(inData):
    """
    traces infections from initial ones
    :param inData: with fields populated after the simulation is done
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


def time_together(ride, i, j, _print=False):
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
    window_i = [times[ride.indexes_orig.index(i)], times[degree + ride.indexes_dest.index(i)]]
    window_j = [times[ride.indexes_orig.index(j)], times[degree + ride.indexes_dest.index(j)]]
    overlap = min(window_j[1], window_i[1]) - max(window_i[0], window_i[0])
    if _print:
        print(ride)
        print(i, j)
        print(times)
        print(window_i, window_j, overlap)
    return overlap


def active_today(inData, params):
    active_ones = inData.passengers[(inData.passengers.active == True)]
    active_ones = active_ones.sample(int(active_ones.shape[0] * params.corona.p))  # those are active today
    active_ones = active_ones[active_ones.state != 'Q']  # except those quarantined
    inData.passengers['active_today'] = False
    inData.passengers['active_today'].loc[
        active_ones.index] = True  # those will be matched and then may be infected

    # if platform is [-1] passenger is not matched
    inData.passengers['platforms'] = inData.passengers.apply(lambda x: [0] if x.active_today else [-1], axis=1)
    inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0],
                                                        axis=1)
    inData.sblts.requests['platform'] = inData.requests['platform']
    return inData


def degrees(replications = 10):
    # simulate contact network evolution (without matching)
    from ExMAS.utils import inData as inData
    params = ExMAS.utils.get_config('ExMAS/spinoffs/corona.json')  # load the default
    params.nP = 3200
    params.corona.participation = 0.65

    inData.requests = pd.read_csv('ExMAS/data/corona/results/requests.csv')
    inData.sblts.requests = pd.read_csv('ExMAS/data/corona/results/sblt_requests.csv')
    inData.sblts.rides = pd.read_csv('ExMAS/data/corona/results/rides.csv')
    inData.passengers = pd.read_csv('ExMAS/data/corona/results/passengers.csv')
    for col in ['times', 'indexes', 'u_paxes', 'indexes_orig', 'indexes_dest']:
        inData.sblts.rides[col] = inData.sblts.rides[col].apply(lambda x: json.loads(x))
    inData.passengers.platforms = inData.passengers.platforms.apply(lambda x: json.loads(x))

    ret = list()
    for repl in range(replications):
        for p in [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1]:
            params.corona.p = p
            inData = make_population(inData,params)
            for day in range(50):
                inData = active_today(inData, params)
                inData.sblts.rides['platform'] = inData.sblts.rides.apply(lambda row: list(set(inData.requests.loc[row.indexes].platform.values)),
                                                axis=1)

                inData.sblts.rides['platform'] = inData.sblts.rides.platform.apply(lambda x: -2 if len(x) > 1 else x[0])
                G_today = ExMAS.utils.make_shareability_graph(inData.sblts.requests[inData.sblts.requests.platform == 0],
                                   inData.sblts.rides[inData.sblts.rides.platform >-2])
                if day == 0:
                    G = G_today
                else:
                    G = nx.compose(G, G_today)
                degree = pd.Series([G.degree(n) for n in G.nodes()])
                ret.append({'repl':repl, 'p':p,'day':day, 'mean':degree.mean(),'std':degree.std()})
                print(ret[-1])
    pd.DataFrame(ret).to_csv('degrees.csv')


def corona_run(workers=8, replications=10, search_space=None, test=False, prep=True, brute=True):
    if test:
        search_space = DotMap()
        search_space.initial_share = [0.001]
        search_space.p = [1]
    else:

        if search_space is None:
            search_space = DotMap()
            search_space.initial_share = [0.001, 0.002, 0.005, 0.01]
            search_space.p = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1]

    from ExMAS.utils import inData as inData
    params = get_config('ExMAS/data/configs/corona.json')
    params = ExMAS.utils.make_paths(params)
    params.logger_level = 'INFO'
    params.nP = 3200
    params.corona.participation = 2000 / 3200
    params.shared_discount = 0.2
    #params.corona.p = 1
    #params.corona.initial_share = 0.001
    params.WtS = 1.35

    if prep:
        params.prep_only = True
        pipe(inData, params)
    else:
        inData.logger = init_log(params)  #
    params.prep_only = False
    params.use_prep = True
    if brute:
        scipy.optimize.brute(func=corona_exploit,
                             ranges=slice_space(search_space, replications=replications),
                             args=(inData, params, search_space),
                             full_output=True,
                             finish=None,
                             workers=workers)
    else:
        pipe(inData, params)


if __name__ == "__main__":
    corona_run(workers=2, replications=10, prep=False, test=False, brute=True)


