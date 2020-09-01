import pandas as pd
import random
import ExMAS
from ExMAS.experiments import slice_space
import networkx as nx
import ExMAS.utils
import scipy
from ExMAS.utils import inData as mData
import json
from dotmap import DotMap
from ExMAS.main import init_log
import matplotlib.pyplot as plt
from ExMAS.utils import get_config
from ExMAS.main import matching
# import seaborn as sns
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
        if travellers[travellers.state == "I"].shape[0] > 0:
            infected_travellers = travellers[travellers.state == "I"].index
            noninfected_travellers = travellers[travellers.state == "S"].index
            for infected_traveller in infected_travellers:
                for noninfected_traveller in noninfected_travellers:
                    got_infected[noninfected_traveller] = day  # did_i_infect_you(t, day, params)
                    infected_by[noninfected_traveller] = infected_traveller

    inData.passengers['infection_day'].update(pd.Series(got_infected))
    inData.passengers['infected_by'].update(pd.Series(infected_by))
    inData.passengers['state'] = inData.passengers.apply(lambda x: 'I' if x.infection_day == day else x.state, axis=1)

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

    # 1. Update travellers
    inData.passengers['active'] = inData.passengers.apply(
        lambda x: True if (x.active_today and x.state != 'Q') else False, axis=1)
    inData.requests = inData.all_requests.loc[inData.passengers.active == True]

    # 2. update rides
    inactives = set(inData.all_requests.loc[inData.passengers.active == False].index)

    inData.all_rides['active'] = inData.all_rides.apply(
        lambda ride: True if len(list(set(ride.indexes) & inactives)) == 0 else False, axis=1)

    # inData.all_rides['active'] = inData.all_rides.apply(lambda ride: True if
    # inData.population.loc[ride.indexes][(inData.population.loc[ride.indexes].active == False)].shape[0] == 0 else False,
    #                                                     axis=1)

    inData.sblts.rides = inData.all_rides[inData.all_rides.active]
    # do matching
    inData.logger.info(
        'Re matching, {} travellers quarantined, {} inactive today. {} out of {} rides remain feasible'.format(
            inData.passengers[inData.passengers.state == "Q"].shape[0],
            inData.passengers[inData.passengers.active_today == False].shape[0],
            inData.sblts.rides.shape[0],
            inData.all_rides.shape[0]))

    return matching(inData, params.shareability, _print=False, make_assertion=False)


def make_population(inData, params):
    """
    generates initial population of _S_uspectible travellers and initial_share if _I_nfected given days prior
    :param params: params.corona.initial_share, params.corona.infected_prior
    :param inData
    :return: inData.population DataFrame with index from inData.requests
    """
    # init population

    share_of_active = params.corona.participation / params.corona.p
    inData.passengers['active'] = False  # part of the simulation
    inData.passengers['state'] = 'S'
    inData.passengers['quarantine_day'] = None
    inData.passengers.active.loc[inData.requests.sample(int(share_of_active * params.nP)).index] = True
    active_ones = inData.passengers[(inData.passengers.active == True)]
    active_ones = active_ones.sample(int(active_ones.shape[0] * params.corona.p))
    active_ones = active_ones[active_ones.state != 'Q']
    inData.passengers['active_today'] = False
    inData.passengers['active_today'].loc[active_ones.index] = True

    inData.passengers['platforms'] = inData.passengers.apply(lambda x: [0] if x.active_today else [-1], axis=1)
    inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0],
                                                        axis=1)
    inData.sblts.requests['platform'] = inData.requests['platform']

    # inData.population = pd.DataFrame(index=inData.requests.index)
    # inData.population['state'] = 'S'  # everyone susapectible
    # first share infected
    if params.corona.one:
        # if we want to trace a single infector
        infector = inData.requests[inData.requests.kind > 1].sample(1).index[0]
        inData.passengers['state'] = inData.passengers.apply(
            lambda x: 'I' if x.name == infector else 'S', axis=1)
    else:
        # if we trace share of initially infected
        inData.passengers['state'] = inData.passengers.apply(
            lambda x: 'S' if random.random() > params.corona.initial_share else 'I', axis=1)
    inData.passengers['infection_day'] = inData.passengers.apply(
        lambda r: 0 if r.state == "I" else None, axis=1)
    inData.passengers['infected_by'] = None
    return inData


def evolve(inData, params, _print=False, _plot=False):
    """
    starts with initial share of infected population and gradually infects co-riders
    :param inData:
    :param params:
    :param _print:
    :param _plot:
    :return:
    """

    def recovery(x):
        if x.quarantine_day == None:
            return x.state
        else:
            if x.quarantine_day + params.corona.recovery == day:
                return 'R'  # back
            else:
                return x.state

    def is_active_today(row):
        if row.active:
            if row.state == 'Q':
                return False
            else:
                return random.random() < params.corona.p
        else:
            return False

    # initialise
    day = 0
    ret = dict()
    # inData.all_requests = inData.requests.copy()  # keep for reference for later updates
    # inData.all_rides = inData.sblts.rides.copy()p
    inData = make_population(inData, params)
    ret[day] = inData.passengers.groupby('state').size()

    while "I" in ret[day].index:
        day += 1
        inData.logger.info('day {}'.format(day))
        # quarantine

        inData.passengers['newly_quarantined'] = inData.passengers.apply(
            lambda r: False if r.infection_day is None else day - r.infection_day == params.corona.time_to_quarantine,
            axis=1)  # are there newly quarantined travellers?
        inData.passengers.quarantine_day = inData.passengers.apply(
            lambda x: day if x.newly_quarantined else x.quarantine_day, axis=1)
        inData.passengers.state = inData.passengers.apply(
            lambda x: recovery(x), axis=1)
        inData.logger.info('recovered {}'.format(
            inData.passengers[inData.passengers.quarantine_day == day - params.corona.recovery].shape[0]))

        active_ones = inData.passengers[(inData.passengers.active == True)]
        active_ones = active_ones.sample(int(active_ones.shape[0] * params.corona.p))
        active_ones = active_ones[active_ones.state != 'Q']
        inData.passengers['active_today'] = False
        inData.passengers['active_today'].loc[active_ones.index] = True

        inData.passengers['platforms'] = inData.passengers.apply(lambda x: [0] if x.active_today else [-1], axis=1)

        inData.requests['platform'] = inData.requests.apply(
            lambda row: inData.passengers.loc[row.name].platforms[0], axis=1)
        inData.sblts.requests['platform'] = inData.requests['platform']

        inData.passengers.state = inData.passengers.apply(lambda r: 'Q' if r.newly_quarantined else r.state, axis=1)

        # remove quarantined requests
        inData = matching(inData, params, _print)  # if so, redo matching
        # inData.logger.warn('Day: {}\t infected: {}\t quarantined: '
        #                    '{}\t recovered: {} \t active today: {}.'.format(day,
        #                                                                     inData.passengers[
        #                                                                         inData.passengers.state == "I"].shape[
        #                                                                         0],
        #                                                                     inData.passengers[
        #                                                                         inData.passengers.state == "Q"].shape[
        #                                                                         0],
        #                                                                     inData.passengers[
        #                                                                         inData.passengers.state == "R"].shape[
        #                                                                         0],
        #                                                                     inData.passengers[
        #                                                                         inData.passengers.active_today == True].shape[
        #                                                                         0]))
        # infection
        inData = infect(inData, day, params)
        ret[day] = inData.passengers.groupby('state').size()
        inData.logger.info(ret[day])

    inData.report = pd.DataFrame(ret)

    if _plot:
        plot_spread(inData)

    if params.get('report', False):
        replication_id = random.randint(0, 100000)
        ret = inData.report.T.fillna(0)
        filename = "nP-{}_init-{}_p-{}_quarantine-{}_recovery-{}_repl-{}.csv".format(
            params.nP * params.corona.participation,
            params.corona.initial_share,
            params.corona.p,
            params.corona.time_to_quarantine,
            params.corona.recovery, replication_id)
        ret.to_csv("ExMAS/data/corona/corona_" + filename)
        inData.passengers.to_csv("ExMAS/data/corona/population_" + filename)

    return inData


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


def pipe(inData, params):
    """ full pipeline for corona study
    reads the data and params
    runs experiments
    saves data into csvs
    """

    if not params.get('use_prep', False):
        params.t0 = pd.Timestamp.now()
        inData = ExMAS.utils.load_G(inData, params, stats=True)  # download the CITY graph

        inData = ExMAS.utils.generate_demand(inData, params)

        inData.passengers['platforms'] = inData.passengers.apply(lambda x: [0], axis=1)
        inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0],
                                                            axis=1)
        inData.sblts.requests['platform'] = inData.requests['platform']

        inData = ExMAS.main(inData, params)

        if params.get('prep_only', False):
            inData.requests.to_csv('ExMAS/data/corona/requests.csv')
            inData.sblts.requests.to_csv('ExMAS/data/corona/sblt_requests.csv')
            inData.sblts.rides.to_csv('ExMAS/data/corona/rides.csv')
            inData.passengers.to_csv('ExMAS/data/corona/passengers.csv')
    else:
        inData.requests = pd.read_csv('ExMAS/data/corona/requests.csv')
        inData.sblts.requests = pd.read_csv('ExMAS/data/corona/sblt_requests.csv')
        inData.sblts.rides = pd.read_csv('ExMAS/data/corona/rides.csv')
        inData.passengers = pd.read_csv('ExMAS/data/corona/passengers.csv')
        for col in ['times', 'indexes', 'u_paxes', 'indexes_orig', 'indexes_dest']:
            inData.sblts.rides[col] = inData.sblts.rides[col].apply(lambda x: json.loads(x))
        inData.passengers.platforms = inData.passengers.platforms.apply(lambda x: json.loads(x))
        # params.just_init = True
        # inData = ExMAS.main(inData, params)

    if not params.get('prep_only', False):
        inData = evolve(inData, params, _print=False, _plot=params.plot)  # <---- MAIN


### deprecated

def did_i_infect_you(time, day, params):
    # generic model to determine if you infected me while we travelled together for 'time' seconds
    if time > params.corona.time_threshold:
        return day
    else:
        return False


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


def corona_exploit(one_slice, *args):
    # function to be used with optimize brute
    inData, params, search_space = args  # read static input
    _inData = inData.copy()
    _params = params.copy()
    stamp = dict()
    stamp['repl'] = time.time()
    # parameterize
    for i, key in enumerate(search_space.keys()):
        val = search_space[key][int(one_slice[int(i)])]
        stamp[key] = val
        print(key, val)
        if key == 'nP':
            _params.nP = val
        else:
            _params.corona[key] = val
    _inData = pipe(_inData, _params)

    return 0


def corona_run(workers=8, replications=10, search_space=None, test=False, prep = True, brute = True):
    if test:
        search_space = DotMap()
        search_space.initial_share = [0.001]
        search_space.p = [1]
    else:

        if search_space is None:
            search_space = DotMap()
            search_space.initial_share = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]
            search_space.p = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1]

    from ExMAS.utils import inData as inData
    params = get_config('ExMAS/data/configs/corona.json')
    params = ExMAS.utils.make_paths(params)
    params.logger_level = 'INFO'

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
    corona_run(workers=1, replications=1, prep = True, test=True, brute = False)
