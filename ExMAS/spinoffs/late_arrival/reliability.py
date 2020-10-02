from scipy.stats import *
import math
import pandas as pd
from dotmap import DotMap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
LOGNORM = True


def simulate_ride(ride, sp=None):
    """
    # 1. takes a scheduled shared ride with: sequence of visited origin and destination nodes
    # and travel times betwen them
    # 2. draws 1000 random:
    #        passenger arrival times of passengers at origins
    #        travel times (optionally)
    #        vehicle arrival times (optionally)
    #        service times (optionally)
    # 3. calculates 1000 realiations of travel times along the shared ride sequence
    # 4. estimates vehicle and pax travel times (o->d)
    # 5. returns df (that may be furthered visualized with 'visualise_ride')
    :param ride:
    :param sample:
    :param sp:
    :return:
    # columns:
    # "pax/veh" + "arrival/departure" + "o/d" +  pax_id  = time instances [s]
    # "pax/veh" + "wait" + "o" + pax_id = time duration in seconds
    # "time" + "pax/veh" = duartion between origin and destination in seconds
    """

    var = sp.var
    saint_ones = sp.get('saint_ones',-1)

    def samples(loc, scale, sp):
        # sample sp.sample times with a given loc, scale
        if LOGNORM:
            r = lognorm.rvs(s=sp.lognorm_s,
                                          scale = scale, loc=loc - sp.punct_shift * scale, size=sp.sample)
        else:
            r = skewnorm.rvs(a=sp.skew, loc=loc, scale=scale * math.pi / 2, size=sp.sample)
        if sp.max0:
            return np.maximum(r,loc)
        else:
            return r

    # simulate
    pax_arrivals = [samples(loc=scheduled, scale=0.1 if i+1 in saint_ones else var.pax_arrival, sp = sp)
                    for i, scheduled in enumerate(ride.scheduled_pax_arrivals)]  # sample distribution of arrival times for each pax
    df = pd.DataFrame(pax_arrivals).T
    df.index.name = 'sample'

    df.columns = ['pax_arrival_o{}'.format(_) for _ in ride.indexes]
    df['dummy_zero'] = 0

    for j, i in enumerate(ride.indexes_orig):
        if j == 0:
            df['veh_arrival_o{}'.format(i)] = samples(loc=ride.dep - var.veh_arrival, scale=var.veh_arrival, sp = sp)
        else:
            df['veh_arrival_o{}'.format(i)] = df['veh_departure_o{}'.format(ride.indexes_orig[j - 1])] + samples(
                ride.times[j], var.travel_time, sp = sp)
        df['veh_departure_o{}'.format(i)] = df[['veh_arrival_o{}'.format(i), 'pax_arrival_o{}'.format(i)]].max(axis=1)
        df['veh_wait_o{}'.format(i)] = df['veh_departure_o{}'.format(i)] - df[['veh_arrival_o{}'.format(i),'dummy_zero']].max(axis = 1) # at least one of them is zero
        df['pax_wait_o{}'.format(i)] = df['veh_departure_o{}'.format(i)] - df[
            ['pax_arrival_o{}'.format(i), 'dummy_zero']].max(axis=1)  # at least one of them is zero

        # df['pax_wait_o{}'.format(i)] = df['veh_departure_o{}'.format(i)] - df[
        #     'pax_arrival_o{}'.format(i)]  # max one is non-zero
        # df['veh_wait_o{}'.format(i)] = df['veh_departure_o{}'.format(i)] - ride.scheduled_pax_arrivals[j]
        #
        # df['pax_wait_o{}'.format(i)] = df['veh_departure_o{}'.format(i)] - ride.scheduled_pax_arrivals[j]
        # df['pax_wait_o{}'.format(i)] = df.apply(lambda x: max(0,x['pax_wait_o{}'.format(i)]),axis = 1)

    for j, i in enumerate(ride.indexes_dest):
        if j == 0:
            df['veh_arrival_d{}'.format(i)] = df['veh_departure_o{}'.format(ride.indexes_orig[-1])] + samples(
                ride.times[j + ride.degree], var.travel_time, sp = sp)
        else:
            df['veh_arrival_d{}'.format(i)] = df['veh_departure_d{}'.format(ride.indexes_dest[j - 1])] + samples(
                ride.times[j + ride.degree], var.travel_time, sp = sp)
        df['veh_departure_d{}'.format(i)] = df['veh_arrival_d{}'.format(i)]
        df['pax_arrival_d{}'.format(i)] = df['veh_arrival_d{}'.format(i)]
        if i == ride.indexes_dest[-1]:
            df['time_veh'] = df['pax_arrival_d{}'.format(i)] - ride.dep

    for i in ride.indexes:
        df['time_pax{}'.format(i)] = df['pax_arrival_d{}'.format(i)] - df['pax_arrival_o{}'.format(i)]
    del df['dummy_zero']
    return df


def evaluate_ride(ride, ride_requests, sp=None):
    # compute KPIs (time, wait, delay, utility, prob for a simulated ride
    # return aggregated results
    def utility_ns(r):
        # expected utility of shared trips
        return sp.price * r.dist / 1000 + sp.VoT * r.ttrav

    def utility_sh(r):
        # expected utility of shared trips
        return (sp.price * (1 - sp.shared_discount) * r.dist / 1000 +
                sp.VoT * sp.WtS * (r.ttrav_sh + sp.delay_value * abs(r.delay)))

    def utility_sh_sample(r, times, waits):
        # utility of sampled shared ride realizations
        return (sp.price * (1 - sp.shared_discount) * r.dist / 1000 +
                sp.VoT * sp.WtS * (times + sp.delay_value * abs(waits + r.delay)))

    # prepare
    ride = ride.squeeze()
    ride['degree'] = len(ride.indexes)
    ride['labels'] = ride.indexes_orig + ride.indexes_dest
    ride['dep'] = int(ride.times[0])  # departure time (scheduled vehicle and pax arrival at origin)
    arrivals = [ride.dep]  # scheduled node arrivals
    for i in ride.times[1:]:
        arrivals.append(i + arrivals[-1])  # propagate times sequentially
    ride['scheduled_veh_arrivals'] = arrivals
    ride['scheduled_pax_arrivals'] = ride.scheduled_veh_arrivals[:ride.degree]  # passengers expected arrival at origins

    # simulate
    df = simulate_ride(ride, sp=sp)

    # evaluate
    # ride_requests = requests.loc[ride.indexes]
    scheduled_delay = list()
    for j, i in enumerate(ride.indexes_orig):
        scheduled_delay.append(ride.scheduled_pax_arrivals[ride.indexes.index(i)] - ride_requests.loc[i].treq)
    ride_requests['delay'] = scheduled_delay
    to_add = dict()
    for i in ride_requests.index:
        req = ride_requests.loc[i]
        to_add[i] = DotMap()
        to_add[i].degree = ride.degree
        to_add[i].pos_orig = ride.indexes_orig.index(i) + 1  # which one are you picked up?
        to_add[i].pos_dest = ride.indexes_dest.index(i) + 1  # which one are you dropped off?
        to_add[i].pickups_after = ride.degree - to_add[i].pos_orig  # which one are you dropped off?
        to_add[i].points_between = to_add[i].pos_dest + ride.degree - to_add[i].pos_orig -1   # which one are you dropped off?
        to_add[i].scheduled_utility = utility_sh(req)  # how much would a shared ride cost if a schedule is right?
        to_add[i].scheduled_utility_ns = utility_ns(req) # how much would a non shared ride cost if a schedule is right?

        # what was actual utility in 1000 simulated rides?
        df['utility{}'.format(i)] = utility_sh_sample(req, df['time_pax{}'.format(i)], df['pax_wait_o{}'.format(i)])


        # utility
        to_add[i].mean_utility = df['utility{}'.format(i)].mean()  # mean value over it
        to_add[i].std_utility = df['utility{}'.format(i)].std()  # standard deviation
        to_add[i].quant_utility = df['utility{}'.format(i)].quantile(sp.quant)  # and 85th percentile
        # travel time
        to_add[i].mean_time = df['time_pax{}'.format(i)].mean()
        to_add[i].std_time = df['time_pax{}'.format(i)].std()
        to_add[i].quant_time = df['utility{}'.format(i)].quantile(sp.quant)
        # waiting time
        to_add[i].mean_wait = df['pax_wait_o{}'.format(i)].mean()
        to_add[i].std_wait = df['pax_wait_o{}'.format(i)].std()
        to_add[i].quant_time = df['utility{}'.format(i)].quantile(sp.quant)

        # how many time ride was not attractive? (deterministically)
        to_add[i].not_attractive = (df['utility{}'.format(i)] > to_add[i].scheduled_utility_ns).sum() / sp.sample
        df['prob{}'.format(i)] = df.apply(lambda x: math.exp(sp.mu * (x['utility{}'.format(i)])) / \
                                 (math.exp(sp.mu * (x['utility{}'.format(i)])) + math.exp(
                                     sp.mu * to_add[i].scheduled_utility_ns)), axis = 1)


        to_add[i].mean_prob = df['prob{}'.format(i)].mean()
        to_add[i].schedule_prob = math.exp(sp.mu * to_add[i].scheduled_utility) / \
                                 (math.exp(sp.mu * to_add[i].scheduled_utility) + math.exp(
                                     sp.mu * to_add[i].scheduled_utility_ns))

    ride_requests = pd.concat([ride_requests, pd.DataFrame(to_add).T.drop(['_typ', 'dtype'], axis=1).astype(float)], axis=1,
                              sort=False)
    ride['scheduled_utility'] = [to_add[i].scheduled_utility for i in to_add.keys()]
    df['lambda'] = 1 - df.time_veh / ride.PassHourTrav_ns
    ride['mean_time'] = df.time_veh.mean()
    ride['std_time'] = df.time_veh.std()
    ret = DotMap()
    ret.samples = df
    ret.ride = ride
    ret.requests = ride_requests
    return ret


def visualise_ride(ret, full = True):
    # make series of plots and graphs of visualized ride

    plt.rcParams['figure.figsize'] = [16, 2]

    plt.rcParams["font.family"] = "Times"
    plt.style.use('seaborn-whitegrid')
    colors = sns.color_palette("muted")

    df = ret.samples
    ride = ret.ride
    ride_requests = ret.requests
    fig, axes = plt.subplots(1, 1)
    # fig.suptitle('Schedule #{} | {}trips | seq: origs {}, dests {}'.format(ride.name,
    #                                                                        ride.degree,
    #                                                                        ride.indexes, ride.indexes_dest))
    ax = axes
    for i, kind in enumerate(['pax_arrival', 'veh_departure']):
        first = True
        for col in df.columns:
            if kind in col and df[col].std() > 0:
                d = df[col]
                if d.std()>0:
                    sns.distplot(df[col], hist = False, ax=ax, color=colors[i*2], label=kind if first else None,
                                 rug=False, hist_kws=dict(alpha=0), kde_kws={"bw":1, 'linestyle':'--' if kind == 'pax_arrival' else "solid"})
                    first = False
                ax.axvline(df[col].mean(), 0, 0.2, color=colors[i*2], linestyle="solid", lw=4)
    ax.legend()
    ax.set_xlabel('travel time [s]')
    ax.set_yticks([])
    #ax.set_title('Schedule realization')
    ax.set_ylim((0, 0.03))
    ax.set_xlim((-10, 1000))
    ym = ax.get_ylim()[1]
    labs = ['1st', '2nd','3rd', '4th pick up','1st', '2nd','3rd', '4th drop off']
    for i, tick in enumerate(ride['scheduled_veh_arrivals']):
        ax.axvline(tick, color='k', linestyle='solid', lw=2)
        ax.text(tick, ym * 1.05, labs[i])


    plt.tight_layout()
    plt.savefig('fig4.png')
    plt.show()
    if not full:
        return


    ai = 0
    plt.rcParams['figure.figsize'] = [12, 3]
    fig, axes = plt.subplots(1, ride.degree, sharex=True, sharey = True)
    #fig.suptitle('Pax and veh waits at consecutive pickups and ride delays of consecutive paxes')
    max_x = 0
    for request in ride_requests.index:
        ax = axes[ai]
        ai += 1
        request = ride_requests.loc[request]
        d = df['time_pax{}'.format(request.name)] - request.ttrav_sh
        if d.std() > 0:
            sns.distplot(d,hist = False, label=r'$\Delta T_i$'.format(request.name),kde_kws={"bw":1, 'linestyle':'solid'},
                         ax=ax, rug=False, hist_kws=dict(alpha=0))
        max_x = max(max_x,(df['time_pax{}'.format(request.name)] - request.ttrav_sh).max())
        sns.distplot(df['veh_wait_o{}'.format(request.name)], hist = False,label=r'$V_i$', ax=ax,kde_kws={"bw":1, 'linestyle':'dotted'},
                     rug=False, hist_kws=dict(alpha=0))
        max_x = max(max_x, df['veh_wait_o{}'.format(request.name)].max())
        d = df['pax_wait_o{}'.format(request.name)]
        if d.std() > 0:
            sns.distplot(d, hist = False,label=r'$WO_i$', ax=ax,kde_kws={'lw':2 , "bw":1, 'linestyle':'dashed'},
                         rug=False, hist_kws=dict(alpha=0))

        ax.set_xlabel(r'delay [s]')
        max_x = max(max_x, df['pax_wait_o{}'.format(request.name)].max())

        ax.set_ylim((0, 0.03))
        ax.set_xlim((0, 100))
        ax.set_yticks([])
    # for ax in axes:
    #     ax.set_xlim((0, max_x))
    #axes[0].set_ylabel('prob')
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    axes[2].get_legend().remove()
    axes[0].set_title('1st')
    axes[1].set_title('2nd')
    axes[2].set_title('3rd')
    axes[3].set_title('4th passenger')
    axes[-1].legend()
    plt.tight_layout()
    plt.savefig('fig5.png')
    plt.show()

    fig, axes = plt.subplots(1, ride.degree, sharex=True, sharey = True)
    fig.suptitle('Utilities $\Delta$ - scheduled vs. realized')
    ai = 0
    for request in ride_requests.index:
        ax = axes[ai]
        ai += 1
        request = ride_requests.loc[request]
        ser = -(df['utility{}'.format(request.name)] -
                ride.scheduled_utility[ai - 1]) / ride.scheduled_utility[ai - 1]
        if ser.std()>0:
            sns.distplot(ser,
                         label='$\Delta$ utility', ax=ax,
                         rug=True, hist_kws=dict(alpha=0.1))
        ax.axvline(ser.mean(), color='k', linestyle='--', lw=1)
        ax.axvline(0, color='k', linestyle='dotted', lw=0.5)
        # ax.set_xlim((ser.quantile(0), ser.quantile(1)))
        ax.set_ylim((0, 50))
    axes[0].legend()
    plt.tight_layout()
    plt.show()
    fig, axes = plt.subplots(1, 1)
    fig.suptitle('Sharing probability CDF')
    for request in ride_requests.index:
        ax = axes
        request = ride_requests.loc[request]
        df['prob{}'.format(request.name)].hist(cumulative=True, density=True, bins=100, ax=ax, alpha = 0.6,
                                               label = request.name)
        ax.axvline(request.schedule_prob, color='k', linestyle='dotted', lw=1)
        ax.axvline(df['prob{}'.format(request.name)].mean(), color='k', linestyle='--', lw=2)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    sns.distplot(df.time_veh,
                 label='ride time', ax=ax,
                 rug=True, hist_kws=dict(alpha=0.1), color = colors[0])
    ax.axvline(df.time_veh.mean(), color='k', linestyle='--', lw=2)
    #ax.axvline(ride.u_veh, color='k', linestyle='dotted', lw=1)
    #ax.set_xlim((ride.u_veh - 2, ax.get_xlim()[1]))
    ax.legend()

    ax = axes[1]
    sns.distplot(df['lambda'],
                 label='ride profitability', ax=ax,
                 rug=True, hist_kws=dict(alpha=0.1), color=colors[1])
    ax.axvline(df['lambda'].mean(), color='k', linestyle='--', lw=2)
    ax.axvline(ride['lambda_r'], color='k', linestyle='dotted', lw=1)
    ax.legend()


    fig.suptitle('Ride time and profitability distributions')
    plt.tight_layout()
    plt.show()



def pipe(schedule, requests, sp, kind =21, vis = False, id = None, vis_full=True):
    """
    - Takes one ride from the schedule,
    - simulates it according to parameterization in 'sp'
    - evaluates the outcome of simulation
    - optionally visualizes
    - returns simulation results and aggregrated results
    :param schedule: set of shared rides (DataFrame)
    :param requests: set of requested trips (DataFrame)
    :param sp: parameters
    :param kind: kind of trip (degree) to be sampled from schedule - optional
    :param vis: fla whether to visualize the simulation or not
    :param id: id of trip in the schedule to simulate - optional
    :param vis_full: flag to visualize only schedule or also KPIs (by default True)
    :return: out put of visualize ride, DotMap with dataFrames of simulated ride
    """
    if id is None:
        ride = schedule[schedule.kind == kind].sample(1)
    else:
        ride = schedule.loc[id]

    ride_requests = requests.loc[ride.squeeze().indexes]


    ret = evaluate_ride(ride, ride_requests, sp = sp)
    if vis:
        visualise_ride(ret, vis_full)
    return ret


def simulate_all(schedule, requests, sp):
    # simulates all rides in the schedule
    ret_trips = list()
    ret_rides = dict()
    for i in schedule.index:
        sp.var.travel_time = 0
        sp.var.veh_arrival = 0
        sp.var.pax_arrival = 60
        sp.shared_discount = 0.6
        ride = schedule.loc[i]
        ride_requests = requests.loc[ride.squeeze().indexes]
        r = evaluate_ride(ride, ride_requests, sp=sp)
        ret_trips.append(r.requests)
        ret_rides[i] = r.ride
    ret = DotMap()
    ret.trips = pd.concat(ret_trips, sort= False)
    ret.rides = pd.DataFrame(ret_rides).T
    return ret


def explore_one(schedule, requests, sp, kind=40, explore_range = range(0,360,30), key = 'pax_arrival', id = None):
    """
    run series of simulations with different input, defined through the 'explore_range'
    :param schedule:
    :param requests:
    :param sp:
    :param kind:
    :param explore_range:
    :param key:
    :param id:
    :return:
    """
    if id is None:
        r = schedule[schedule.kind == kind].sample(1).squeeze()
    else:
        r = schedule.loc[id]
    ride_requests = requests.loc[r.indexes]
    trips = DotMap()
    for i in explore_range:
        sp.var[key] = i

        simulated = evaluate_ride(r, ride_requests, sp=sp)
        for j in r.indexes:
            s = simulated.requests.loc[j]
            s.name = 'key'
            trips[j][i] = s
    ret = DotMap()
    for j in r.indexes:
        ret[j] = pd.DataFrame(dict(trips[j]))


    return ret







