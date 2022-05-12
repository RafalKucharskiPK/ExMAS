import math

import numpy as np
import pandas as pd
import random







def prepare_PoA(inData, CALC_SUBRIDES = False):
    """
    precompute attributes needed for Equilibrium based matching
    :param inData:
    :return:
    """
    # A: Prepare incidence matrix (copied from matching)
    im = inData.sblts.rides
    r = inData.sblts.requests

    request_indexes = dict()
    request_indexes_inv = dict()
    for i, index in enumerate(r.index.values):
        request_indexes[index] = i
        request_indexes_inv[i] = index

    im_indexes = dict()
    im_indexes_inv = dict()
    for i, index in enumerate(im.index.values):
        im_indexes[index] = i
        im_indexes_inv[i] = index

    r = r.reset_index()

    nR = r.shape[0]

    def add_binary_row(r):
        ret = np.zeros(nR)
        for i in r.indexes:
            ret[request_indexes[i]] = 1
        return ret

    step = 1
    inData.logger.warn('Prepare for game:  {}/6'.format(step))
    step+=1

    im['row'] = im.apply(add_binary_row, axis=1)  # row to be used as constrain in optimization
    mtx = np.vstack(im['row'].values).T  # creates a numpy array for the constrains
    m = pd.DataFrame(mtx).astype(int)

    im['index'] = im.index.copy()

    im = im.reset_index(drop=True)

    m.index.name = 'trips'
    m.columns.name = 'rides'

    m_user_costs = m.copy()

    for col in m.columns:
        new_col = [0] * inData.sblts.requests.shape[0]
        indexes = inData.sblts.rides.loc[col]['indexes']
        u_paxes = inData.sblts.rides.loc[col].u_paxes
        for l, i in enumerate(indexes):
            new_col[i] = u_paxes[l]
        m_user_costs[col] = new_col
    m_user_costs = m_user_costs.round(1)
    m_user_costs = m_user_costs.replace(0, np.nan)

    ranking = m_user_costs.rank(1, ascending=True, method='first')  # rank rides by lowest cost

    inData.logger.warn('Prepare for game:  {}/6'.format(step))
    step+=1


    beta = -10  # behavioural parameter

    # matrices: row = pax , col = ride , val = cost, ranking, prob, etc.
    probs = m_user_costs.replace(0, np.inf)  # set rides without this pax to inf
    probs = probs.applymap(lambda x: math.exp(beta * x))  # calculate exp to MNL
    probs = probs.div(probs.sum(axis=1), axis=0)  # divide by sum of utilities
    rel_ranking = ranking.div(ranking.max(axis=1), axis=0)

    inData.sblts.ranking_matrix = ranking
    inData.sblts.rel_ranking_matrix = rel_ranking
    inData.sblts.incidence_matrix = m
    inData.sblts.m = mtx
    inData.sblts.cost_matrix = m_user_costs
    inData.sblts.probabilities_matrix = probs

    # compute degrees
    inData.sblts.rides['degree'] = inData.sblts.rides.apply(lambda x: len(x.indexes), axis=1)


    # delays
    inData.sblts.rides['treqs'] = inData.sblts.rides.apply(lambda x: inData.sblts.requests.loc[x.indexes].treq.values,
                                                           axis=1)

    inData.logger.warn('Prepare for game:  {}/6'.format(step))
    step+=1

    def calc_deps(r):
        deps = [r.times[0]]
        for d in r.times[1:r.degree]:
            deps.append(deps[-1] + d)  # departure times
        t = inData.sblts.requests
        return deps

    inData.sblts.rides['deps'] = inData.sblts.rides.apply(calc_deps, axis=1)

    inData.sblts.rides['delays'] = inData.sblts.rides['deps'] - inData.sblts.rides['treqs']

    inData.sblts.rides['ttravs'] = inData.sblts.rides.apply(lambda r: [sum(r.times[i + 1:r.indexes_orig.index(r.indexes[i]) + r.degree+ 1 + r.indexes_dest.index(r.indexes[i])]) for i in range(r.degree)], axis = 1)

    inData.sblts.rides['pruned'] = True

    multis = list()
    for i, ride in inData.sblts.rides.iterrows():
        for t in ride.indexes:
            multis.append([ride.name, t])
    multis = pd.DataFrame(index=pd.MultiIndex.from_tuples(multis))

    inData.logger.warn('Prepare for game:  {}/6'.format(step))
    step+=1

    multis['ride'] = multis.index.get_level_values(0)
    multis['traveller'] = multis.index.get_level_values(1)
    multis = multis.join(inData.sblts.requests[['treq', 'dist', 'ttrav']], on='traveller')
    multis = multis.join(inData.sblts.rides[['u_veh', 'u_paxes','degree', 'indexes', 'ttravs', 'delays']], on='ride')
    multis['order'] = multis.apply(lambda r: r.indexes.index(r.traveller), axis=1)
    multis['ttrav_sh'] = multis.apply(lambda r: r.ttravs[r.order], axis=1)
    multis['delay'] = multis.apply(lambda r: r.delays[r.order], axis=1)
    #multis['u'] = multis.apply(lambda r: r.u_paxes[r.order], axis=1)
    multis['shared'] = multis.degree>1
    multis['ride_time'] = multis.u_veh
    multis = multis[['ride','traveller','shared', 'degree','treq','ride_time','dist','ttrav','ttrav_sh','delay']]
    inData.sblts.rides_multi_index = multis

    rides = inData.sblts.rides
    rides['indexes_set'] = rides.indexes.apply(set)

    def givemesubsets(row):
        # returns list of all the subgroup indiced contained in a ride
        return rides[rides.indexes_set.apply(lambda x: x.issubset(row.indexes_set))].index.values

    rides['subgroups'] = rides.apply(givemesubsets, axis=1)

    def givemesupersets(row):
        # returns list of all the subgroup indiced contained in a ride
        return rides[rides.indexes_set.apply(lambda x: row.indexes_set.issubset(x))].index.values

    rides['supergroups'] = rides.apply(givemesupersets, axis=1)

    inData.sblts.rides = rides

    inData.logger.warn('Prepare for game:  {}/6'.format(step))
    step+=1



    # possible obj functions to PoA (per ride, not per traveller in ride)

    # A: mean ranking for each ride
    inData.sblts.rides['rankings'] = inData.sblts.rides.apply(
        lambda ride: [ranking.loc[traveller][ride.name] for traveller in ride.indexes], axis=1)
    inData.sblts.rides['mean_ranking'] = inData.sblts.rides.apply(lambda ride: sum(ride.rankings) / len(ride.indexes),
                                                                  axis=1)

    # B: mean relative ranking for each ride
    inData.sblts.rides['rel_rankings'] = inData.sblts.rides.apply(
        lambda ride: [rel_ranking.loc[traveller][ride.name] for traveller in ride.indexes], axis=1)
    inData.sblts.rides['mean_rel_ranking'] = inData.sblts.rides.apply(
        lambda ride: sum(ride.rel_rankings) / len(ride.indexes),
        axis=1)
    # C: mean PoA for travellers
    inData.sblts.rides['PoAs'] = inData.sblts.rides.apply(
        lambda ride: [m_user_costs.loc[traveller][ride.name] - m_user_costs.loc[traveller].min() for traveller in
                      ride.indexes], axis=1)

    inData.sblts.rides['mean_PoA'] = inData.sblts.rides.apply(lambda ride: sum(ride.PoAs) / len(ride.indexes), axis=1)

    # D: total PoA
    inData.sblts.rides['total_PoA'] = inData.sblts.rides.apply(lambda ride: sum(ride.PoAs), axis=1)

    # E: Sum of Squares PoA
    inData.sblts.rides['squared_PoA'] = inData.sblts.rides.apply(
        lambda ride: sum(_ * _ for _ in ride.PoAs) / len(ride.indexes), axis=1)

    # Probabilities
    inData.sblts.rides['probs'] = inData.sblts.rides.apply(
        lambda ride: [inData.sblts.probabilities_matrix.loc[traveller][ride.name] for traveller in
                      ride.indexes], axis=1)

    # minimal probability
    inData.sblts.rides['min_prob'] = inData.sblts.rides.apply(lambda ride: min(ride.probs), axis=1)

    # log sum of probabilities
    inData.sblts.rides['logsum_prob'] = inData.sblts.rides.apply(lambda ride: sum(math.log(_) for _ in ride.probs),
                                                            axis=1)

    inData.logger.warn('Prepare for game:  {}/6'.format(step))
    step+=1

    if CALC_SUBRIDES:
        # identify rides contained in rides eg. ride (1,3) is contained in ride (1,3,5)
        ret = dict()
        sums = mtx.sum(axis=0)
        for i in range(mtx.shape[1]):  # check this ride
            ret[i] = list()
            for j in range(mtx.shape[1]):  # if other rides are contained with it
                if j != i:  # if they are different
                    if sums[j] < sums[i]:  # and of lower degree
                        # and the contiaining ride j contains all travellers of contained ride i
                        if sum(mtx[:, j] * mtx[:, i]) == sums[j]:
                            ret[i].append(j)  # append for a dict
        inData.sblts.rides['subrides'] = pd.Series(ret)  # store in the DataFrame

    return inData


def calc_solution_PoA(inData):
    """
    calc price of anarchy per solution
    :param inData:
    :return:
    """
    # prep
    indexes = dict()
    utilities = dict()
    for _ in inData.sblts.requests.index:
        indexes[_] = list()
        utilities[_] = list()
    for i, row in inData.sblts.rides.iterrows():
        for j, traveller in enumerate(row.indexes):
            indexes[traveller] += [i]
            utilities[traveller] += [row.u_paxes[j]]

    inData.sblts.requests['ride_indexes'] = pd.Series(indexes)
    inData.sblts.requests['ride_utilities'] = pd.Series(utilities)

    # best possible
    inData.sblts.requests['best'] = inData.sblts.requests['ride_utilities'].apply(lambda x: min(x))
    # worst possible
    inData.sblts.requests['worst'] = inData.sblts.requests['ride_utilities'].apply(lambda x: max(x))
    inData.sblts.requests['PoA'] = inData.sblts.requests['u_sh'] - inData.sblts.requests['best']
    inData.sblts.requests['PoA_relative'] = (inData.sblts.requests['u_sh'] - inData.sblts.requests['best']) / \
                                            inData.sblts.requests['best']
    inData.sblts.requests['ranking'] = inData.sblts.requests.apply(
        lambda x: int(inData.sblts.ranking_matrix.loc[x.name][x.ride_id]), axis=1)
    return inData


def test_leader_follower(inData,nShuffle = 1):
    ret = dict()
    for i in range(nShuffle):
        inData.logger.warning('leader-follower shuffle no:{}'.format(i))
        solution = leader_follower(inData, shuffle=True)
        inData.sblts.rides.selected = inData.sblts.rides.apply(lambda x: 1 if x.name in solution else 0, axis=1)
        inData.sblts.schedule = inData.sblts.rides[inData.sblts.rides.selected == 1].copy()
        inData = calc_solution_PoA(inData)
        ret[i] = KPIs(inData)
        inData.logger.info(ret[i])
    return pd.DataFrame(ret)


def leader_follower(inData, shuffle = True, costs = 'ranking_matrix'):
    """
    solves the leader-follower traveller-based algorithm
    it assigns random order of decision making travellers
    each of them selects best option for him (group)
    and selects for his co-travellers from the group (followers)
    :param inData:
    :param shuffle:
    :param costs: what does traveller look at making decisions (matrix from inData.sblts)
    :return: indexes of selected rides
    """
    served = list() # list of those already served (as a decision-makers and when decisiom was made by others)
    selected = list() # list of selected rides
    ranks = inData.sblts[costs].values.copy() # matrix where rows are travellers and columns are groups (rides)
    # filled with nans for infeasible combinations
    # and with values for incidence (traveller being part of the group)
    # values may be: utility, probability or ranking
    travellers = list(range(ranks.shape[0]))
    if shuffle:
        random.shuffle(travellers)
    inData.logger.info('order of travellers {}'.format(travellers))
    for i in travellers:  # iterate over each traveller
        if not i in served: # unless he was not served (other traveller made a decision)
            best = np.nanargmin(ranks[i,:]) # select best remaining option
            selected += [best] # add this ride to solution
            followers = ranks[:,best] # members of selected group
            followers = list(np.argwhere(~np.isnan(followers)).flatten())
            served += followers # to be added as already served
            inData.logger.info("{} selected {} with {}".format(i,best,followers)) # log
            for follower in followers:
                ranks[:, np.argwhere(~np.isnan(ranks[follower,:]))] = np.nan  # remove rides containing traveller i

        else:
            inData.logger.info("{} already served".format(i))
    return selected


def KPIs(inData):
    ret = dict(PassUtility=inData.sblts.schedule.u_pax.sum(),
               VehKm=inData.sblts.schedule.u_veh.sum(),
               PoA_total=inData.sblts.schedule['mean_PoA'].sum(),
               PoA_std=inData.sblts.schedule['mean_PoA'].std(),
               ranking_mean=inData.sblts.schedule['mean_rel_ranking'].mean(),
               ranking_std=inData.sblts.schedule['mean_rel_ranking'].std())
    return pd.Series(ret)


def test_obj_fun(inData, params, obj = 'u_pax', _plot = False):
    # computes ILP with a given obj function and returns KPIs of the solution
    from ExMAS.main import matching
    params.matching_obj = obj
    if "prob" in obj:
        params.minmax = 'max'
    else:
        params.minmax = 'min'
    if obj == 'worst':
        params.minmax = 'max'
        params.matching_obj = 'u_pax'
    inData = matching(inData, params, plot = False)
    if _plot:
        import matplotlib.pyplot as plt
        m_solution = inData.sblts.m.copy()
        fig, ax = plt.subplots()
        for col in m_solution.columns:
            if inData.sblts.rides.loc[col].selected==0:
                m_solution[col] = inData.sblts.m[col]
            else:
                m_solution[col] = inData.sblts.m[col]*5

        ax.imshow(m_solution, cmap='Greys', interpolation = 'Nearest')
        ax.set_ylabel('rides')
        _ = ax.set_xlabel('trips')
        print('grey - feasible, black - selected')
    inData = calc_solution_PoA(inData)

    return KPIs(inData)


def brute(inData, log_step=500000):
    """
    iterates all possible combinations of rides in trip-ride assignment
    selects and outputs those who meet the criteria of assignment (matching)
    each traveller is assigned to exactly one ride
    Extremally computationally expensive and slow"""
    import itertools

    mtx = np.array(inData.sblts.incidence_matrix.values)
    feasible = list()
    solutions = itertools.product([True, False], repeat=mtx.shape[1])
    for i, solution in enumerate(solutions):
        sums = mtx[:, np.array(solution)].sum(axis=1)
        if len(sums[sums != 1]) == 0:
            feasible.append(np.array(solution))
            print(solution)
        if divmod(i, log_step)[1] == 0: # log
            print(i)
    return feasible