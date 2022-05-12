"""
# ExMAS
> Exact Matching of Attractive Shared rides (ExMAS) for system-wide strategic evaluations
---

Pruning algorithms to filter feasible shared rides into the one fitting into a give notion of equilibrium
Used in the game-theoretical study of the paper ....

----
Rafał Kucharski, TU Delft, 2020 r.m.kucharski (at) tudelft.nl
"""


import itertools
import pandas as pd
import numpy as np

def algo_EXMAS(inData, price_column='UNIFORM'):
    # prunes the rides to only those whose costs are lower than for single ride for all travellers

    rm = inData.sblts.rides_multi_index  # ride (group) - traveller data
    rides = inData.sblts.rides  # rides data

    rides['pruned_EXMAS'] = True  # this will be used as a filter in pruning
    rm['pruned_EXMAS'] = True  # store data about pruned ride-traveller pairs

    inData.sblts.rides_multi_index = rm  # store back
    inData.sblts.rides = rides  #          tables

    return inData



def algo_TNE(inData, price_column='UNIFORM'):
    # prunes the rides to only those whose costs are lower than for single ride for all travellers

    rm = inData.sblts.rides_multi_index  # ride (group) - traveller data
    rides = inData.sblts.rides  # rides data

    # see the price of single ride for each ride
    rm['price_single'] = rm.apply(
        lambda r: rm[(rm.traveller == r.traveller) & (rm.shared == False)][price_column].max(), axis=1)

    rm['surplus'] = rm.price_single - rm[price_column] # difference between single and shared
    rm['pruned_TNE_user'] = (rm[price_column] < rm.price_single) | (rm.shared == False) # filter to those which are not
    # shared or lower cost than single
    pruned = rm.groupby('ride')['pruned_TNE_user'].min().to_frame('pruned_TNE')  # ride is pruned if all travellers meet above
    pruned['ride'] = pruned.index  # update data
    if 'pruned_TNE' in rm.columns:
        del rm['pruned_TNE']
    rides['pruned_TNE'] = pruned['pruned_TNE']  # this will be used as a filter in pruning
    rm = rm.join(pruned['pruned_TNE'], on='ride')  # store data about pruned ride-traveller pairs

    inData.sblts.rides_multi_index = rm  # store back
    inData.sblts.rides = rides  #          tables

    return inData


def algo_HERMETIC(inData, price_column='UNIFORM'):
    # checks if the groups are hermetic and returns only hermetic ones ('pruned' == True
    rm = inData.sblts.rides_multi_index
    rides = inData.sblts.rides

    def hermetic(ride):
        # checks if ride is hermetic
        G = ride.name
        prices_G = rm.loc[G, :][[price_column]]  # prices of travellers in G
        hermetic = True # assume it is hermetic
        for H in ride.subgroups:  # iterate all subgroups
            # determine whether there exists at least one traveller who wants to use G rather than H
            df = rm.loc[H, :][[price_column]].join(prices_G, lsuffix='_H', rsuffix='_G') # prices in subgroup
            df['surplus'] = df.iloc[:, 0] - df.iloc[:, 1]  # difference in prices
            if df.surplus.max() < 0: # if all are happy in subgroup
                hermetic = False # raise exit flag
                inData.logger.info('non-hermetic group ' + str(G))
                return hermetic # return
        return hermetic

    rides['hermetic'] = rides.apply(hermetic, axis=1)  # check which ones are hermetic
    rides['pruned_HERMETIC'] = rides['hermetic']  # use hermetic as pruned

    inData.sblts.rides_multi_index = rm
    inData.sblts.rides = rides

    return inData


def algo_RUE(inData, price_column='UNIFORM'):
    # determines set of mergeable groups
    rm = inData.sblts.rides_multi_index  # ride (group) - traveller data
    rides = inData.sblts.rides  # rides data

    lsuffix = '_G1G2'  # suffixes for merges
    rsuffix = '_G' # suffixes for merges
    mergeable = list()  # output - used as mutual exclusive constrain in ILP
    indexes_set = rides.indexes_set  # set of travellers for each ride
    for i in rides.index:  # iterate over rides
        ride = rides.loc[i]
        subgroups = ride.subgroups # subgroups
        costs_G = rm.loc[ride.name, :][['traveller', price_column]]  # costs in group G
        for G1, G2 in itertools.combinations(subgroups, 2):  # split group into two subgroups
            if indexes_set[G1].isdisjoint(indexes_set[G2]):  # if disjoint
                if indexes_set[G1].union(indexes_set[G2]) == ride.indexes_set:  # and sum to G
                    df = rm.loc[[G1, G2], :][['traveller', price_column]] # see costs in subgroups
                    df = df.join(costs_G, on='traveller', lsuffix=lsuffix, rsuffix=rsuffix)  # compare with costs in G
                    df['surplus'] = df[price_column + lsuffix] - df[price_column + rsuffix]  # see which is cheaper
                    if df.surplus.min() >= 0 and df.surplus.max() > 0: # if no one is worse and someone is better off
                        mergeable.append([G1, G2]) # add to constrains
                        inData.logger.info('Mergeable groups: {}-{}'.format(G1, G2))
    inData.sblts['mutually_exclusives_RUE'] = mergeable # return

    return inData


def algo_RSIE(inData, price_column='UNIFORM', _print=False):
    # see who wants to switch from one group to another

    rm = inData.sblts.rides_multi_index  # ride (group) - traveller data
    rides = inData.sblts.rides  # rides data
    lsuffix = '_x' # suffixes for merges
    rsuffix = '_y'
    unstables = list()  # output - used as mutual exclusive constrain in ILP
    indexes_set = rides.indexes_set # set of travellers for each ride

    def are_unstable(G1, G2):
        # see if two groups are unstable
        for i in indexes_set[G1]:  # examine each i in G1
            G2s_with_i = G2s.union({i})  # move i to G2
            for r in rides[rides.indexes_set == G2s_with_i].index:  # loop over rides where i joining G2
                if rm.loc[r, i][price_column] < rm.loc[G1, i][price_column]:  # condition 1 (i want to join G1)
                    costs_of_G2_with_i = rm.loc[pd.IndexSlice[r, list(G2s)], :][
                        ['traveller', price_column]]  # costs for travellers in G2 with i
                    compare = pd.merge(costs_of_G2, costs_of_G2_with_i, on=['traveller'])  # compare prices
                    compare['surplus'] = compare[price_column + lsuffix] - compare[
                        price_column + rsuffix]  # see which is cheaper
                    if compare.surplus.min() >= 0:  # if no one is better off
                        if _print:  # debugging only
                            print('Group1:', G1, G1s)
                            print('Group2:', G2, G2s)
                            print('Moving traveller:', i)
                            print('Group2 with i:', r, G2s_with_i)
                            print('Costs for i in G1:', rm.loc[G1, i][price_column])
                            print('Costs for i in G2:', rm.loc[r, i][price_column])
                            print('Costs for G2 without i \n ', costs_of_G2[price_column])
                            print('Costs for G2 with i \n ', costs_of_G2_with_i[price_column])
                        return True
        return False

    for G2 in rides.index:  # loop over ride (better first G2 - faster)
        if G2 % 20 == 0: # print every 20 rides
            inData.logger.warning('Searching unstable pairs {}/{}. {} found so far'.format(G2,
                                                                                           inData.sblts.rides.shape[0],
                                                                                           len(unstables)))
        G2s = indexes_set[G2]  # travellers in G2
        costs_of_G2 = rm.loc[G2, :][['traveller', price_column]]  # costs of group G2 before joining
        for G1 in rides.index:  # pairs
            G1s = indexes_set[G1]  # travellers in G1
            if indexes_set[G1].isdisjoint(indexes_set[G2]):  # if rides are disjoint
                if are_unstable(G1, G2):
                    unstables.append([G1, G2])
    inData.sblts['mutually_exclusives_RSIE'] = unstables

    return inData


def algo_TSE(inData, price_column='UNIFORM'):
    # heuristical algorithm to determine one TSE equilibrium
    # there is no ILP in this, this is already pruning and assignment

    rm = inData.sblts.rides_multi_index  # ride (group) - traveller data
    rides = inData.sblts.rides  # rides data
    selected = list()  # list of selected rides
    rides['TSE_obj_fun'] = rm.groupby('ride')['cost_efficiency'].max()  # see efficiencies (criteria for assignment)
    rides = rides[rides.pruned]  # only pruned rides

    assigned = list() # control if everyone is assigned
    while len(assigned) < inData.sblts.requests.shape[0]: # until everyone is assigned
        best = np.nanargmin(rides['TSE_obj_fun'])  # select best remaining option
        selected += [best]  # add this ride to solution
        followers = rm.loc[best, :].index # see who travelles with this group (ride)
        rides_to_remove = rm.loc[pd.IndexSlice[:, followers], :].ride.unique() # remove since which are not feasible,
        # at least one of they travellers is already assigned
        rides['TSE_obj_fun'] = rides.apply(lambda x: np.inf if x.name in rides_to_remove else x.TSE_obj_fun, axis=1)
        # filter assignable to remaining ones
        assigned = assigned + list(followers.values)  # update list of assigned

    inData.sblts.rides.selected = inData.sblts.rides.apply(lambda x: 1 if x.name in selected else 0, axis=1) # output
    # mimic of ILP
    inData.sblts.schedule = inData.sblts.rides[inData.sblts.rides.selected == 1].copy()

    return inData

def determine_prunings(inData, ALGOS):
    # filter rides according to given prunings and mutually exclusives
    rides = inData.sblts.rides  # rides data

    # clear
    rides['pruned'] = True
    mutually_exclusives = []

    def lambda_prune(row):
        pruned  = True
        if 'EXMAS' in ALGOS:
            pruned = pruned and row.pruned_EXMAS
        if 'TNE' in ALGOS:
            pruned = pruned and row.pruned_TNE
        if 'HERMETIC' in ALGOS:
            pruned = pruned and row.pruned_HERMETIC
        return pruned

    rides['pruned'] = rides.apply(lambda x: lambda_prune(x), axis = 1)

    if 'RUE' in ALGOS:
        mutually_exclusives += inData.sblts.mutually_exclusives_RUE
    if 'RSIE' in ALGOS:
        mutually_exclusives += inData.sblts.mutually_exclusives_RSIE

    inData.sblts.rides = rides
    inData.sblts.mutually_exclusives = mutually_exclusives

    inData.logger.warn('Prunings:  {}'.format(ALGOS))
    inData.logger.warn('Pruned nRides {}/{}'.format(inData.sblts.rides[inData.sblts.rides.pruned == True].shape[0],
                                                    inData.sblts.rides.shape[0]))
    inData.logger.warn('Mutually exclusives {}'.format(len(inData.sblts.mutually_exclusives)))

    return inData









