"""
# ExMAS
> Exact Matching of Attractive Shared rides (ExMAS) for system-wide strategic evaluations
---

Pricing algorothms splitting the total ride costs among travellers.
Used in the game-theoretical study of the paper ....

----
Rafał Kucharski, TU Delft, 2020 r.m.kucharski (at) tudelft.nl
"""

import numpy as np

def update_costs(inData, params):
    '''
    Set a costs attributes for the rides
    :param inData: there are two tables, one is inData.sblts.rides taken directly from ExMAS,
    another it rides_multi_index, where data about each ride-traveller information is stored.
    It is generated in the games.PreparePoA
    :param params:
    :return:
    '''

    rm = inData.sblts.rides_multi_index  # ride (group) - traveller data
    rides = inData.sblts.rides  # rides data

    rm['distance'] = rm.ride_time*params.avg_speed  # update distances from travel times

    rm['cost_veh'] = params.veh_cost*rm['distance'] + params.fixed_ride_cost  # vehicle running costs

    # formula for the user costs (disutility)
    rm['cost_user'] = params.time_cost * rm.ttrav_sh + \
                         params.wait_cost * abs(rm.delay) + \
                         params.sharing_penalty_fixed * rm.shared + \
                         params.sharing_penalty_multiplier * params.time_cost * rm.shared * rm.ttrav_sh

    rides['costs_user'] = rm.groupby('ride').sum()['cost_user']  # sum user costs for a ride
    rides['costs_veh'] = rm.groupby('ride').max()['cost_veh']  # assign vehicle costs for rides
    rides['costs_total'] = rides['costs_user'] + rides['costs_veh']  # total ride costs (vehicle + all users)

    rm['total_group_cost'] = rm.apply(lambda r: rm.loc[r.ride, :].cost_user.sum() + rm.loc[r.ride, :].cost_veh.max(),
                                      axis=1)  # assign  total ride costs in rm table
    rides['total_group_cost'] = rides['costs_total'] # seems repetition of costs total
    rides['cost_efficiency'] = rides['total_group_cost'] / rides.degree  # total cost per rider

    # cost of a single ride for a user
    rm['cost_single'] = rm.apply(
        lambda r: rm[(rm.traveller == r.traveller) & (rm.shared == False)]['total_group_cost'].max(), axis=1)


    rm['total_singles'] = rm.apply(lambda r: rm.loc[r.ride, :].cost_single.sum(), axis=1) # costs of single rides per
    # group
    rides['total_singles'] = rm.groupby('ride').sum()['cost_single'] # total single ride costs per ride
    rides['residual'] = rides['costs_total'] - rides['total_singles'] # residual, for some pricing algos
    rm['residual_user'] = rm.apply(lambda r: rides.loc[r.ride].residual, axis=1)

    inData.sblts.rides_multi_index = rm  # store back results
    inData.sblts.rides = rides

    return inData

def uniform_split(inData):
    # splits the costs equally among travellers

    rm = inData.sblts.rides_multi_index
    rides = inData.sblts.rides

    # total costs are divided per all travellers
    rm['UNIFORM'] = rm.apply(lambda r: rides.loc[r.ride].total_group_cost/r.degree, axis=1)  # this is used for pruning
    rides['UNIFORM'] = rides.total_group_cost # this is objective fun of matching

    rm['desired_{}'.format('UNIFORM')] = rm.apply(lambda r: rm[rm.traveller == r.traveller].UNIFORM.min(), axis=1)

    inData.sblts.rides_multi_index = rm
    inData.sblts.rides = rides

    return inData


def externality_split(inData):
    # costs based on externalities (how much would group pay without me)
    # eq. 25

    def get_externality(r):
        # get minimal cost among subgroups without each traveller
        ride, traveller = r.ride, r.traveller

        if r.degree > 1:  # if group is shared
            subgroups = rides.loc[rides.loc[ride].subgroups].indexes_set  # all subgroups
            price = r.total_group_cost  # start with the supergroup cost (to be overwritten)
            for subgroup, subgroup_indexes in subgroups.iteritems():  # iterate all subgroup
                if len(subgroup_indexes) == (r.degree - 1):  # see if subgroup has degree - 1 travellers
                    if traveller not in subgroup_indexes:  # see if traveller not in this group

                        price = min(price, rides.loc[subgroup].total_group_cost) # if so, update the price
                        # (if lower than current)
            price = r.total_group_cost - price  # eq. 25
        else:
            price = r.total_group_cost - 0  # for empty groups the price is null
        return price

    rm = inData.sblts.rides_multi_index
    rides = inData.sblts.rides

    rm['EXTERNALITY'] = rm.apply(get_externality, axis=1)  # this will be used in pruning
    rides['EXTERNALITY'] = rides.total_group_cost  # this will be objective fun in matching

    rm['desired_{}'.format('EXTERNALITY')] = rm.apply(lambda r: rm[rm.traveller == r.traveller].EXTERNALITY.min(), axis=1)

    inData.sblts.rides_multi_index = rm
    inData.sblts.rides = rides

    return inData

def residual_split(inData):
    # it splits the difference as compared to single rides proportionally to user costs contributions
    # eq. 29, 30

    rm = inData.sblts.rides_multi_index
    rides = inData.sblts.rides

    # it uses residual of rides as computed in update_costs
    rm['RESIDUAL'] = rm.apply(lambda x: x.residual_user * x.cost_single / x.total_singles +
                                        x.cost_single,axis = 1)
    # and splits it proportionally to costs_users
    rides['RESIDUAL'] = rides['residual']  # this will be objective fun in matching

    # to do this shall be p_i + c_i(single)
    rm['desired_{}'.format('RESIDUAL')] = rm.apply(lambda r: rm[rm.traveller == r.traveller].RESIDUAL.min(),
                                                      axis=1)

    inData.sblts.rides_multi_index = rm
    inData.sblts.rides = rides

    return inData




def subgroup_split(inData):
    # determines groups of lower costs and assigns costs based on alternatives
    # algorithm 7

    def get_subgroup_price(r):
        # assigns traveller prices by their best alternatives
        indexes_set = r.indexes_set  # travellers of this group
        subgroups = r.subgroups # subgroups of this group
        subgroup_indexes = rides.loc[subgroups][['indexes_set']]  # travellers indexes in the subgroups

        prices = dict()  # return dict to populat
        while len(indexes_set) > 0:  # until everyone is assigned
            effs = rides.loc[subgroups].cost_efficiency  # see the efficiencies of remaining subgroups
            J, z  = effs.idxmin(), effs.min()  # pick up the subgroup of greatest efficiency and its index
            for i in rides.loc[J].indexes_set:
                prices[i] = z # assign the prices

            indexes_set = indexes_set - rides.loc[J].indexes_set  # remove those from the best group
            subgroup_indexes = rides.loc[subgroups][['indexes_set']]
            subgroup_indexes['f'] = subgroup_indexes.apply(
                lambda x: len(rides.loc[J].indexes_set.intersection(x.indexes_set)) == 0, axis=1) # update which
            # subgroups remain assignable
            subgroups = subgroup_indexes[subgroup_indexes.f].index  # filter to those not assigned
            # loop and assign the ones who are not assigned left

        return prices

    rm = inData.sblts.rides_multi_index
    rides = inData.sblts.rides

    prices = dict() # prices to update
    for i, r in rides.iterrows():
        prices.update(get_subgroup_price(r)) # for each ride see price for travellers

    rm['SUBGROUP'] = rm.apply(lambda x: prices[x.traveller], axis = 1) # this is used for pruning
    rides['SUBGROUP'] = rm.groupby('ride').sum()['SUBGROUP']  # this is objective fun of matching

    inData.sblts.rides_multi_index = rm
    inData.sblts.rides = rides

    return inData











