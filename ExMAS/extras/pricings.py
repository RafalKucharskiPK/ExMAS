

def update_costs(inData, params):

    rm = inData.sblts.rides_multi_index

    rm['distance'] = rm.ride_time*params.avg_speed

    rm['cost_veh'] = params.veh_cost*rm['distance'] + params.fixed_ride_cost

    rm['cost_user'] = params.time_cost * rm.ttrav_sh + \
                         params.wait_cost * abs(rm.delay) + \
                         params.sharing_penalty_fixed * rm.shared + \
                         params.sharing_penalty_multiplier * rm.shared * rm.ttrav_sh

    inData.sblts.rides['costs_user'] = rm.groupby('ride').sum()['cost_user']
    inData.sblts.rides['costs_veh'] = rm.groupby('ride').max()['cost_veh']

    rm['cost_single'] = rm.apply(
        lambda r: rm[(rm.traveller == r.traveller) & (rm.shared == False)]['cost_user'].max(), axis=1)

    inData.sblts.rides_multi_index = rm

    return inData

def uniform_split(inData):

    rm = inData.sblts.rides_multi_index
    rides = inData.sblts.rides

    rm['uniform_split'] = rm.apply(lambda r: (rm.loc[r.ride,:].cost_user.sum()+r.cost_veh)/r.degree, axis=1)

    inData.sblts.rides_multi_index = rm
    inData.sblts.rides = rides

    return inData








