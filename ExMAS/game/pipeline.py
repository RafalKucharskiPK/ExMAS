import sys
sys.path.append(".")

from ExMAS.utils import inData as inData
from ExMAS.main import matching, evaluate_shareability
from ExMAS.game import games, pricings, prunings, timewindow_benchmark
import ExMAS.main
import ExMAS.utils

import pandas as pd


EXPERIMENT_NAME = 'test'


def prep(params_path='ExMAS/spinoffs/game/ams.json'):


    params = ExMAS.utils.get_config(params_path, root_path=None)  # load the default
    params.t0 = pd.to_datetime(params.t0)
    params.logger_level = 'WARNING'
    params.matching_obj = 'u_veh'

    # parameterization
    params.veh_cost = 2.3 * params.VoT / params.avg_speed  # operating costs per kilometer
    params.fixed_ride_cost = 1  # ride fixed costs (per vehicle)
    params.time_cost = params.VoT  # travellers' cost per travel time
    params.wait_cost = params.time_cost * 1.5  # and waiting
    params.sharing_penalty_fixed = 0  # fixed penalty (EUR) per
    params.sharing_penalty_multiplier = 0  # fixed penalty (EUR) per

    params.max_detour = 120  # windows
    params.max_delay = 120  # windows

    from ExMAS.utils import inData as inData

    inData = ExMAS.utils.load_G(inData, params, stats=True)  # download the graph

    # initial settings
    params.minmax = 'min'
    params.multi_platform_matching = False
    params.assign_ride_platforms = True
    params.nP = 400
    params.simTime = 0.15
    params.shared_discount = 0.2

    # prepare ExMAS
    inData = ExMAS.utils.generate_demand(inData, params)  # generate requests

    return inData, params


def single_eval(inData, params, EXPERIMENT_NAME, MATCHING_OBJS, PRUNINGS, PRICING,
                minmax=('min', 'max'),
                store_res=True):
    inData = prunings.determine_prunings(inData, PRUNINGS)  # set pruned to boolean flag for matching
    inData.results.rides['pruned_Pricing-{}_Pruning-{}'.format(PRICING, PRUNINGS)] = inData.sblts.rides['pruned']

    inData.sblts.rides['platform'] = inData.sblts.rides.pruned.apply(
        lambda x: 1 if x else -1)  # use only pruned rides in the matching
    inData.sblts.requests['platform'] = 1
    inData.requests['platform'] = inData.requests.apply(lambda x: [1], axis=1)

    for params.matching_obj in MATCHING_OBJS:  # for each objective function
        for params.minmax in minmax:  # best and worst prices of anarchy
            res_name = 'Experiment-{}_Pricing-{}_Objective-{}_Pruning-{}_minmax-{}'.format(EXPERIMENT_NAME,
                                                                                           PRICING,
                                                                                           MATCHING_OBJS,
                                                                                           PRUNINGS,
                                                                                           params.minmax)  # name of experiment
            inData.logger.warning(res_name)
            if 'TSE' not in PRUNINGS:
                inData = matching(inData, params, make_assertion=False)  # < - main matching
            else:
                inData = prunings.algo_TSE(inData,
                                           params.matching_obj)  # here we do not do ILP, but heuristical algorithm

            inData = evaluate_shareability(inData, params)

            if store_res:
                inData.results.rides[res_name] = inData.sblts.rides.selected.values  # store results (selected rides)

                inData.sblts.rides.selected.name = res_name
                inData.results.rm = inData.results.rm.join(inData.sblts.rides.selected,
                                                           on='ride')  # store selected rides in the multiindex table

                rm = inData.results.rm
                pruneds = inData.sblts.rides[inData.sblts.rides.pruned==True].index
                rm['bestpossible_{}'.format(PRICING)] = rm.apply(lambda r: rm.loc[pruneds,:][(rm.traveller == r.traveller)][PRICING].min(),
                                                               axis=1)
                selecteds = rm.loc[inData.sblts.rides[inData.sblts.rides.selected==True].index]
                inData.sblts.res['eq13'] = selecteds[selecteds['bestpossible_{}'.format(PRICING)] == selecteds[PRICING]].shape[0]/selecteds.shape[0]


                inData.sblts.res['pricing'] = MATCHING_OBJS[0]  # columns for KPIs table
                inData.sblts.res['algo'] = PRUNINGS[0]
                inData.sblts.res['experiment'] = EXPERIMENT_NAME
                inData.sblts.res['minmax'] = params.minmax
                inData.sblts.res['obj'] = params.matching_obj
                inData.results.KPIs[res_name] = inData.sblts.res  # stack columns with results
    return inData


def single_eval_windows(inData, params, EXPERIMENT_NAME, MATCHING_OBJS, PRUNINGS, PRICING,
                minmax=('min', 'max'),
                store_res=True):  # evaluate windows-based approach
    # this has to be called last, since it screws the inData.sblts.rides

    params.multi_platform_matching = False
    params.assign_ride_platforms = True

    # clear
    inData.sblts.rides['pruned'] = True
    inData.sblts.mutually_exclusives = []
    params.matching_obj = 'u_veh'
    windows = timewindow_benchmark.ExMAS_windows(inData, params)

    # compute degrees
    inData.sblts.rides['degree'] = inData.sblts.rides.apply(lambda x: len(x.indexes), axis=1)

    # delays
    inData.sblts.rides['treqs'] = inData.sblts.rides.apply(lambda x: inData.sblts.requests.loc[x.indexes].treq.values,
                                                           axis=1)

    def calc_deps(r):
        deps = [r.times[0]]
        for d in r.times[1:r.degree]:
            deps.append(deps[-1] + d)  # departure times
        t = windows.sblts.requests
        return deps

    windows.sblts.rides['deps'] = windows.sblts.rides.apply(calc_deps, axis=1)

    windows.sblts.rides['delays'] = windows.sblts.rides['deps'] - windows.sblts.rides['treqs']

    windows.sblts.rides['ttravs'] = windows.sblts.rides.apply(lambda r: [sum(r.times[i + 1:r.indexes_orig.index(r.indexes[i]) + r.degree+ 1 + r.indexes_dest.index(r.indexes[i])]) for i in range(r.degree)], axis = 1)


    multis = list()
    for i, ride in windows.sblts.rides.iterrows():
        for t in ride.indexes:
            multis.append([ride.name, t])
    multis = pd.DataFrame(index=pd.MultiIndex.from_tuples(multis))



    multis['ride'] = multis.index.get_level_values(0)
    multis['traveller'] = multis.index.get_level_values(1)
    multis = multis.join(windows.sblts.requests[['treq', 'dist', 'ttrav']], on='traveller')
    multis = multis.join(windows.sblts.rides[['u_veh', 'u_paxes', 'degree', 'indexes', 'ttravs', 'delays']], on='ride')
    multis['order'] = multis.apply(lambda r: r.indexes.index(r.traveller), axis=1)
    multis['ttrav_sh'] = multis.apply(lambda r: r.ttravs[r.order], axis=1)
    multis['delay'] = multis.apply(lambda r: r.delays[r.order], axis=1)
    # multis['u'] = multis.apply(lambda r: r.u_paxes[r.order], axis=1)
    multis['shared'] = multis.degree > 1
    multis['ride_time'] = multis.u_veh
    multis = multis[
        ['ride', 'traveller', 'shared', 'degree', 'treq', 'ride_time', 'dist', 'ttrav', 'ttrav_sh', 'delay']]
    windows.sblts.rides_multi_index = multis


    windows = pricings.update_costs(windows, params)

    for params.matching_obj in ['u_veh']:  # two objective functions
        for params.minmax in ['min', 'max']:  # best and worst prices of anarchy
            res_name = 'Experiment-{}_Pricing-{}_Objective-{}_Pruning-{}_minmax-{}'.format(EXPERIMENT_NAME,
                                                                                           PRICING,
                                                                                           MATCHING_OBJS,
                                                                                           PRUNINGS,
                                                                                           params.minmax)  # name of experiment
            inData.logger.warning(res_name)
            windows = matching(windows, params, make_assertion=False)  # < - main matching
            windows = evaluate_shareability(windows, params)
            windows.sblts.res['pricing'] = MATCHING_OBJS[0]
            windows.sblts.res['algo'] = PRUNINGS[0]
            windows.sblts.res['minmax'] = params.minmax
            windows.sblts.res['obj'] = params.matching_obj
            inData.results.KPIs[res_name] = windows.sblts.res
    return inData


def process_results(inData, EXPERIMENT_NAME):
    # called at the end of pipeline to wrap-up the results
    ret_veh = dict()
    for col in inData.results.rides.columns:
        if '-' in col:  # cols with results
            ret_veh[col] = inData.results.rides[inData.results.rides[col] == True][['u_veh', 'costs_veh']].sum()
    ret_veh = pd.DataFrame(ret_veh).T

    ret_pax = dict()
    for col in inData.results.rm.columns:
        if '-' in col:  # cols with results
            ret_pax[col] = inData.results.rm[inData.results.rm[col] == 1][['ttrav_sh', 'cost_user', 'degree']].sum()

    ret_pax = pd.DataFrame(ret_pax).T

    inData.results.KPIs = pd.DataFrame(inData.results.KPIs).T

    inData.results.KPIs = pd.concat([ret_pax, ret_veh, inData.results.KPIs], axis=1)
    inData.results.KPIs.to_csv(EXPERIMENT_NAME + '_KPIs.csv')
    inData.results.rides.to_csv(EXPERIMENT_NAME + '_rides.csv')
    inData.results.rm.to_csv(EXPERIMENT_NAME + '_rm.csv')

    return inData


def pipe(EXPERIMENT_NAME):
    inData, params = prep()  # load params, load graph, create demand

    PRUNINGS = dict()  # algorithms to apply and their names
    PRUNINGS['EXMAS'] = prunings.algo_EXMAS
    PRUNINGS['TNE'] = prunings.algo_TNE
    PRUNINGS['HERMETIC'] = prunings.algo_HERMETIC
    PRUNINGS['RUE'] = prunings.algo_RUE
    PRUNINGS['RSIE'] = prunings.algo_RSIE
    PRUNINGS['TSE'] = prunings.algo_TSE

    PRICINGS = dict()  # pricings to apply and their names
    PRICINGS['UNIFORM'] = pricings.uniform_split
    PRICINGS['EXTERNALITY'] = pricings.externality_split
    PRICINGS['RESIDUAL'] = pricings.residual_split
    PRICINGS['SUBGROUP'] = pricings.subgroup_split

    # clear
    inData.sblts.mutually_exclusives = []
    inData.sblts.rides['pruned'] = True
    inData = ExMAS.main(inData, params, plot=False)  # create feasible groups

    inData = games.prepare_PoA(inData)  # prepare data structures
    inData = pricings.update_costs(inData, params)  # determine costs per group and per traveller
    for PRICING, pricing in PRICINGS.items():
        inData = pricing(inData)  # apply pricing strategy

    inData.results.rides = inData.sblts.rides.copy()  # copy tables to collect results
    inData.results.rm = inData.sblts.rides_multi_index.copy()
    inData.results.KPIs = dict()


    params.multi_platform_matching = True
    params.assign_ride_platforms = False

    for PRICING, pricing in PRICINGS.items():
        inData = pricing(inData)  # apply pricing strategy
        for PRUNING, pruning in PRUNINGS.items():
            inData = pruning(inData, price_column=PRICING)  # apply pruning strategies for a given pricing strategy
        for PRUNING, pruning in PRUNINGS.items():  # perform assignment for single prunings
            inData = single_eval(inData, params,
                                 EXPERIMENT_NAME=EXPERIMENT_NAME,
                                 MATCHING_OBJS=['total_group_cost'],  # this can be more
                                 PRUNINGS=[PRUNING],  # and this can be more
                                 PRICING=PRICING,  # this is taken from first level loop
                                 minmax=('min', 'max'))  # direction BPoA, WPoA


    inData = process_results(inData, EXPERIMENT_NAME)

    return inData


if __name__ == '__main__':
    pipe(EXPERIMENT_NAME)
