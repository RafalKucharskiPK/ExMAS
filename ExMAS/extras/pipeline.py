from ExMAS.utils import inData as inData
from ExMAS.main import matching, evaluate_shareability
from ExMAS.extras import games, pricings, prunings, timewindow_benchmark
import ExMAS.main
import ExMAS.utils

import pandas as pd
EXPERIMENT_NAME = 'many'


def prep(params_path='../../ExMAS/spinoffs/game/pipe.json'):

    params = ExMAS.utils.get_config(params_path, root_path = '../../')  # load the default
    params.t0 = pd.to_datetime(params.t0)
    params.logger_level = 'WARNING'
    params.matching_obj = 'u_veh'

    # parameterization
    params.veh_cost = 1.3 * params.VoT / params.avg_speed  # operating costs per kilometer
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
    params.simTime = 0.25
    params.shared_discount = 0.25

    # prepare ExMAS
    inData = ExMAS.utils.generate_demand(inData, params)  # generate requests

    return inData, params


def single_eval(inData, params, MATCHING_OBJS, PRUNINGS, PRICING,
                minmax = ('min','max'),
                store_res = True):

    inData = prunings.determine_prunings(inData, PRUNINGS) # set pruned to boolean flag for matching

    inData.sblts.rides['platform'] = inData.sblts.rides.pruned.apply(
        lambda x: 1 if x else -1)  # use only pruned rides in the matching
    inData.sblts.requests['platform'] = 1
    inData.requests['platform'] = inData.requests.apply(lambda x: [1], axis=1)

    for params.matching_obj in MATCHING_OBJS:  # for each objective function
        for params.minmax in minmax:  # best and worst prices of anarchy
            res_name = 'Scenario-{}_Pricing-{}_Objective-{}_Pruning-{}_minmax-{}'.format(EXPERIMENT_NAME,
                                                                                                      PRICING,
                                                                                                      MATCHING_OBJS,
                                                                                                      PRUNINGS,
                                                                                                      params.minmax)  # name of experiment
            inData.logger.warning(res_name)
            if 'TSE' not in PRUNINGS:
                inData = matching(inData, params, make_assertion=False)  # < - main matching
            else:
                inData = prunings.algo_TSE(inData, params.matching_obj)  # here we do not do ILP, but heuristical algorithm

            inData = evaluate_shareability(inData, params)

            if store_res:
                inData.results.rides[res_name] = inData.sblts.rides.selected.values  # store results (selected rides)
                inData.sblts.rides.selected.name = res_name
                inData.results.rm = inData.results.rm.join(inData.sblts.rides.selected,
                                                           on='ride')  # store selected rides in the multiindex table
                inData.sblts.res['pricing'] =str(MATCHING_OBJS)  # columns for KPIs table
                inData.sblts.res['algo'] = str(PRUNINGS)
                inData.sblts.res['minmax'] = params.minmax
                inData.sblts.res['obj'] = params.matching_obj
                inData.results.KPIs[res_name] = inData.sblts.res  # stack columns with results
    return inData


def single_eval_windows(inData, params, pruning_algorithm, PRICING, ALGO):  # evaluate windows-based approach
    # this has to be called last, since it screws the inData.sblts.rides
    params.multi_platform_matching = False
    params.assign_ride_platforms = True

    #clear
    inData.sblts.rides['pruned'] = True
    inData.sblts.mutually_exclusives = []
    params.matching_obj = 'u_veh'
    windows = timewindow_benchmark.ExMAS_windows(inData, params)

    for params.matching_obj in ['u_veh']:  # two objective functions
        for params.minmax in ['min', 'max']:  # best and worst prices of anarchy
            res_name = '{}-{}-{}-{}'.format(PRICING, ALGO, params.matching_obj, params.minmax)  # name of experiment
            inData.logger.warning(res_name)
            windows = matching(windows, params, make_assertion=False)  # < - main matching
            windows = evaluate_shareability(windows, params)
            windows.sblts.res['pricing'] = PRICING
            windows.sblts.res['algo'] = ALGO
            windows.sblts.res['minmax'] = params.minmax
            windows.sblts.res['obj'] = params.matching_obj
            inData.results.KPIs[res_name] = windows.sblts.res
    return inData


def process_results(inData):
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
    # ret_pax['min_max'] = ret_pax.apply(lambda x: 'min' if 'min' in x.name else 'max', axis=1)
    #ret_pax['ttrav_sh'] = ret_pax['ttrav_sh'] / ret_pax.mean()['ttrav_sh']
    #ret_pax['cost_user'] = ret_pax['cost_user'] / ret_pax.mean()['cost_user']
    #ret_pax['degree'] = ret_pax['degree'] / 100

    inData.results.KPIs = pd.DataFrame(inData.results.KPIs).T

    inData.results.KPIs = pd.concat([ret_pax, ret_veh, inData.results.KPIs], axis=1)
    inData.results.KPIs.to_csv(EXPERIMENT_NAME+'_KPIs.csv')
    inData.results.rides.to_csv(EXPERIMENT_NAME + '_rides.csv')
    inData.results.rm.to_csv(EXPERIMENT_NAME + '_rm.csv')

    return inData


def pipe():

    inData, params = prep()  # load params, load graph, create demand

    # clear
    inData.sblts.mutually_exclusives = []
    inData.sblts.rides['pruned'] = True
    inData = ExMAS.main(inData, params, plot=False)  # create feasible groups

    inData = games.prepare_PoA(inData)  # prepare data structures
    inData = pricings.update_costs(inData, params)  # determine costs per group and per traveller

    inData.results.rides = inData.sblts.rides.copy()  # copy tables to collect results
    inData.results.rm = inData.sblts.rides_multi_index.copy()
    inData.results.KPIs = dict()

    PRICING = 'u_veh'  # start with basic ExMAS
    PRUNING = 'EXMAS'
    inData = single_eval(inData, params,
                         MATCHING_OBJS=['u_veh'],  # this can be more
                         PRUNINGS=[],  # and this can be more
                         PRICING='EXMAS')


    params.multi_platform_matching = True
    params.assign_ride_platforms = False

    PRUNINGS=dict() # algorithms to apply and their names
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

    for PRICING, pricing in PRICINGS.items():
        inData = pricing(inData)  # apply pricing strategy
        for PRUNING, pruning in PRUNINGS.items():
            inData = pruning(inData, price_column=PRICING)  # apply pruning strategies for a given pricing strategy
        for PRUNING, pruning in PRUNINGS.items():  # perform assignment for single prunings
            inData = single_eval(inData, params,
                                 MATCHING_OBJS = ['total_group_cost'],  # this can be more
                                 PRUNINGS = [PRUNING],  # and this can be more
                                 PRICING = PRICING,  # this is taken from first level loop
                                 minmax = ('min','max'))  # direction BPoA, WPoA

    PRUNING = 'WINDOWS'
    PRICING =  'EXMAS'
    inData = single_eval_windows(inData, params, None, PRICING, PRUNING)

    inData = process_results(inData)

    return inData


if __name__ == '__main__':
    pipe()
