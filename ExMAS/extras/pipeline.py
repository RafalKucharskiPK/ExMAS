from ExMAS.utils import inData as inData
from ExMAS.main import matching, evaluate_shareability
from ExMAS.extras import games, pricings, prunings, timewindow_benchmark
import ExMAS.main
import ExMAS.utils

import pandas as pd
EXPERIMENT_NAME = 'TEST'


def prep(params_path = '../../ExMAS/spinoffs/game/pipe.json'):

    params = ExMAS.utils.get_config(params_path, root_path = '../../')  # load the default
    params.t0 = pd.to_datetime(params.t0)
    params.logger_level = 'WARNING'
    params.matching_obj = 'u_veh'
    params.veh_cost = 1.3 * params.VoT / params.avg_speed  # operating costs per kilometer
    params.fixed_ride_cost = 0.3  # ride fixed costs (per vehicle)
    params.time_cost = params.VoT  # travellers' cost per travel time
    params.wait_cost = params.time_cost * 1.5  # and waiting
    params.sharing_penalty_fixed = 0  # fixed penalty (EUR) per
    params.sharing_penalty_multiplier = 0  # fixed penalty (EUR) per
    params.minmax = 'min'
    params.multi_platform_matching = False
    params.assign_ride_platforms = True

    params.max_detour = 120  # windows
    params.max_delay = 120  # windows

    from ExMAS.utils import inData as inData

    inData = ExMAS.utils.load_G(inData, params, stats=True)  # download the graph

    # initial settings
    params.minmax = 'min'
    params.multi_platform_matching = False
    params.assign_ride_platforms = True
    params.nP = 50
    params.shared_discount = 0.2

    # prepare ExMAS
    inData = ExMAS.utils.generate_demand(inData, params)  # generate requests

    return inData, params

def single_eval(inData, params, pruning_algorithm, PRICING, ALGO,
                minmax = ('min','max'),
                store_res = True):

    #clear
    inData.sblts.rides['pruned'] = True
    inData.sblts.mutually_exclusives = []
    if pruning_algorithm is not None:
        inData = pruning_algorithm(inData, price_column=PRICING)  # apply pruning strategy
    inData.logger.warn('Pruned nRides {}/{}'.format(inData.sblts.rides[inData.sblts.rides.pruned == True].shape[0],
                                             inData.sblts.rides.shape[0]))
    inData.sblts.rides['platform'] = inData.sblts.rides.pruned.apply(
        lambda x: 1 if x else -1)  # use only pruned in the
    inData.sblts.requests['platform'] = 1
    inData.requests['platform'] = inData.requests.apply(lambda x: [1], axis=1)
    if ALGO == 'TSE':
        params.matching_obj = PRICING
        params.minmax = 'min'
        res_name = '{}-{}-{}-{}'.format(PRICING, ALGO, params.matching_obj, params.minmax)  # name of experiment
        inData = evaluate_shareability(inData, params)
        if store_res:

            inData.results.rides[res_name] = inData.sblts.rides.selected.values  # store results (selected rides)
            inData.sblts.rides.selected.name = res_name
            inData.results.rm = inData.results.rm.join(inData.sblts.rides.selected,
                                                       on='ride')  # stor selected rides in the multiindex table
            inData.sblts.res['pricing'] = PRICING
            inData.sblts.res['algo'] = ALGO
            inData.sblts.res['minmax'] = params.minmax
            inData.sblts.res['obj'] = params.matching_obj
            inData.results.KPIs[res_name] = inData.sblts.res
        return inData


    for params.matching_obj in [PRICING]:  # two objective functions
        for params.minmax in minmax:  # best and worst prices of anarchy
            res_name = '{}-{}-{}-{}'.format(PRICING, ALGO, params.matching_obj, params.minmax)  # name of experiment
            inData.logger.warning(res_name)
            inData = matching(inData, params, make_assertion=False)  # < - main matching
            inData = evaluate_shareability(inData, params)
            if store_res:
                inData.results.rides[res_name] = inData.sblts.rides.selected.values  # store results (selected rides)
                inData.sblts.rides.selected.name = res_name
                inData.results.rm = inData.results.rm.join(inData.sblts.rides.selected,
                                                           on='ride')  # stor selected rides in the multiindex table
                inData.sblts.res['pricing'] = PRICING
                inData.sblts.res['algo'] = ALGO
                inData.sblts.res['minmax'] = params.minmax
                inData.sblts.res['obj'] = params.matching_obj
                inData.results.KPIs[res_name] = inData.sblts.res
    return inData


def single_eval_windows(inData, params, pruning_algorithm, PRICING, ALGO):
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
            #inData.results.rides[res_name] = windows.sblts.rides.selected.values  # store results (selected rides)
            #windows.sblts.rides.selected.name = res_name
            #inData.results.rm = inData.results.rm.join(windows.sblts.rides.selected,
            #                                           on='ride')  # stor selected rides in the multiindex table
            windows.sblts.res['pricing'] = PRICING
            windows.sblts.res['algo'] = ALGO
            windows.sblts.res['minmax'] = params.minmax
            windows.sblts.res['obj'] = params.matching_obj
            inData.results.KPIs[res_name] = windows.sblts.res
    return inData

def process_results(inData):
    ret_veh = dict()
    for col in inData.results.rides.columns:
        if '-' in col:
            ret_veh[col] = inData.results.rides[inData.results.rides[col] == True][['u_veh', 'costs_veh']].sum()
    ret_veh = pd.DataFrame(ret_veh).T

    ret_pax = dict()
    for col in inData.results.rm.columns:
        if '-' in col:
            ret_pax[col] = inData.results.rm[inData.results.rm[col] == 1][['ttrav_sh', 'cost_user', 'degree']].sum()

    ret_pax = pd.DataFrame(ret_pax).T
    # ret_pax['min_max'] = ret_pax.apply(lambda x: 'min' if 'min' in x.name else 'max', axis=1)
    ret_pax['ttrav_sh'] = ret_pax['ttrav_sh'] / ret_pax.mean()['ttrav_sh']
    ret_pax['cost_user'] = ret_pax['cost_user'] / ret_pax.mean()['cost_user']
    ret_pax['degree'] = ret_pax['degree'] / 100

    inData.results.KPIs = pd.DataFrame(inData.results.KPIs).T

    inData.results.df = pd.concat([ret_pax, ret_veh, inData.results.KPIs], axis=1)
    inData.results.df.to_csv(EXPERIMENT_NAME+'_KPIs.csv')
    inData.results.rides.to_csv(EXPERIMENT_NAME + '_rides.csv')
    inData.results.rm.to_csv(EXPERIMENT_NAME + '_rm.csv')


    return inData


def pipe():

    inData, params = prep()


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
    ALGO = 'EXMAS'

    inData = single_eval(inData, params, None, PRICING, ALGO)

    params.multi_platform_matching = True
    params.assign_ride_platforms = False


    ALGOS=dict() # algorithms to apply and their names
    ALGOS['TSE'] = prunings.algo_TSE
    ALGOS['TNE'] = prunings.algo_TNE
    ALGOS['HERMETIC'] = prunings.algo_HERMETIC
    ALGOS['RUE'] = prunings.algo_RUE
    ALGOS['RSIE'] = prunings.algo_RSIE

    PRICINGS = dict() # pricings to apply and their names
    PRICINGS['UNIFORM'] = pricings.uniform_split
    PRICINGS['EXTERNALITY'] = pricings.externality_split
    PRICINGS['RESIDUAL'] = pricings.residual_split
    PRICINGS['SUBGROUP'] = pricings.subgroup_split      

    for PRICING, pricing in PRICINGS.items():
        inData = pricing(inData)  # apply pricing strategy
        for ALGO, algorithm in ALGOS.items():
            inData = single_eval(inData, params, algorithm, PRICING, ALGO)

    ALGO = 'WINDOWS'
    PRICING =  'EXMAS'
    inData = single_eval_windows(inData, params, None, PRICING, ALGO)

    inData = process_results(inData)

    return inData

if __name__ == '__main__':
    pipe()
