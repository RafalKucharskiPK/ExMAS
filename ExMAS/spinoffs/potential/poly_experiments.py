import pandas as pd
import ExMAS.utils
import ExMAS.main
import ExMAS.experiments
import os
from dotmap import DotMap
import glob

from osmnx.distance import get_nearest_node

def get_centers():
    Centers = pd.DataFrame(data={'name': ['Dam Square', 'Station Zuid', 'Concertgebouw', 'Sloterdijk'],
                                 'x': [4.8909126, 4.871887, 4.8768717, 4.8351158],
                                 'y': [52.373095, 52.338948, 52.3563125, 52.3888349]})
    Centers['node'] = Centers.apply(lambda center: get_nearest_node(inData.G, (center.y, center.x)), axis=1)
    return Centers

def search_space_csvs():
        # system settings analysis
        space = DotMap(_dynamic=False)
        path = "ExMAS/spinoffs/potential/requests"

        space.requests = glob.glob(path + "/requests_*.csv")

        return space

def search_space_requests():
    space = DotMap(_dynamic=False)
    space.nP = [500,1000,2000]
    space.nCenters = [1, 2, 3, 4]
    space.gammdist_shape = [1.5, 2, 2.5, 4, 6]
    space.gamma_imp_shape = [1, 1.15, 2, 3]
    space.replication = list(range(1,11))
    return space


def create_demand(one_slice, *args):
    inData, params, search_space = args  # read static input
    _inData = inData.copy()
    _params = params.copy()
    params.t0 = pd.Timestamp(params.t0)
    stamp = dict()
    stamp['city'] = _params.city

    for i, key in enumerate(search_space.keys()):
        val = search_space[key][int(one_slice[int(i)])]
        stamp[key] = val
        if key == 'nP':
            _params.nP = val
        else:
            _params[key] = val

    inData = ExMAS.utils.synthetic_demand_poly(_inData, _params) # <- MAIN
    replication = _params.get('replication',1)
    inData.requests.to_csv('ExMAS/spinoffs/potential/generated_requests/requests_np-{}_nCenters-{}_gammdistshape-{}_gammaimpshape-{}_repl-{}.csv'.format(_params.nP,
                                                                                                      _params.nCenters,
                                                                                                      _params.gammdist_shape,
                                                                                                      _params.gamma_imp_shape,
                                                                                                              replication))
    return 0



def exploit_csvs(one_slice, *args):
    # function to be used with optimize brute
    inData, params, search_space = args  # read static input
    _inData = inData.copy()
    _params = params.copy()
    stamp = dict()
    stamp['city'] = _params.city
    # parameterize
    for i, key in enumerate(search_space.keys()):
        val = search_space[key][int(one_slice[int(i)])]
        stamp[key] = val
        if key == 'nP':
            _params.nP = val
        else:
            _params[key] = val

    inData.requests = ExMAS.utils.load_requests(_params['requests'])
    t0 = pd.Timestamp.now()
    params.logger_level = 'WARNING'
    inData = ExMAS.main(inData, _params)


    stamp['dt'] = str(pd.Timestamp.now() - t0)
    stamp['search_space'] = inData.sblts.rides.shape[0]
    ret = pd.concat([inData.sblts.res, pd.Series(stamp)])

    filename = _params['requests'].replace('requests_','results_KPI_')
    ret.to_frame().to_csv(filename)
    inData.sblts.requests.to_csv(filename.replace('results_KPI_', 'request_results_'))
    inData.sblts.schedule.to_csv(filename.replace('results_KPI_', 'schedule_results_'))
    print(filename, pd.Timestamp.now(), 'done')
    return 0

cwd = os.getcwd()
os.chdir(os.path.join(cwd,'../../..'))



ExMAS.experiments.experiment(workers  = 1,
                                space = search_space_requests(),
                                config = "ExMAS/data/configs/potential.json",
                                func = create_demand)

# ExMAS.experiments.experiment(workers = 1,
#                              space = search_space_csvs(),
#                              config = "ExMAS/data/configs/potential.json",
#                              func = exploit_csvs,
#                              logger_level= 'WARNING')
