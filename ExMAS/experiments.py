#!/usr/bin/env python3

"""
Multiple parallel runs to exploit the search space of exepriments
"""


import scipy
import pandas as pd
from dotmap import DotMap
import os

import ExMAS
import ExMAS.utils
from ExMAS.utils import inData as mData
#from corona import *

pd.options.mode.chained_assignment = None


########################
# EXPLOIT SEARCH SPACE #
########################

def full_space():
    # system settings analysis
    full_space = DotMap()
    full_space.nP = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 1800, 2000, 2500, 3000]
    full_space.shared_discount = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]  # , 0.25, 0.3, 0.35]
    full_space.horizon = [60, 120, 300, 600, 1200, -1]  # , -1]
    full_space.profitability = [False]
    full_space.matching_obj = ['u_veh', 'u_pax', 'degree']
    return full_space


def test_space():
    # small search space to see if code works
    full_space = DotMap()
    full_space.nP = [100, 200]  # number of requests per sim time
    return full_space

#########
# UTILS #
#########


def slice_space(s, replications=1):
    # util to feed the np.optimize.brute
    def sliceme(l):
        return slice(0, len(l), 1)

    ret = list()
    sizes = list()
    size = 1
    for key in s.keys():
        ret += [sliceme(s[key])]
        sizes += [len(s[key])]
        size *= sizes[-1]
    if replications > 1:
        sizes += [replications]
        size *= sizes[-1]
        ret += [slice(0, replications, 1)]
    print('Search space to explore of dimensions {} and total size of {}'.format(sizes, size))
    return tuple(ret)


def exploit_search_space(one_slice, *args):
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
    inData.stats = ExMAS.utils.networkstats(inData)
    inData = ExMAS.utils.generate_demand(inData, params)
    t0 = pd.Timestamp.now()
    inData = ExMAS.main(inData, _params, plot=False)

    stamp['dt'] = str(pd.Timestamp.now() - t0)
    stamp['search_space'] = inData.sblts.rides.shape[0]
    ret = pd.concat([inData.sblts.res, pd.Series(stamp)])

    filename = "".join([c for c in str(stamp) if c.isalpha() or c.isdigit() or c == ' ']).rstrip().replace(" ", "_")
    ret.to_frame().to_csv(os.path.join('ExMAS/data/results', filename + '.csv'))
    print(filename, pd.Timestamp.now(), 'done')
    return 0


########
# FULL #
########


def experiment(space=None, config = 'ExMAS/data/configs/default.json', workers=-1, replications = 1):
    """
    Explores the search space `space` starting from base configuration from `config` using `workers` parallel threads`
    :param space:
    :param config:
    :param workers:
    :param replications:
    :return: set of csvs in 'data/results`
    """
    inData = mData
    params = ExMAS.utils.get_config(config)
    params.logger_level = 'CRITICAL'
    params.np = params.nP
    params.max_degree = 8
    params.t0 = pd.to_datetime('15:00')
    inData = ExMAS.utils.load_G(inData, params, stats=False)
    if space is None:
        search_space = full_space()
    else:
        search_space = space

    scipy.optimize.brute(func=exploit_search_space,
                         ranges=slice_space(search_space,replications = replications),
                         args=(inData, params, search_space),
                         full_output=True,
                         finish=None,
                         workers=workers)



if __name__ == "__main__":


    experiment(space=test_space(), workers = 1)
    # experiment(space=big_bullies(), workers=1)
    # experiment(space=big_lambdas(), workers=2)
    # experiment(space=full_space(), workers=-1)
    # experiment(space=profit(), workers=1)
    # experiment(space=beh_res(), workers=4)
    # experiment(space=capacities(), workers=2)

    # experiment(space=beh_res(), workers=2)
