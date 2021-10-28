"""
# ExMAS - TRANSITIZE
> Exact Matching of Attractive Shared rides (ExMAS) for system-wide strategic evaluations
> Module to pool requests to stop-to-stop and multi-stop rides, aka TRANSITIZE
---

Run experiments on the parameters grid


----
Rafał Kucharski, TU Delft,GMUM UJ  2021 rafal.kucharski@uj.edu.pl
"""






from ExMAS.transitize.main import pipeline
from dotmap import DotMap
from ExMAS.experiments import *


def space():
    # system settings analysis
    space = DotMap()
    space.nP = [800, 1000, 1200]
    space.s2s_discount = [0.5, 0.66, 0.75]  # discount for stop to stop pooling
    space.multistop_discount = [0.8, 0.85, 0.9]
    return full_space

def exploit_search_space_transitize(one_slice, *args):
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
        _params[key] = val



    stamp['dt'] = str(pd.Timestamp.now())

    filename = "".join([c for c in str(stamp) if c.isalpha() or c.isdigit() or c == ' ']).rstrip().replace(" ", "_")

    pipeline(inData, params, filename)
    print(filename, pd.Timestamp.now(), 'done')
    return 0

def experiment(space=None, config='ExMAS/data/configs/transit_debug.json', workers=-1, replications=1,
               func=exploit_search_space_transitize, logger_level='CRITICAL'):
    """
    Explores the search space `space` starting from base configuration from `config` using `workers` parallel threads`
    :param space:
    :param config:
    :param workers:
    :param replications:
    :return: set of csvs in 'data/results`
    """
    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, '../..'))
    inData = mData
    #os.chdir("C:/Users/sup-rkucharski/PycharmProjects/ExMAS")

    params = ExMAS.utils.get_config(config)
    params.logger_level = logger_level
    params.np = params.nP
    params.t0 = pd.to_datetime('17:00')
    inData = ExMAS.utils.load_G(inData, params, stats=False)
    if space is None:
        search_space = full_space()
    else:
        search_space = space

    scipy.optimize.brute(func=func,
                         ranges=slice_space(search_space, replications=replications),
                         args=(inData, params, search_space),
                         full_output=True,
                         finish=None,
                         workers=workers)
experiment(space=space())

