import pandas as pd
import ExMAS.utils
import ExMAS.main
import ExMAS.experiments
import os
from dotmap import DotMap
import glob


def search_space_of_csv():
        # system settings analysis
        space = DotMap()
        path = "/Users/rkucharski/Documents/GitHub/ExMAS/ExMAS/spinoffs/potential/requests"

        space.requests = glob.glob(path + "/*.csv")

        return space


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
    #inData.stats = ExMAS.utils.networkstats(inData)
    inData.requests = ExMAS.utils.load_requests(params['requests'])
    #inData = ExMAS.utils.generate_demand(inData, params)
    t0 = pd.Timestamp.now()
    inData = ExMAS.main(inData, _params, plot=False)

    stamp['dt'] = str(pd.Timestamp.now() - t0)
    stamp['search_space'] = inData.sblts.rides.shape[0]
    ret = pd.concat([inData.sblts.res, pd.Series(stamp)])

    filename = "".join([c for c in str(stamp) if c.isalpha() or c.isdigit() or c == ' ']).rstrip().replace(" ", "_")
    ret.to_frame().to_csv(os.path.join('ExMAS/data/results', filename + '.csv'))
    print(filename, pd.Timestamp.now(), 'done')
    return 0

cwd = os.getcwd()
os.chdir(os.path.join(cwd,'../../..'))

ExMAS.experiments.experiment(space = search_space_of_csv(), config = "ExMAS/data/configs/potential.json", func = exploit_csvs)
