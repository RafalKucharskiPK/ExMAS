import pandas as pd
import random
from dotmap import DotMap
import json
import os, sys
from tqdm import tqdm
import ExMAS.utils
import logging
import warnings
from ExMAS.main_prob import noise_generator as stochastic_noise


def initialise_indata_dotmap():
    inData = DotMap()
    inData['passengers'] = pd.DataFrame(columns=['id', 'pos', 'status'])
    inData.passengers = inData.passengers.set_index('id')
    inData['requests'] = pd.DataFrame(
        columns=['pax', 'origin', 'destination', 'treq', 'tdep', 'ttrav', 'tarr', 'tdrop']).set_index(
        'pax')
    return inData


def get_config(path, root_path=None):
    with open(path) as json_file:
        data = json.load(json_file)
        config = DotMap(data)
    config['t0'] = pd.Timestamp('15:00')

    if root_path is not None:
        config.paths.G = os.path.join(root_path, config.paths.G)  # graphml of a current .city
        config.paths.skim = os.path.join(root_path, config.paths.skim)  # csv with a skim between the nodes of the .city

    return config


def load_nyc_csv(_inData, _params):
    # loads the csv with trip requests
    # filter for the trips within a predefined time window (either exact, or a batch with a given frequency)
    try:
        _params.paths.nyc_requests
    except:
        raise Exception("no nyc trips data path specified")

    trips = pd.read_csv(_params.paths.nyc_requests, index_col=0)  # load csv (prepared in the other notebook)
    trips.pickup_datetime = pd.to_datetime(trips.pickup_datetime)  # convert to times

    # A: Filter for simulation times
    if _params.get('freq', 'False'):  # given frequency (default '10min')
        batches = trips.groupby(pd.Grouper(key='pickup_datetime', freq=_params.get('freq', '10min')))
        if _params.get('batch', 'False'):  # random batch
            batch = list(batches.groups.keys())[_params.batch]  # particular batch
        else:  # random 'freq'-minute batch# i-th batch
            batch = random.choice(list(batches.groups.keys()))
        df = batches.get_group(batch)
    else:  # exact date and sim-time
        early = pd.to_datetime(_params.date) + pd.to_timedelta(_params.t0 + ":00")
        late = pd.to_datetime(_params.date) + pd.to_timedelta(_params.t0 + ":00") + pd.to_timedelta(_params.simTime,
                                                                                                    unit='H')
        df = trips[(trips.pickup_datetime >= early) & (trips.pickup_datetime < late)]

    # B: Populate missing fields

    df['status'] = 0
    df.pos = df['origin']
    _inData.passengers = df
    requests = df
    requests['dist'] = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)
    requests['treq'] = (trips.pickup_datetime - trips.pickup_datetime.min())
    requests['ttrav'] = requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
    # requests.ttrav = pd.to_timedelta(requests.ttrav)
    # if params.get('avg_speed',False):
    #    requests.ttrav = (pd.to_timedelta(requests.ttrav) / _params.avg_speed).dt.floor('1s')
    requests.tarr = [request.pickup_datetime + request.ttrav for _, request in requests.iterrows()]
    requests = requests.sort_values('treq')
    requests['pax_id'] = requests.index.copy()
    _inData.requests = requests
    _inData.passengers.pos = _inData.requests.origin
    _params.nP = _inData.requests.shape[0]
    return _inData


def nyc_csv_prepare_batches(_inData, _params):
    try:
        _params.paths.nyc_requests
    except:
        raise Exception("no nyc trips data path specified")

    trips = pd.read_csv(_params.paths.nyc_requests, index_col=0)  # load csv (prepared in the other notebook)
    trips.pickup_datetime = pd.to_datetime(trips.pickup_datetime)  # convert to times

    batches = trips.groupby(pd.Grouper(key='pickup_datetime', freq=_params.get('freq', '10min')))
    return batches, trips


def nyc_pick_batch(batches, trips, inData, _params, batch_no):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _inData = inData.copy()
        batch = list(batches.groups.keys())[batch_no]
        df = batches.get_group(batch)
        df['status'] = 0
        df.pos = df['origin']
        _inData.passengers = df
        requests = df
        requests['dist'] = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)
        requests['treq'] = (trips.pickup_datetime - trips.pickup_datetime.min())
        requests['ttrav'] = requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
        requests.tarr = [request.pickup_datetime + request.ttrav for _, request in requests.iterrows()]
        requests = requests.sort_values('treq')
        requests['pax_id'] = requests.index.copy()
        _inData.requests = requests
        _inData.passengers.pos = _inData.requests.origin
        _params.nP = _inData.requests.shape[0]
    return _inData


def prepare_batches(number_of_batches, config, filter_function=lambda x: len(x.requests) > 0,
                    output_params=True):
    copy_wd = os.getcwd()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    params = get_config(config)
    logger = embed_logger(params.get("logger_level", None))
    inData = initialise_indata_dotmap()
    inData = ExMAS.utils.load_G(inData, params, stats=True)

    batches, trips = nyc_csv_prepare_batches(inData, params)

    logger.info("Preparing NYC batches \n")
    inDatas = []
    pbar = tqdm(total=number_of_batches)
    counter = 0
    batch_no = 0
    while counter < number_of_batches:
        try:
            temp = nyc_pick_batch(batches, trips, inData, params, batch_no)
            if filter_function(temp):
                inDatas.append(temp)
                pbar.update(1)
                counter += 1
            else:
                logger.debug('Batch no: ', counter, ' skipped due to filter')
                pass
            batch_no += 1
        except:
            logger.info('Impossible to attach batch number: ', batch_no)
            batch_no += 1
            pass
    pbar.close()

    os.chdir(copy_wd)
    logger.info("Batches READY! \n")
    if output_params:
        return inDatas, params
    else:
        return inDatas


def run_exmas_nyc_batches(exmas_algorithm, params, indatas, noise_generator=None,
                          topo_params=DotMap({'variable': None}), replications=1, logger_level=None, stepwise=False):
    logger = embed_logger(logger_level)
    results = []
    settings = []
    params.logger_level = "CRITICAL"
    logger.info(" Calculating ExMAS values \n ")
    for i in range(len(indatas)):
        logger.info(" Batch no. " + str(i))
        step = 0
        noise = stochastic_noise(step=0, noise=None, params=params, batch_length=len(indatas[i].requests),
                                 constrains=params.stepwise_probs.get('constrains', None), type=noise_generator)
        for j in tqdm(range(replications)):
            if topo_params.variable is None:
                try:
                    temp = exmas_algorithm(indatas[i], params, noise, False)
                    results.append(temp.copy())
                    step += 1
                    if stepwise:
                        noise = stochastic_noise(step=step, noise=noise, params=params,
                                                 batch_length=len(indatas[i].requests),
                                                 constrains=None, type=noise_generator)
                    elif not stepwise:
                        noise = stochastic_noise(step=0, noise=None, params=params,
                                                 batch_length=len(indatas[i].requests),
                                                 constrains=params.stepwise_probs.get('constrains', None),
                                                 type=noise_generator)
                    else:
                        raise Exception('Incorrect type of stepwise (should be True/False)')
                    settings.append({'Replication_ID': j, 'Batch': i})
                except:
                    logger.debug('Impossible to attach batch number: ' + str(i))
                    pass
            else:
                for k in range(len(topo_params['values'])):
                    params[topo_params['variable']] = topo_params['values'][k]
                    try:
                        temp = exmas_algorithm(indatas[i], params, None, False)
                        results.append(temp.copy())
                        settings.append({'Replication': j, 'Batch': i, topo_params.variable: topo_params['values'][k]})
                    except:
                        logger.debug('Impossible to attach batch number: ' + str(i))
                        pass

    logger.info("Number of calculated results for batches is: ", len(results))
    logger.info("ExMAS calculated \n")
    return results, settings


def init_log(logger_level, logger=None):
    if logger_level == 'DEBUG':
        level = logging.DEBUG
    elif logger_level == 'WARNING':
        level = logging.WARNING
    elif logger_level == 'CRITICAL':
        level = logging.CRITICAL
    elif logger_level == 'INFO':
        level = logging.INFO
    else:
        raise Exception("Not accepted logger level, please choose: 'DEBUG', 'WARNING', 'CRITICAL', 'INFO'")
    if logger is None:
        logging.basicConfig(stream=sys.stdout, format='%(asctime)s-%(levelname)s-%(message)s',
                            datefmt='%H:%M:%S', level=level)

        logger = logging.getLogger()

        logger.setLevel(level)
        return logging.getLogger(__name__)
    else:
        logger.setLevel(level)
        return logger


def embed_logger(log):
    if log is None:
        return init_log('WARNING')
    elif log in ["DEBUG", "WARNING", "CRITICAL", "INFO"]:
        return init_log(log)
    else:
        raise Exception("Not accepted logger level, please choose: 'DEBUG', 'WARNING', 'CRITICAL', 'INFO'")
