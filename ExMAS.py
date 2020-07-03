"""
# ExMAS
> Exact Matching of Attractive Shared rides (ExMAS) for system-wide strategic evaluations
---

![MAP](/data/map.png)

ExMAS allows you to match trips into attractive shared rides.

For a given:
* network (`osmnx` graph),
* demand (microscopic set of trips $q_i = (o_i, d_i, t_i)$)
* parameters (behavioural, like _willingness-to-share_ and system like _discount_ for shared rides)

It computes:
* optimal set of shared rides (results of bipartite matching with a given objective)
* shareability graph
* set of all feasible rides
* KPIs of sharing
* trip sharing attributes

ExMAS is a `python` based open-source package applicable to general networks and demand patterns.

If you find this code useful in your research, please consider citing:

>_Kucharski R. , Cats. O 2020.
Exact matching of attractive shared rides (ExMAS) for system-wide strategic evaluations,
Transportation Research Part B 139 (2020) 285-310_
https://doi.org/10.1016/j.trb.2020.06.006


[Quickstart tutorial](https://github.com/RafalKucharskiPK/ExMAS/blob/master/notebooks/ExMAS.ipynb)

----
Rafał Kucharski, TU Delft, 2020 r.m.kucharski (at) tudelft.nl
"""



__author__ = "Rafal Kucharski"
__copyright__ = "Copyright 2020, TU Delft"
__credits__ = ["Oded Cats, Arjan de Ruijter, Subodh Dubey, Nejc Gerzinic"]
__version__ = "1.0.1"
__maintainer__ = "Rafal Kucharski"
__email__ = "rafalkucharski.box _at_ gmail . com"

import ast
import math
import time
from itertools import product

from dotmap import DotMap
from enum import Enum

import numpy as np
import pandas as pd
import networkx as nx
import pulp

import matplotlib.pyplot as plt

from ExMAS_utils import init_log
pd.options.mode.chained_assignment = None


##########
# CONSTS #
##########

# columns of ride-candidates DataFrame
RIDE_COLS = ['indexes', 'u_pax', 'u_veh', 'kind', 'u_paxes', 'times', 'indexes_orig', 'indexes_dest']


class SbltType(Enum):  # type of shared ride. first digit is the degree, second is type (FIFO/LIFO/other)
    SINGLE = 1
    FIFO2 = 20
    LIFO2 = 21
    TRIPLE = 30
    FIFO3 = 30
    LIFO3 = 31
    MIXED3 = 32
    FIFO4 = 40
    LIFO4 = 41
    MIXED4 = 42
    FIFO5 = 50
    LIFO5 = 51
    MIXED5 = 52
    PLUS5 = 100


##############
# MAIN CALLS #
##############

# ALGORITHM 3
def main(_inData, params, plot=False):
    """
    main call
    :param _inData: input (graph, requests, .. )
    :param params: parameters
    :param plot: flag to plot charts for consecutive steps
    :return: inData.sblts.schedule - selecgted shared rides
    inData.sblts.rides - all ride candidates
    inData.sblts.res -  KPIs
    """
    _inData.logger = init_log(params)  # initialize console logger

    _inData = single_rides(_inData, params) # prepare requests as a potential single rides

    degree = 1

    _inData = pairs(_inData, params, plot=plot)
    degree = 2
    _inData.logger.info('Degree {} \tCompleted'.format(degree))

    if degree < params.max_degree:
        _inData = make_shareability_graph(_inData)

        while degree < params.max_degree and _inData.sblts.R[degree].shape[0] > 0:
            _inData.logger.info('trips to extend at degree {} : {}'.format(degree,
                                                                           _inData.sblts.R[degree].shape[0]))
            _inData = extend_degree(_inData, params, degree)
            degree += 1
            _inData.logger.info('Degree {} \tCompleted'.format(degree))
        if degree == params.max_degree:
            _inData.logger.info('Max degree reached {}'.format(degree))
            _inData.logger.info('Trips still possible to extend at degree {} : {}'.format(degree,
                                                                                          _inData.sblts.R[degree].shape[0]))
        else:
            _inData.logger.info(('No more trips to exted at degree {}'.format(degree)))


    _inData.sblts.rides = _inData.sblts.rides.reset_index(drop=True)  # set index
    _inData.sblts.rides['index'] = _inData.sblts.rides.index  # copy index

    if params.get('without_matching', False):
        return _inData  # quit before matching
    else:
        _inData = matching(_inData, params, plot=plot)
        _inData.logger.info('Calculations  completed')
        _inData = evaluate_shareability(_inData, params, plot=plot)

        return _inData

########
# CORE #
########


def single_rides(_inData, params):
    """
    prepare _inData.requests for calculations
    :param _inData: 
    :param params: parameters
    :return: 
    """

    def f_delta():
        # maximal possible delay of a trip (computed before join)
        return (1 / params.WtS - 1) * req.ttrav + \
               (params.price * params.shared_discount * req.dist / 1000) / (req.VoT * params.WtS)

    def utility_PT():
        # utility of trip with PT - not used
        if params.PT_discount == DotMap() or params.PT_beta == DotMap():
            return 999999
        else:
            return params.price * (1 - params.PT_discount) * req.dist / 1000 + req.VoT * params.PT_beta * req.timePT

    # prepare requests
    req = _inData.requests.copy().sort_index()
    t0 = req.treq.min() # set 0 as the earliest departure time
    req.treq = (req.treq - t0).dt.total_seconds().astype(int)  # recalc times for seconds starting from zero
    req.ttrav = req.ttrav.dt.total_seconds().divide(params.avg_speed).astype(int)  # recalc travel times using speed

    if params.get('heterogenic', False):
        pass
    else:
        req['VoT'] = params.VoT  # heterogeneity not applied

    req['delta'] = f_delta()  # assign maximal delay in seconds
    req['u'] = params.price * req.dist / 1000 + req.VoT * req.ttrav
    req = req.sort_values(['treq', 'pax_id'])  # sort
    req = req.reset_index()

    # req['timePT'] = 99999
    req['u_PT'] = utility_PT()

    # output
    _inData.sblts.requests = req.copy()
    df = req.copy()
    df['kind'] = SbltType.SINGLE.value  # assign a type for a ride
    df['indexes'] = df.index
    df['times'] = df.apply(lambda x: [x.treq, x.ttrav], axis=1)  # sequence of travel times
    df = df[['indexes', 'u', 'ttrav', 'kind', 'times']]  # columns to store as a shared ride
    df['indexes'] = df['indexes'].apply(lambda x: [x])
    df['u_paxes'] = df['u'].apply(lambda x: [x])

    df.columns = ['indexes', 'u_pax', 'u_veh', 'kind', 'times', 'u_paxes'] # synthax for the output rides
    df = df[['indexes', 'u_pax', 'u_veh', 'kind', 'u_paxes', 'times']]
    df['indexes_orig'] = df.indexes  # copy order of origins for single rides
    df['indexes_dest'] = df.indexes  # and dest
    df = df[RIDE_COLS]

    _inData.sblts.SINGLES = df.copy()  # single trips
    _inData.sblts.log.sizes[1] = {'potential': df.shape[0], 'feasible': df.shape[0]}
    _inData.sblts.rides = df.copy()

    _inData.sblts.R = dict()  # all the feasible rides
    _inData.sblts.R[1] = df.copy() # of a given degree

    return _inData


# ALGORITHM 1
def pairs(_inData, params, process=True, check=True, plot=False):
    """
    Identifies pair-wise shareable trips S_ij, i.e. for which utility of shared ride is greater than utility of
    non-shared ride for both trips i and j.
    First S_ij.FIFO2 trips are identified, i.e. sequence o_i,o_j,d_i,d_j.
    Subsequently, from FIFO2 trips we identify LIFO2 trips, i.e. o_i,o_j,d_j,d_i

    :param _inData: main data structure, with .skim (node x node dist matrix) , .requests (with origin, dest and treq)
    :param params: .json populated dictionary of parameters
    :param process: boolean flag to calculate measures at the end of calulations
    :param check: run test to make sure results are consistent
    :param plot: plot matrices illustrating the shareability
    :return: _inData with .sblts
    """
    # input
    req = _inData.sblts.requests.copy()  # work with single requests

    #VECTORIZED FUNCTIONS TO QUICKLY COMPUTE FORMULAS ALONG THE DATAFRAME
    def utility_ns_i():
        # utility of non-shared trip i
        return params.price * r.dist_i / 1000 + r.VoT_i * r.ttrav_i

    def utility_ns_j():
        # utility of non shared trip j
        return params.price * r.dist_j / 1000 + r.VoT_j * r.ttrav_j

    def utility_sh_i():
        # utility of shared trip i
        return (params.price * (1 - params.shared_discount) * r.dist_i / 1000 +
                r.VoT_i * params.WtS * (r.t_oo + params.pax_delay + r.t_od + params.delay_value * abs(r.delay_i)))

    def utility_sh_j():
        # utility of shared trip j
        return (params.price * (1 - params.shared_discount) * r.dist_j / 1000 +
                r.VoT_j * params.WtS * (r.t_od + r.t_dd + params.pax_delay +
                                        params.delay_value * abs(r.delay_j)))

    def utility_i():
        # difference u_sh_i - u_ns_i (has to be positive)
        return (params.price * r.dist_i / 1000 * params.shared_discount
                + r.VoT_i * (r.ttrav_i - params.WtS * (r.t_oo + r.t_od + params.pax_delay + params.delay_value * abs(r.delay_i))))

    def utility_j():
        # difference u_sh_i - u_ns_i
        return (params.price * r.dist_j / 1000 * params.shared_discount
                + r.VoT_j * (r.ttrav_j - params.WtS * (r.t_od + r.t_dd + params.pax_delay + params.delay_value * abs(r.delay_j))))

    def utility_i_LIFO():
        # utility of LIFO trip for i
        return (params.price * r.dist_i / 1000 * params.shared_discount
                + r.VoT_i * (r.ttrav_i - params.WtS * (
                        r.t_oo + r.t_od + 2 * params.pax_delay + r.t_dd + params.delay_value * abs(r.delay_i))))

    def utility_j_LIFO():
        # utility of LIFO trip for i
        return (params.price * r.dist_i / 1000 * params.shared_discount
                + r.VoT_j * (r.ttrav_j - params.WtS * (r.t_od + params.delay_value * abs(r.delay_j))))

    def utility_sh_i_LIFO():
        # difference u_sh_i_LIFO - u_ns_i
        return (params.price * (1 - params.shared_discount) * r.dist_i / 1000 +
                r.VoT_i * params.WtS * (r.t_oo + r.t_od + r.t_dd + 2 * params.pax_delay + params.delay_value * abs(r.delay_i)))

    def utility_sh_j_LIFO():
        # difference u_sh_j_LIFO - u_ns_j
        return (params.price * (1 - params.shared_discount) * r.dist_j / 1000 +
                r.VoT_j * params.WtS * (r.t_od + params.delay_value * abs(r.delay_j)))

    def query_skim(r, _from, _to, _col, _filter=True):
        """
        returns trip times for given node pair _from, _to and stroes into _col of df
        :param r: current set of queries
        :param _from: column name in r designating origin
        :param _to: column name in r designating destination
        :param _col: name of column in 'r' where matrix entries are stored
        :param _filter: do we filter the skim for faster queries (used always apart from the last query for LIFO2)
        :return: attributes in r
        """
        #
        if _filter:
            skim = the_skim.loc[
                r[_from].unique(), r[_to].unique()].unstack().to_frame()  # reduce the skim size for faster join
        else:
            skim = the_skim.unstack().to_frame()  # unstack and to_frame for faster column representation of matrix
        # skim matrix is unstacked (column vector) with two indexes
        skim.index.names = ['o', 'd']  # unify names for join
        skim.index = skim.index.set_names("o", level=0)
        skim.index = skim.index.set_names("d", level=1)

        skim.columns = [_col]
        # requests now has also two indexes
        r = r.set_index([_to, _from], drop=False)
        r.index = r.index.set_names("o", level=0)
        r.index = r.index.set_names("d", level=1)

        return r.join(skim, how='left')   # perform the jin to get the travel time for each request

    def sp_plot(_r, r, nCall, title):
        # function to plot binary shareability matrix at respective stages
        _r[1] = 0  # init boolean column
        if nCall == 0:
            _r.loc[r.index, 1] = 1  # initialize for first call
            sizes['initial'] = params.nP * params.nP
        else:
            _r.loc[r.set_index(['i', 'j']).index, 1] = 1
        sizes[title] = r.shape[0]
        _inData.logger.info(r.shape[0]+ '\t', title)
        mtx = _r[1].unstack().values
        axes[nCall].spy(mtx)
        axes[nCall].set_title(title)
        axes[nCall].set_xticks([])

    def check_me_FIFO():
        # does the asserion checks,
        # on one single random row
        t = r.sample(1).iloc[0]
        assert the_skim.loc[t.origin_i, t.origin_j] == t.t_oo  # check travel times consistency
        assert the_skim.loc[t.origin_j, t.destination_i] == t.t_od
        assert the_skim.loc[t.destination_i, t.destination_j] == t.t_dd
        assert abs(int(
            nx.shortest_path_length(_inData.G, t.origin_i, t.origin_j,
                                    weight='length') / params.avg_speed) - t.t_oo) < 2
        assert abs(int(nx.shortest_path_length(_inData.G, t.origin_j, t.destination_i,
                                               weight='length') / params.avg_speed) - t.t_od) < 2
        # and on the whole dataset
        try:
            assert (r.t_i - r.ttrav_i > -2).all()  # share time is not smaller than direct
            assert (r.t_j - r.ttrav_j > -2).all()
            assert (abs(r.delay_i) <= r.delta_i).all()  # is the delay within bounds
            assert (abs(r.delay_j) <= r.delta_j).all()
            assert (r.u_i <= utility_ns_i() * 1.01).all()  # do we have positive sharing utility
            assert (r.u_j <= utility_ns_j() * 1.01).all()  # do we have positive sharing utility
        except:
            _inData.logger.critical('FIFO pairs assertion failed')
            _inData.logger.warning(r[~(r.t_i - r.ttrav_i > -2)])  # share time is not smaller than direct
            _inData.logger.warning(r[~(r.t_j - r.ttrav_j > -2)])
            _inData.logger.warning(r[~(abs(r.delay_i) <= r.delta_i)])  # is the delay within bounds
            _inData.logger.warning(r[~(abs(r.delay_j) <= r.delta_j)])
            _inData.logger.warning(r[~(r.u_i <= utility_ns_i() * 1.01)])  # do we have positive sharing utility
            _inData.logger.warning(r[~(r.u_j <= utility_ns_j() * 1.01)])  # do we have positive sharing utility
            assert (r.t_i - r.ttrav_i > -2).all()  # share time is not smaller than direct
            assert (r.t_j - r.ttrav_j > -2).all()
            assert (abs(r.delay_i) <= r.delta_i).all()  # is the delay within bounds
            assert (abs(r.delay_j) <= r.delta_j).all()
            assert (r.u_i <= utility_ns_i() * 1.01).all()  # do we have positive sharing utility
            assert (r.u_j <= utility_ns_j() * 1.01).all()  # do we have positive sharing utility


    def check_me_LIFO():
        # does the asserion checks, on one single random row and on the whole dataser
        t = r.sample(1).iloc[0]
        try:
            assert the_skim.loc[t.origin_i, t.origin_j] == t.t_oo  # check travel times consistency
            assert the_skim.loc[t.origin_j, t.destination_j] == t.t_od
            assert the_skim.loc[t.destination_j, t.destination_i] == t.t_dd
            assert abs(int(
                nx.shortest_path_length(_inData.G, t.origin_i, t.origin_j,
                                        weight='length') / params.avg_speed) - t.t_oo) < 2
            assert abs(int(nx.shortest_path_length(_inData.G, t.destination_j, t.destination_i,
                                                   weight='length') / params.avg_speed) - t.t_dd) < 2
            assert (r.t_i + 3 - r.ttrav_i > 0).all()  # share time is not smaller than direct (3 seconds for rounding)
            assert (abs(r.delay_i) <= r.delta_i).all()  # is the delay within bounds
            assert (abs(r.delay_j) <= r.delta_j).all()
            assert (r.u_i <= utility_ns_i() * 1.01).all()  # do we have positive sharing utility
            assert (r.u_j <= utility_ns_j() * 1.01).all()  # do we have positive sharing utility
        except:
            _inData.logger.critical('LIFO pairs assertion failed')
            _inData.logger.warning(r[~(r.t_i + 3 - r.ttrav_i > 0)])  # share time is not smaller than direct (3 seconds for rounding)
            _inData.logger.warning(r[~(abs(r.delay_i) <= r.delta_i)])  # is the delay within bounds
            _inData.logger.warning(r[~(abs(r.delay_j) <= r.delta_j)])
            _inData.logger.warning(r[~(r.u_i <= utility_ns_i() * 1.01)])  # do we have positive sharing utility
            _inData.logger.warning(r[~(r.u_j <= utility_ns_j() * 1.01)])  # do we have positive sharing utility
            assert (r.t_i + 3 - r.ttrav_i > 0).all()  # share time is not smaller than direct (3 seconds for rounding)
            assert (abs(r.delay_i) <= r.delta_i).all()  # is the delay within bounds
            assert (abs(r.delay_j) <= r.delta_j).all()
            assert (r.u_i <= utility_ns_i() * 1.01).all()  # do we have positive sharing utility
            assert (r.u_j <= utility_ns_j() * 1.01).all()  # do we have positive sharing utility

        return r[r.delay_j >= r.delta_j]

    if plot:
        fig, axes = plt.subplots(1, 4, figsize=(16,4))
    _r = None

    sizes = dict()
    # MAIN CALULATIONS
    _inData.logger.info('Initializing pairwise trip shareability between {0} and {0} trips.'.format(params.nP))
    r = pd.DataFrame(index=pd.MultiIndex.from_product([req.index, req.index]))  # new df with a pairwise index
    _inData.logger.info('creating combinations')
    cols = ['origin', 'destination', 'ttrav', 'treq', 'delta', 'dist', 'VoT']
    r[[col + "_i" for col in cols]] = req.loc[r.index.get_level_values(0)][cols].set_index(r.index)  # assign columns
    r[[col + "_j" for col in cols]] = req.loc[r.index.get_level_values(1)][cols].set_index(r.index)  # time consuming

    r['i'] = r.index.get_level_values(0)  # assign index to columns
    r['j'] = r.index.get_level_values(1)
    r = r[~(r.i == r.j)]  # remove diagonal
    _inData.logger.info(str(r.shape[0]) + '\t nR*(nR-1)')
    _inData.sblts.log.sizes[2] = {'potential': r.shape[0]}

    # first condition (before querying any skim)
    if params.horizon > 0:
        r = r[abs(r.treq_i - r.treq_j) < params.horizon]
    q = '(treq_j + delta_j >= treq_i - delta_i)  & (treq_j - delta_j <= (treq_i + ttrav_i + delta_i))'
    r = r.query(q)  # this reduces the size of matrix quite a lot
    if plot:
        _r = r.copy()
        _r[1] = 0
        sp_plot(_r, r, 0, 'departure compatibility')
    if len(r) == 0:
        _inData.sblts.FIFO2 = r  # early exit with empty result
        return _inData

    # make the skim smaller  (query only between origins and destinations)
    skim_indices = list(set(r.origin_i.unique()).union(r.origin_j.unique()).union(
        r.destination_j.unique()).union(r.destination_j.unique()))  # limit skim to origins and destination only
    the_skim = _inData.skim.loc[skim_indices, skim_indices].div(params.avg_speed).astype(int)
    _inData.the_skim = the_skim

    r = query_skim(r, 'origin_i', 'origin_j', 't_oo')  # add t_oo to main dataframe (r)
    q = '(treq_i + t_oo + delta_i >= treq_j - delta_j) & (treq_i + t_oo - delta_i <= treq_j + delta_j)'
    r = r.query(q)  # can we arrive at origin of j within his time window?

    # now we can see if j is reachebale from i with the delay acceptable for both
    # determine delay for i and for j (by def delay/2, otherwise use bound of one delta and remainder for other trip)
    r['delay'] = r.treq_i + r.t_oo - r.treq_j
    r['delay_i'] = r.apply(lambda x: min(abs(x.delay / 2), x.delta_i, x.delta_j) * (1 if x.delay < 0 else -1), axis=1)
    r['delay_j'] = r.delay + r.delay_i

    r = r[abs(r.delay_j) <= r.delta_j / params.delay_value]  # filter for acceptable
    r = r[abs(r.delay_i) <= r.delta_i / params.delay_value]
    if plot:
        sp_plot(_r, r, 1, 'origins shareability')
    if len(r) == 0:
        _inData.sblts.FIFO2 = r  # early exit with empty result
        return _inData

    r = query_skim(r, 'origin_j', 'destination_i', 't_od')
    r = r[utility_i() > 0]  # and filter only for positive utility
    if plot:
        sp_plot(_r, r, 2, 'utility for i')
    if len(r) == 0:
        _inData.sblts.FIFO2 = r
        return _inData
    rLIFO = r.copy()

    r = query_skim(r, 'destination_i', 'destination_j', 't_dd')  # and now see if it is attractive also for j
    # now we have times for all segments: # t_oo_i_j # t_od_j_i # dd_i_j
    # let's compute utility for j
    r = r[utility_j() > 0]
    if plot:
        sp_plot(_r, r, 3, 'utility for j')
    if len(r) == 0:
        _inData.sblts.FIFO2 = r
        return _inData

    # profitability
    r['ttrav'] = r.t_oo + r.t_od + r.t_dd + 2 * params.pax_delay

    r = r.set_index(['i', 'j'], drop=False)  # done - final result of pair wise FIFO shareability

    if process:
        # lets compute some more measures
        r['kind'] = SbltType.FIFO2.value
        r['indexes'] = r.apply(lambda x: [int(x.i), int(x.j)], axis=1)
        r['indexes_orig'] = r.indexes
        r['indexes_dest'] = r.apply(lambda x: [int(x.i), int(x.j)], axis=1)

        r['u_i'] = utility_sh_i()
        r['u_j'] = utility_sh_j()

        r['t_i'] = r.t_oo + r.t_od + params.pax_delay
        r['t_j'] = r.t_od + r.t_dd + params.pax_delay
        r['delta_ij'] = r.apply(lambda x: x.delta_i - params.delay_value * abs(x.delay_i) - (x.t_i - x.ttrav_i), axis=1)
        r['delta_ji'] = r.apply(lambda x: x.delta_j - params.delay_value * abs(x.delay_j) - (x.t_j - x.ttrav_j), axis=1)
        r['delta'] = r[['delta_ji', 'delta_ij']].min(axis=1)
        r['u_pax'] = r['u_i'] + r['u_j']
        check_me_FIFO() if check else None

    _inData.sblts.FIFO2 = r.copy()
    del r

    # LIFO2
    r = rLIFO
    r = query_skim(r, 'destination_j', 'destination_i', 't_dd')  # set different sequence of times
    r.t_od = r.ttrav_j
    r = r[utility_i_LIFO() > 0]
    r = r[utility_j_LIFO() > 0]
    r = r.set_index(['i', 'j'], drop=False)
    r['ttrav'] = r.t_oo + r.t_od + r.t_dd + 2 * params.pax_delay
    # if sp.profitability:
    #    q = '(1 - ttrav/(ttrav_i+ttrav_j)) >{}'.format(sp.shared_discount)
    #    r = r.query(q) #only profitable trips

    if plot:
        _inData.logger.info(r.shape[0] + '\tLIFO pairs')
        sizes['LIFO'] = r.shape[0]

    if r.shape[0] > 0 and process:
        r['kind'] = SbltType.LIFO2.value

        r['indexes'] = r.apply(lambda x: [int(x.i), int(x.j)], axis=1)
        r['indexes_orig'] = r.indexes
        r['indexes_dest'] = r.apply(lambda x: [int(x.j), int(x.i)], axis=1)

        r['u_i'] = utility_sh_i_LIFO()
        r['u_j'] = utility_sh_j_LIFO()

        r['t_i'] = r.t_oo + r.t_od + r.t_dd + 2 * params.pax_delay
        r['t_j'] = r.t_od
        r['delta_ij'] = r.apply(
            lambda x: x.delta_i - params.delay_value * abs(x.delay_i) - (x.t_oo + x.t_od + x.t_dd - x.ttrav_i), axis=1)
        r['delta_ji'] = r.apply(lambda x: x.delta_j - params.delay_value * abs(x.delay_j), axis=1)
        r['delta'] = r[['delta_ji', 'delta_ij']].min(axis=1)
        r['u_pax'] = r['u_i'] + r['u_j']
        check_me_LIFO() if check else None

    _inData.sblts.LIFO2 = r.copy()
    _inData.sblts.pairs = pd.concat([_inData.sblts.FIFO2, _inData.sblts.LIFO2],
                                    sort=False).set_index(['i', 'j', 'kind'],
                                                          drop=False).sort_index()
    _inData.sblts.log.sizes[2]['feasible'] = _inData.sblts.LIFO2.shape[0] + _inData.sblts.FIFO2.shape[0]
    _inData.sblts.log.sizes[2]['feasibleFIFO'] = _inData.sblts.FIFO2.shape[0]
    _inData.sblts.log.sizes[2]['feasibleLIFO'] = _inData.sblts.LIFO2.shape[0]

    for df in [_inData.sblts.FIFO2.copy(), _inData.sblts.LIFO2.copy()]:
        if df.shape[0] > 0:
            df['u_paxes'] = df.apply(lambda x: [x.u_i, x.u_j], axis=1)
            df['u_veh'] = df.ttrav
            df['times'] = df.apply(
                lambda x: [x.treq_i + x.delay_i, x.t_oo + params.pax_delay, x.t_od, x.t_dd], axis=1)

            df = df[RIDE_COLS]

            _inData.sblts.rides = pd.concat([_inData.sblts.rides, df], sort=False)
    gain = (1 - float(r.shape[0]) / (params.nP * (params.nP - 1)))*100
    _inData.logger.info('Reduction of feasible pairs by {:.2f}%'.format(gain))
    if plot:
        if 'figname' in params.keys():
            plt.savefig(params.figname)
        fig, ax = plt.subplots(figsize=(164,4))
        pd.Series(sizes).plot(kind='barh', ax=ax, color='black') if plot else None
        ax.set_xscale('log')
        ax.invert_yaxis()
        plt.show()

    return _inData


def make_shareability_graph(_inData):
    """
    Prepares the shareability graphs from trip pairs
    :param _inData: inData.sblts.rides
    :return: inDara.sblts.S
    """
    rides = _inData.sblts.rides
    rides['degree'] = rides.apply(lambda x: len(x.indexes), axis=1)

    R2 = rides[rides.degree == 2].copy()
    R2['i'] = R2.indexes.apply(lambda x: x[0]) # for edge list
    R2['j'] = R2.indexes.apply(lambda x: x[1])
    R2 = R2.reset_index(drop=True)
    R2['index_copy'] = R2.index

    _inData.sblts.R[2] = R2
    _inData.sblts.S = nx.from_pandas_edgelist(_inData.sblts.R[2], 'i', 'j',
                                              edge_attr=['kind', 'index_copy'],
                                              create_using=nx.MultiDiGraph())  # create a graph
    return _inData


def enumerate_ride_extensions(r, S):
    """
    r rides
    S graph
    """
    ret = list()
    # find trips shareable with all trips of ride r
    S_r = None
    for t in r.indexes:  # iterate trips of ride r
        outs = set(S.neighbors(t))  # shareable trips with e
        S_r = outs if S_r is None else S_r & outs  # iterative intersection of trips
        if len(S_r) == 0:
            break  # early exit
    for q in S_r:  # iterate candidates
        E = [[S[e][q][i]['index_copy'] for i in list(S[e][q])] for e in
             r.indexes]  # list of (possibly) two edges connecting trips of r with q
        exts = list(product(*E))[0]
        if len(exts) > 0:
            ret.append(exts)
    return ret


# ALGORITHM 2/3
def extend_degree(_inData, params, degree):
    R = _inData.sblts.R

    # faster queries through dict
    dist_dict = _inData.sblts.requests.dist.to_dict()  # distances
    ttrav_dict = _inData.sblts.requests.ttrav.to_dict()  # travel times
    treq_dict = _inData.sblts.requests.treq.to_dict()  # requests times
    VoT_dict = _inData.sblts.requests.VoT.to_dict()  # valuoes of time

    nPotential = 0
    retR = list()  # for output

    for _, r in R[degree].iterrows():  # iterate through all rides to extend
        newtrips, nSearched = extend(r, _inData.sblts.S, R, params, degree, dist_dict, ttrav_dict, treq_dict, VoT_dict)
        retR.extend(newtrips)
        nPotential += nSearched

    df = pd.DataFrame(retR, columns=['indexes', 'indexes_orig', 'u_pax', 'u_veh', 'kind',
                                     'u_paxes', 'times', 'indexes_dest']) # data synthax for rides

    df = df[RIDE_COLS]
    df = df.reset_index()
    _inData.logger.info('At degree {} feasible extensions found out of {} searched'.format(degree,
                                                                                              df.shape[0],
                                                                                              nPotential))

    _inData.sblts.R[degree + 1] = df # store output
    _inData.sblts.rides = pd.concat([_inData.sblts.rides, df], sort=False)
    if df.shape[0] > 0:
        assert_extension(_inData, params, degree + 1)

    return _inData


# ALGORITHM 2 a
def extend(r, S, R, params, degree, dist_dict, ttrav_dict, treq_dict, VoT_dict):
    """
    extends a single ride of a given degree with all feasible rides of degree+1
    calls trip_sharing_utility to test if ride is attractive
    :param r: shared ride
    :param S: graph
    :param R: all rides of this degree
    :param params: 
    :param degree: 
    :param dist_dict: 
    :param ttrav_dict: 
    :param treq_dict: 
    :param VoT_dict: 
    :return: 
    """
    deptimefun = lambda dep: max([abs(dep + delay) for delay in delays])  # minmax
    deptimefun = np.vectorize(deptimefun)
    accuracy = 10
    retR = list()
    potential = 0
    for extension in enumerate_ride_extensions(r, S):  # all possible extensions
        Eplus, Eminus, t, kind = list(), list(), list(), None
        E = dict()  # star extending r with q
        indexes_dest = r.indexes_dest.copy()
        potential += 1

        for trip in extension:  # E = Eplus + Eminus
            t = R[2].loc[trip]
            E[t.i] = t  # trips lookup table to determine times
            if t.kind == 20:  # to determine destination sequence
                Eplus.append(indexes_dest.index(t.i))  # FIFO
            elif t.kind == 21:
                Eminus.append(indexes_dest.index(t.i))  # LIFO

        q = t.j

        if len(Eminus) == 0:
            kind = 0  # pure FIFO
            pos = degree
        elif len(Eplus) == 0:
            kind = 1  # pure LIFO
            pos = 0
        else:
            if min(Eminus) > max(Eplus):
                pos = min(Eminus)
                kind = 2
            else:
                kind = -1

        if kind >= 0:  # feasible ride
            re = DotMap()  # new extended ride
            re.indexes = r.indexes + [q]
            re.indexes_orig = re.indexes
            indexes_dest.insert(pos, q)  # new destination order
            re.indexes_dest = indexes_dest

            # times[1] = oo, times[2] = od, times[3]=dd

            new_time_oo = [E[re.indexes_orig[-2]].times[1]]  # this is always the case

            if pos == degree:  # insert as last destination
                new_time_od = [E[re.indexes_dest[0]].times[2]]
                new_time_dd = [E[re.indexes_dest[-2]].times[3]]
                new_times = [r.times[0:degree] +
                             new_time_oo +
                             new_time_od +
                             r.times[degree + 1:] +
                             new_time_dd]

            elif pos == 0:  # insert as first destination
                new_times = [r.times[0:degree] +
                             E[re.indexes_orig[-2]].times[1:3] +
                             [E[re.indexes_dest[1]].times[3]] +
                             r.times[degree + 1:]]

            else:
                new_time_od = [E[re.indexes_dest[0]].times[2]]
                new_times = r.times[0:degree] + new_time_oo + new_time_od
                if len(r.times[degree + 1:degree + 1 + pos - 1]) > 0:  # not changed sequence before insertion
                    new_times += r.times[degree + 1:degree + 1 + pos - 1]  # only for degree>3

                # insertion
                new_times += [E[re.indexes_dest[pos - 1]].times[-1]]  # dd
                new_times += [E[re.indexes_dest[pos + 1]].times[-1]]  # dd

                if len(r.times[(degree + 1 + pos):]) > 0:  # not changed sequence after insertion
                    new_times += r.times[degree + 1 + pos:]  # only for degree>3

                new_times = [new_times]

            new_times = new_times[0]
            re.times = new_times

            # detrmine utilities
            dists = [dist_dict[_] for _ in re.indexes]  # distances
            ttrav_ns = [ttrav_dict[_] for _ in re.indexes]  # non shared travel times
            VoT = [VoT_dict[_] for _ in re.indexes]  # VoT of sharing travellers
            # shared travel times
            ttrav = [sum(new_times[i + 1:degree + 2 + re.indexes_dest.index(re.indexes[i])]) for i in
                     range(degree + 1)]

            # first assume null delays
            feasible_flag = True
            for i in range(degree + 1):
                if trip_sharing_utility(params, dists[i], 0, ttrav[i], ttrav_ns[i], VoT[i]) < 0:
                    feasible_flag = False
                    break
            if feasible_flag:
                # determine optimal departure time (if feasible with null delay)
                treq = [treq_dict[_] for _ in re.indexes]  # distances

                delays = [new_times[0] + sum(new_times[1:i]) - treq[i] for i in range(degree + 1)]

                dep_range = int(max([abs(_) for _ in delays]))

                if dep_range == 0:
                    x = [0]
                else:
                    x = np.arange(-dep_range, dep_range, min(dep_range, accuracy))
                d = (deptimefun(x))

                delays = [abs(_ + x[np.argmin(d)]) for _ in delays]
                # if _print:
                #    pd.Series(d, index=x).plot()  # option plot d=f(dep)
                u_paxes = list()

                for i in range(degree + 1):

                    u_paxes.append(trip_sharing_utility(params, dists[i], delays[i], ttrav[i], ttrav_ns[i], VoT[i]))
                    if u_paxes[-1] < 0:
                        feasible_flag = False
                        break
                if feasible_flag:
                    re.u_paxes = [shared_trip_utility(params, dists[i], delays[i], ttrav[i], VoT[i]) for i in
                                  range(degree + 1)]
                    re.pos = pos
                    re.times = new_times
                    re.u_pax = sum(re.u_paxes)
                    re.u_veh = sum(re.times[1:])
                    if degree > 4:
                        re.kind = 100
                    else:
                        re.kind = 10 * (degree + 1) + kind
                    retR.append(dict(re))

    return retR, potential


def matching(_inData, params, plot=False, make_assertion=True):
    """
    called from the main loop
    :param _inData:
    :param plot:
    :param make_assertion: check if results are consistent
    :return: inData.sblts.schedule - selected rides (and keys to them in inData.sblts.requests)
    """
    rides = _inData.sblts.rides.copy()
    requests = _inData.sblts.requests.copy()

    opt_outs = False
    multi_platform_matching = params.get('multi_platform_matching', False)
    
    if not multi_platform_matching: # classic matching for single platform
        selected = match(im=rides, r=requests, params=params, plot=plot,
                         make_assertion=make_assertion, logger = _inData.logger)
        rides['selected'] = pd.Series(selected)

    else:  # matching to multiple platforms
        # select only rides for which all travellers are assigned to this platform
        rides['platform'] = rides.apply(lambda row: list(set(_inData.sblts.requests.loc[row.indexes].platform.values)),
                                        axis=1)

        rides['platform'] = rides.platform.apply(lambda x: -2 if len(x) > 1 else x[0])
        rides['selected'] = 0

        opt_outs = -1 in rides.platform.unique() # do we have travellers opting out

        for platform in rides.platform.unique():
            if platform>=0:
                platform_rides = rides[rides.platform == platform]
                selected = match(im=platform_rides, r=requests[requests.platform == platform], params=params,
                                 plot=plot, make_assertion=False, logger = _inData.logger)

                rides['selected'].update(pd.Series(selected))
        
    schedule = rides[rides.selected == 1].copy()

    req_ride_dict = dict()
    for i, trips in schedule.iterrows():
        for trip in trips.indexes:
            req_ride_dict[trip] = i
    requests['ride_id'] = pd.Series(req_ride_dict)
    ttrav_sh, u_sh, kinds = dict(), dict(), dict()
    for i, sh in schedule.iterrows():
        for j, trip in enumerate(sh.indexes):
            pos_o = sh.indexes_orig.index(trip) + 1
            pos_d = sh.indexes_dest.index(trip) + 1 + len(sh.indexes)
            ttrav_sh[trip] = sum(sh.times[pos_o:pos_d])
            u_sh[trip] = sh.u_paxes[j]
            kinds[trip] = sh.kind

    requests['ttrav_sh'] = pd.Series(ttrav_sh)
    requests['u_sh'] = pd.Series(u_sh)
    requests['kind'] = pd.Series(kinds)

    requests['position'] = requests.apply(
        lambda x: schedule.loc[x.ride_id].indexes.index(x.name) if x.ride_id in schedule.index else -1, axis=1)
    schedule['degree'] = schedule.apply(lambda x: len(x.indexes), axis=1)

    if make_assertion:  # test consitency
        assert opt_outs or len(
            requests.ride_id) - requests.ride_id.count() == 0  # all trips are assigned
        to_assert = requests[requests.ride_id >= 0] # only non optouts

        assert (to_assert.u_sh <= to_assert.u + 0.5).all
        assert (to_assert.ttrav <= (to_assert.ttrav_sh + 3)).all()
        if multi_platform_matching:
            for i in schedule.index.values:
                # check if all schedules are for travellers from the same platform
                assert _inData.requests.loc[schedule.loc[i].indexes].platform.nunique() == 1

    # store the results back
    _inData.sblts.rides = rides
    _inData.sblts.schedule = schedule
    _inData.sblts.requests = requests
    return _inData
    

def match(im, r, params, plot=False, make_assertion=True, logger = None):
    """
    main call of bipartite matching on a graph
    :param im: possible rides
    :param r: requests
    :param params: parameter (including objective function)
    :param plot:
    :param make_assertion: test the results at the end
    :return: rides, secelcted rides, reuests
    """
    request_indexes = dict()
    request_indexes_inv = dict()
    for i, index in enumerate(r.index.values):
        request_indexes[index] = i
        request_indexes_inv[i] = index

    im_indexes = dict()
    im_indexes_inv = dict()
    for i, index in enumerate(im.index.values):
        im_indexes[index] = i
        im_indexes_inv[i] = index

    im['lambda_r'] = im.apply(
        lambda x: params.shared_discount if x.kind == 1 else 1 - x.u_veh / sum([r.loc[_].ttrav for _ in x.indexes]),
        axis=1)

    im['PassHourTrav_ns'] = im.apply(lambda x: sum([r.loc[_].ttrav for _ in x.indexes]), axis=1)

    r = r.reset_index()

    if params.profitability:
        im = im[im.lambda_r >= params.shared_discount]
        logger.info('Out of {} trips  {} are directly profitable.'.format(r.shape[0],
                                                                    im.shape[0])) if logger is not None else None

    nR = r.shape[0]

    def add_binary_row(r):
        ret = np.zeros(nR)
        for i in r.indexes:
            ret[request_indexes[i]] = 1
        return ret

    logger.info('Matching {} trips to {} rides in order to minimize {}'.format(nR,
                                                                               im.shape[0],
                                                                               params.matching_obj)) if logger is not None else None
    im['row'] = im.apply(add_binary_row, axis=1)  # row to be used as constrain in optimization
    m = np.vstack(im['row'].values).T  # creates a numpy array for the constrains

    if plot:
        plt.rcParams['figure.figsize'] = [20, 3]
        plt.imshow(m)
        # plt.spy(m, c = 'blue')
        plt.show()

    im['index'] = im.index.copy()

    im = im.reset_index(drop=True)

    # optimization
    prob = pulp.LpProblem("Matching problem", pulp.LpMinimize)  # problem

    variables = pulp.LpVariable.dicts("r", (i for i in im.index), cat='Binary')  # decision variables

    cost_col = params.matching_obj
    if cost_col == 'degree':
        costs = im.indexes.apply(lambda x: -(10 ** len(x)))
    elif cost_col == 'u_pax':
        costs = im[cost_col]  # set the costs
    else:
        costs = im[cost_col]  # set the costs

    prob += pulp.lpSum([variables[i] * costs[i] for i in variables]), 'ObjectiveFun'  # ffef

    j = 0  # adding constrains
    for imr in m:
        j += 1
        prob += pulp.lpSum([imr[i] * variables[i] for i in variables if imr[i] > 0]) == 1, 'c' + str(j)

    prob.solve()  # main otpimization call

    logger.info('Problem solution: {}. \n'
          'Total costs for single trips:  {:13,} '
          '\nreduced by matching to: {:20,}'.format(pulp.LpStatus[prob.status], int(sum(costs[:nR])),
                                                    int(pulp.value(prob.objective)))) if logger is not None else None

    assert pulp.value(prob.objective) <= sum(costs[:nR]) + 2  # we did not go above original

    locs = dict()
    for variable in prob.variables():
        i = int(variable.name.split("_")[1])

        locs[im_indexes_inv[i]] = (int(variable.varValue))
        # _inData.logger.info("{} = {}".format(int(variable.name.split("_")[1]), int(variable.varValue)))

    return locs


def evaluate_shareability(_inData, params, plot=False):
    """
    Calc KPIs for the results of assigning trips to shared rides
    :param _inData:
    :param params:
    :param plot:
    :return:
    """

    # plot
    ret = DotMap()
    r = _inData.sblts.requests.copy()

    schedule = _inData.sblts.schedule.copy()
    schedule['ttrav'] = schedule.apply(lambda x: sum(x.times[1:]), axis=1)

    fare = 0

    ret['VehHourTrav'] = schedule.ttrav.sum()
    ret['VehHourTrav_ns'] = r.ttrav.sum()

    ret['PassHourTrav'] = r.ttrav_sh.sum()
    ret['PassHourTrav_ns'] = r.ttrav.sum()

    ret['PassUtility'] = r.u_sh.sum()
    ret['PassUtility_ns'] = r.u.sum()

    #ret['mean_ride_lambda'] = schedule.lambda_r.mean()
    ret['mean_lambda'] = 1 - schedule[schedule.kind > 1].u_veh.sum() / schedule[
        schedule.kind > 1].PassHourTrav_ns.sum()

    #ret['shared_fares'] = schedule[schedule.kind > 1].PassHourTrav_ns.sum() * sp.price * (
    #       1 - sp.shared_discount)
    #ret['full_fares'] = schedule[schedule.kind == 1].PassHourTrav_ns.sum() * sp.price
    ret['revenue_s'] = schedule.PassHourTrav_ns.sum() * params.price * (1 - params.shared_discount)
    ret['revenue_ns'] = schedule.PassHourTrav_ns.sum() * params.price
    ret['Fare_Discount'] = (ret['revenue_s'] - ret['revenue_ns']) / ret['revenue_ns']

    split = schedule.groupby('kind').sum()
    split['kind'] = split.index
    split['name'] = split.kind.apply(lambda x: SbltType(x).name)
    split = split.set_index('name')
    del split['kind']

    ret['nR'] = r.shape[0]

    ret['SINGLE'] = schedule[(schedule.kind == 1)].shape[0]
    ret['PAIRS'] = schedule[(schedule.kind > 1) & (schedule.kind < 30)].shape[0]
    ret['TRIPLES'] = schedule[(schedule.kind >= 30) & (schedule.kind < 40)].shape[0]
    ret['QUADRIPLES'] = schedule[(schedule.kind >= 40) & (schedule.kind < 50)].shape[0]
    ret['QUINTETS'] = schedule[(schedule.kind >= 50) & (schedule.kind < 100)].shape[0]
    ret['PLUS5'] = schedule[(schedule.kind == 100)].shape[0]
    ret['shared_ratio'] = 1 - ret['SINGLE'] / ret['nR']

    # df = pd.DataFrame(_inData.sblts.log.sizes).T[['potential', 'feasible']].reindex([1, 2, 3, 4])
    nR = r.shape[0]

    # df['selected'] = [ret['SINGLE'], ret['PAIRS'], ret['TRIPLES'], ret['QUADRIPLES']]
    # df['theoretical'] = [nR, nR ** 2, nR ** 3, nR ** 4]
    # _inData.sblts.log.sizes = df.fillna(0).astype(int)

    r['start'] = pd.to_datetime(r.treq, unit='s')
    r['end'] = pd.to_datetime(r.treq + r.ttrav, unit='s')
    fsns = fleet_size(r)
    ret['fleet_size_nonshared'] = max(fsns)

    schedule['start'] = pd.to_datetime([t[0] for t in schedule.times.values], unit='s')
    schedule['end'] = schedule.start + pd.to_timedelta([sum(t[1:]) for t in schedule.times.values], unit='s')
    fs = fleet_size(schedule)
    ret['fleet_size_shared'] = max(fs)
    if schedule[schedule.kind > 1].shape[0] > 0:
        shared_vehkm = schedule[schedule.kind > 1].u_veh.sum()
        ns_vehkm = r[r.kind > 1].ttrav.sum()
        ret['lambda_shared'] = 1 - shared_vehkm / ns_vehkm
    else:
        ret['lambda_shared'] = 0

    # ret['fleet_size_shared'] = max(fs)
    if plot:
        fig, ax = plt.subplots()
        ax.set_ylabel("number of rides")
        ax.set_xlabel("time")

        fsns.plot(drawstyle='steps', ax=ax)
        fs.plot(drawstyle='steps', ax=ax)
        ax.set_xticks([])

        plt.savefig('fleet.svg')

    _inData.logger.info(ret)
    _inData.sblts.res = pd.Series(ret)
    return _inData


#########
# UTILS #
#########

def trip_sharing_utility(params, dist, dep_delay, ttrav, ttrav_ns, VoT):
    # trip sharing utility for a trip, trips are shared only if this is positive.
    # difference
    return (params.price * dist / 1000 * params.shared_discount
            + VoT * (ttrav_ns - params.WtS * (ttrav + params.delay_value * abs(dep_delay))))


def shared_trip_utility(params, dist, dep_delay, ttrav, VoT):
    #  utility of a shared trip
    return (params.price * (1 - params.shared_discount) * dist / 1000 +
            VoT * params.WtS * (ttrav + params.delay_value * abs(dep_delay)))


def make_schedule(t, r):
    columns = ['node', 'times', 'req_id', 'od']
    degree = 2 * len(ast.literal_eval(t.indexes))
    df = pd.DataFrame(None, index=range(degree), columns=columns)
    x = ast.literal_eval(t.indexes_orig)
    s = [r.loc[i].origin for i in x] + [r.loc[i].destination for i in x]
    df.node = pd.Series(s)
    df.req_id = x + ast.literal_eval(t.indexes_dest)
    df.times = t.times
    df.od = pd.Series(['o'] * len(ast.literal_eval(t.indexes)) + ['d'] * len(ast.literal_eval(t.indexes)))
    return df


def fleet_size(requests):
    requests = requests.sort_values('start')
    pickups = requests.set_index('start')
    pickups['starts'] = 1
    ret = pickups.resample('60s').sum().cumsum()[['starts']]
    dropoffs = requests.set_index('end')
    dropoffs['ends'] = 1
    d = dropoffs.resample('60s').sum().ends.cumsum()
    ret = ret.join(d, how='outer')
    ret.starts = ret.starts.fillna(ret.starts.max())
    ret.ends = ret.ends.fillna(0)
    return ret.starts - ret.ends


def assert_extension(_inData, params, degree=3, nchecks=4, t=None):
    """
    Function checks whether all the resulting extended trips are coreectly calculated.
    Checks if ride travel times are in line with skim times.
    Used to debug, can be made silent or inactive for speed up (though it is definitely not a killer in performance)
    :param _inData:
    :param params:
    :param degree:
    :param nchecks:
    :param t:
    :return:
    """
    if t is None:
        rides = _inData.sblts.R[degree]
    else:
        rides = None
    the_skim = _inData.the_skim
    r = _inData.sblts.requests
    for _ in range(nchecks + 1):
        if t is None:
            t = rides.sample(1).iloc[0]
        os = t.indexes_orig
        ds = t.indexes_dest
        skim_times = list()
        degree = len(os)
        for i in range(degree - 1):
            o1 = r.loc[os[i]].origin
            o2 = r.loc[os[i + 1]].origin
            skim_times.append(the_skim.loc[o1, o2] + params.pax_delay)
        skim_times.append(the_skim.loc[r.loc[os[-1]].origin, r.loc[ds[0]].destination])

        for i in range(degree - 1):
            d1 = r.loc[ds[i]].destination
            d2 = r.loc[ds[i + 1]].destination
            skim_times.append(the_skim.loc[d1, d2])

        try:
            assert skim_times == t.times[1:]
            # if nchecks == 0:
            # _inData.logger.warning(skim_times, t.times[1:])
        except AssertionError as error:
            _inData.logger.critical('Assertion Error for extension')
            # _inData.logger.warning(t)
            _inData.logger.warning(params.pax_delay)
            _inData.logger.warning(skim_times)
            _inData.logger.warning(t.times[1:])
            assert skim_times == t.times[1:]


if __name__ == "__main__":

    from ExMAS_utils import init_log, make_paths, get_config, load_G, generate_demand, inData

    params = get_config('data/configs/default.json')
    params = make_paths(params)

    params.t0 = pd.Timestamp.now()

    from ExMAS_utils import inData as inData

    inData = load_G(inData, params, stats=True)  # download the CITY graph

    inData = generate_demand(inData, params, avg_speed=False)

    main(inData, params)



