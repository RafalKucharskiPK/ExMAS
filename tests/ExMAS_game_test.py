#!/usr/bin/env python

__author__ = "Rafal Kucharski"
__email__ = "r.m.kucharski@tudelft.nl"
__license__ = "MIT"

import unittest
import os
import sys
sys.path.append(".")
import pandas as pd
from ExMAS.game import games, pricings, prunings, pipeline

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'test_config.json')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # add local path for Travis CI

class TestGame(unittest.TestCase):

    def test_game(self):
        import ExMAS.utils

        params = ExMAS.utils.get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))

        
        #params.t0 = pd.Timestamp.now()

        params.logger_level = 'INFO'
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
        params.nP = 100
        params.simTime = 0.15
        params.shared_discount = 0.2
        params.t0 = pd.to_datetime('15:00')

        # prepare ExMAS
        inData = ExMAS.utils.generate_demand(inData, params)  # generate requests

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
        for PRICING, pricing in PRICINGS.items():
            inData = pricing(inData)  # apply pricing strategy
            for PRUNING, pruning in PRUNINGS.items():
                inData = pruning(inData, price_column=PRICING)  # apply pruning strategies for a given pricing strategy
            for PRUNING, pruning in PRUNINGS.items():  # perform assignment for single prunings
                inData = pipeline.single_eval(inData, params,
                                    EXPERIMENT_NAME='jupyter',
                                    MATCHING_OBJS=['total_group_cost'],  # this can be more
                                    PRUNINGS=[PRUNING],  # and this can be more
                                    PRICING=PRICING,  # this is taken from first level loop
                                    minmax=('min', 'max'))  # direction BPoA, WPoA







