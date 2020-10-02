#!/usr/bin/env python

__author__ = "Rafal Kucharski"
__email__ = "r.m.kucharski@tudelft.nl"
__license__ = "MIT"

import unittest
import os
import sys
import pandas as pd

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'test_config.json')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # add local path for Travis CI


class TestSimulationResults(unittest.TestCase):

    def test_results(self):
        import ExMAS.utils

        params = ExMAS.utils.get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))

        params.t0 = pd.Timestamp.now()

        from ExMAS.utils import inData as inData

        inData = ExMAS.utils.load_G(inData, params, stats=True)  # download the CITY graph

        inData = ExMAS.utils.generate_demand(inData, params)

        inData = ExMAS.main(inData, params)

        KPIs = inData.sblts.res

        assert KPIs.VehHourTrav < KPIs.VehHourTrav_ns  # less vehicle hours

        assert KPIs.PassUtility < KPIs.PassUtility_ns # greater utility

        assert KPIs.shared_ratio > 0 # some vehicles share

    def test_big(selfs):
        import ExMAS.utils

        params = ExMAS.utils.get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))
        params.nP = 1000
        params.shared_discount = 0.15

        params.t0 = pd.Timestamp.now()

        from ExMAS.utils import inData as inData

        inData = ExMAS.utils.load_G(inData, params, stats=True)  # download the CITY graph

        inData = ExMAS.utils.generate_demand(inData, params)

        inData = ExMAS.main(inData, params)

        KPIs = inData.sblts.res

        assert KPIs.VehHourTrav < KPIs.VehHourTrav_ns  # less vehicle hours

        assert KPIs.PassUtility < KPIs.PassUtility_ns  # greater utility

        assert KPIs.shared_ratio > 0  # some vehicles share

    def test_discount_works(selfs):
        import ExMAS.utils

        params = ExMAS.utils.get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))
        # params = ExMAS.utils.make_paths(params)

        params.t0 = pd.Timestamp.now()

        params.shared_discount = 0.2

        from ExMAS.utils import inData as inData

        inData = ExMAS.utils.load_G(inData, params, stats=True)  # download the CITY graph

        inData = ExMAS.utils.generate_demand(inData, params)

        params.shared_discount = 0.2
        inData = ExMAS.main(inData, params)
        KPI_02 = inData.sblts.res

        params.shared_discount = 0.4
        inData = ExMAS.main(inData, params)
        KPI_04 = inData.sblts.res

        assert  KPI_04.shared_ratio >  KPI_02.shared_ratio  # some vehicles share

    def test_VoT_works(selfs):
        import ExMAS.utils

        params = ExMAS.utils.get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))
        # params = ExMAS.utils.make_paths(params)

        params.t0 = pd.Timestamp.now()

        from ExMAS.utils import inData as inData

        inData = ExMAS.utils.load_G(inData, params, stats=True)  # download the CITY graph

        inData = ExMAS.utils.generate_demand(inData, params)

        params.VoT = 0.0035
        inData = ExMAS.main(inData, params)
        KPI_35 = inData.sblts.res

        params.VoT = 0.009
        inData = ExMAS.main(inData, params)
        KPI_90 = inData.sblts.res

        assert KPI_35.shared_ratio > KPI_90.shared_ratio  # some vehicles share


    def test_WtS_works(selfs):
        import ExMAS.utils

        params = ExMAS.utils.get_config(CONFIG_PATH, root_path=os.path.dirname(__file__))
        # params = ExMAS.utils.make_paths(params)

        params.t0 = pd.Timestamp.now()

        from ExMAS.utils import inData as inData

        inData = ExMAS.utils.load_G(inData, params, stats=True)  # download the CITY graph

        inData = ExMAS.utils.generate_demand(inData, params)

        params.WtS = 1.35
        inData = ExMAS.main(inData, params)
        KPI_135 = inData.sblts.res

        params.WtS = 1.1
        inData = ExMAS.main(inData, params)
        KPI_11 = inData.sblts.res

        assert KPI_11.shared_ratio > KPI_135.shared_ratio  # some vehicles share

