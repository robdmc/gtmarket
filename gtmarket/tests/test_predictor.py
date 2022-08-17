#! /usr/bin/env python

#############################################################################################
# TO RUN THESE TEST:
#     coverage erase &&  coverage run --source='.'  test_predictor.py -v && coverage report
#     python -m unittest test_predictor.DealsTest.test_growth -v   # for a single test
#############################################################################################


from copy import deepcopy
import datetime
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from unittest import TestCase, main
import easier as ezr
import fleming
import os
import time

from predictor import (
    ModelParams,
    SDRTeam,
    Deals,
    ModelParamsHist,
)


class CachedObjects:
    pkc = ezr.pickle_cache_state('active')

    @ezr.pickle_cached_container()
    def pipe_stats(self):  # pragma: no cover because not called when loading cache
        from simbiz import live_api as smb
        opp_loader = smb.OppLoader()
        opp_history_loader = smb.OppHistoryLoader()
        order_product_obj = smb.OrderProducts()
        ps = smb.PipeStats(opp_loader, opp_history_loader, order_product_obj)

        # I'm running this here so that all the required state is loaded into
        # the cached pipe_stats_obj
        params = ModelParams()
        params.fit_from_live_data(ps)
        # sdr = SDRTeam(params)
        # deals = Deals(sdr)
        # starting = fleming.floor(datetime.datetime.now())
        # ending = starting + relativedelta(years=1)
        # deals.get_predicted_revenue(starting, ending)
        return ps


class ModelParamsLiveTest(TestCase):
    FITTED_ATTRIBUTES = [
        'segment_days_to_win',
        'segment_deal_size',
        'segment_win_rate',
        'stage_pipe_value',
        'stage_time_to_win',
        'combined_pipe_value'
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.pipe_stats = CachedObjects().pipe_stats

    def test_that_live_fits_change_things(self):
        # Get a default params blob
        params = ModelParams()
        blob = params.to_blob()

        # Null out the fitted fields from the blob (make then negative)
        blob = self._null_out_blob(blob)
        params.from_blob(blob)

        # Make sure nulling happened
        blob = params.to_blob()
        self.assertTrue(self._all_blob_null(blob))

        # Now fit from live data
        params.fit_from_live_data(self.pipe_stats)
        blob = params.to_blob()

        # Make sure no nulls in blob
        self.assertFalse(self._any_blob_null(blob))

        # Check that serialization round trip works
        serialized = params.to_json()
        params2 = ModelParams()
        params2.from_json(serialized)
        self.assertDictEqual(params.to_blob(), params2.to_blob())

    def _null_out_blob(self, blob):
        for att in self.FITTED_ATTRIBUTES:
            for key in blob[att]:
                blob[att][key] = -1
        return deepcopy(blob)

    def _all_blob_null(self, blob):  # pragma: no cover
        for att in self.FITTED_ATTRIBUTES:
            for key in blob[att]:
                if blob[att][key] >= 0:
                    return False
        return True

    def _any_blob_null(self, blob):  # pragma: no cover
        for att in self.FITTED_ATTRIBUTES:
            for key in blob[att]:
                if blob[att][key] < 0:
                    return True
        return False


class SDRTeamTest(TestCase):
    def test_constant_rates(self):
        params = ModelParams()
        sdr_team = SDRTeam(params)
        sorted_plan = sorted(sdr_team.hiring_plan, key=lambda r: r['date'])
        starting_number_of_reps = sorted_plan[0]['num_hires']
        start_date = sorted_plan[1]['date']
        rate_func = sdr_team.get_sal_rate_function(starting_date=start_date)
        expected_rate = starting_number_of_reps * sdr_team.rep_sal_rate_per_day
        actual_rate = rate_func(-100)
        self.assertAlmostEqual(actual_rate, expected_rate, places=4)

    def test_ramp(self):
        import numpy as np
        params = ModelParams()
        sdr_team = SDRTeam(params)
        sorted_plan = sorted(sdr_team.hiring_plan, key=lambda r: r['date'])
        start_date = sorted_plan[1]['date']
        total_num_hires = sum(r['num_hires'] for r in sorted_plan[:2])
        starting_number_of_reps = sorted_plan[0]['num_hires']
        start_rate = starting_number_of_reps * sdr_team.rep_sal_rate_per_day
        final_rate = sdr_team.rep_sal_rate_per_day * total_num_hires

        days = 2
        expected_rate = start_rate + (final_rate - start_rate) * (1 - np.exp(-days / sdr_team.ramp_time_constant_days))

        rate_func = sdr_team.get_sal_rate_function(starting_date=start_date)
        actual_rate = rate_func(days)
        self.assertAlmostEqual(actual_rate, expected_rate, places=4)

    def test_assymptote(self):
        params = ModelParams()
        sdr_team = SDRTeam(params)
        total_reps = sum(r['num_hires'] for r in sdr_team.hiring_plan)
        expected_rate = total_reps * sdr_team.rep_sal_rate_per_day
        rate_func = sdr_team.get_sal_rate_function()
        actual_rate = rate_func(10000)
        self.assertAlmostEqual(actual_rate, expected_rate)

    def test_num_reps(self):
        params = ModelParams()
        sdr_team = SDRTeam(params)
        sorted_plan = sorted(sdr_team.hiring_plan, key=lambda r: r['date'])
        df = sdr_team.get_rep_time_series(sorted_plan[0]['date'], sorted_plan[-1]['date'])
        current_num_reps = 0
        for entry in sorted_plan:
            current_num_reps += entry['num_hires']
            self.assertEqual(current_num_reps, df.loc[entry['date'], 'num_reps'])

    def test_sal_rate_time_series(self):
        params = ModelParams()
        sdr_team = SDRTeam(params)
        sorted_plan = sorted(sdr_team.hiring_plan, key=lambda r: r['date'])
        total_reps = sum(r['num_hires'] for r in sorted_plan)
        final_rate = total_reps * sdr_team.rep_sal_rate_per_day
        starting = sorted_plan[-1]['date'] + relativedelta(years=5)
        starting = parse(str(starting.date()))
        ending = starting + relativedelta(days=5)
        df = sdr_team.get_sal_rate_time_series(starting, ending)
        self.assertAlmostEqual(final_rate, df.sals_per_day.iloc[-1])


class DealsTest(TestCase):

    def test_assymptote(self):
        today = fleming.floor(datetime.datetime.now(), day=1)

        # Make a default params
        params = ModelParams()

        # Change the hiring plan to a single rep hired a long time ago
        params.hiring_plan = [{'date': parse('1/1/2018'), 'num_hires': 1}]

        # Say it only takes the rep one day to ramp so that steady state reached quickly
        params.ramp_time_constant_days = 1

        # Pretend we will only make enterprise sals
        params.segment_allocation = {
            'commercial': 0,
            'enterprise': 1,
            'velocity': 0
        }

        # Say it only takes a day from sal to win so that steady state reached quickly
        params.segment_days_to_win['enterprise'] = 1

        # Create a deals object for this scenario
        sdr = SDRTeam(params)
        deals = Deals(sdr)

        # Make interval of interest far in the future so that steady state is reached
        starting = today + relativedelta(days=1)
        future_start = starting + relativedelta(years=2)
        ending = future_start + relativedelta(years=1)

        # Compute how much revenue was generated in that interval
        df = deals.get_predicted_revenue(starting=starting, ending_exclusive=ending)
        df = df.loc[future_start:ending, :]
        df = df - df.iloc[0, :]
        actual_value = df.grand_total.iloc[-1]

        # Compute what you would expect at steady state
        segment = 'enterprise'
        win_rate = params.segment_win_rate[segment]
        sal_creation_rate = params.rep_sal_rate_per_day
        deal_value = params.segment_deal_size[segment]
        duration = (ending - future_start).days
        expected_value = sal_creation_rate * win_rate * deal_value * duration

        # Make sure actual and expected agree to within a percent
        percent_difference = 100 * abs(expected_value - actual_value) / actual_value
        self.assertTrue(percent_difference < 1)

    def test_growth(self):
        import numpy as np
        today = fleming.floor(datetime.datetime.now(), day=1)

        # Make a default params
        params = ModelParams()

        # Change the hiring plan to a single rep hired a long time ago
        params.hiring_plan = [{'date': parse('1/1/2018'), 'num_hires': 1}]

        # Say it only takes the rep one day to ramp so that steady state reached quickly
        params.ramp_time_constant_days = 1

        # Pretend we will only make enterprise sals
        params.segment_allocation = {
            'commercial': 0,
            'enterprise': 1,
            'velocity': 0
        }

        # Say it only takes a month to win a deal to get a nice ramp
        days_to_win = 30.4
        params.segment_days_to_win['enterprise'] = days_to_win

        # Create a deals object for this scenario
        sdr = SDRTeam(params)
        deals = Deals(sdr)

        # Make interval of interest starting today for a year
        starting = today
        ending = starting + relativedelta(years=1)

        # Compute how much revenue was generated in that interval
        df = deals.get_predicted_revenue(starting=starting, ending_exclusive=ending)

        # Compute what you would expect at steady state
        segment = 'enterprise'
        win_rate = params.segment_win_rate[segment]
        sal_creation_rate = params.rep_sal_rate_per_day
        deal_value = params.segment_deal_size[segment]
        K = sal_creation_rate * win_rate * deal_value
        t = np.arange(len(df))
        df['t'] = t
        df['expected'] = K * t - K * days_to_win * (1 - np.exp(-t / days_to_win))
        df['pct_error'] = 100 * (df.grand_total - df.expected).abs() / df.expected

        # Make sure that the biggest error is under one percent
        self.assertTrue(df.pct_error.abs().max() < 1)


class ModelParamsHistTest(TestCase):
    def test_history(self):
        file_name = '/tmp/test.sqlite'
        if os.path.isfile(file_name):
            os.unlink(file_name)
        hist = ModelParamsHist(sqlite_file=file_name)
        for nn in range(0, 3):
            time.sleep(1.1)
            mp = ModelParams()
            mp.segment_win_rate['commercial'] = nn
            hist.store(mp)

        df = hist.get_history()
        times_set = set()

        for nn, tup in enumerate(df.itertuples()):
            mp = ModelParams().from_json(tup.data)
            times_set.add(tup.utc_seconds)
            self.assertEqual(nn, mp.segment_win_rate['commercial'])

        self.assertEqual(len(times_set), 3)


if __name__ == '__main__':  # pragma: no cover
    main()
