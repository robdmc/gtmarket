import datetime
import warnings
from unittest import TestCase, main

import fleming
import easier as ezr
import pandas as pd
import numpy as np
from scipy import stats
from dateutil.relativedelta import relativedelta


class OppLoader(ezr.pickle_cache_mixin):
    pkc = ezr.pickle_cache_state('reset')

    EARLIEST_CREATION_DATE = pd.Timestamp('1/1/2019')

    def __init__(
            self,
            limit_to_standardized_dates=True,
            today=None,
            pilots_are_new_biz=False,
            sales_expansion_are_new_biz=False):
        """
        The sales process has changed quite a bit.  Limiting to opps
        created after 1/1/2019 will ensure that we only consider
        opps in new sales process

        today:  Any opp with created date past "today" will get ignored.
        """
        self.limit_to_standardized_dates = limit_to_standardized_dates
        if today is None:
            self.today = fleming.floor(datetime.datetime.now(), day=1)
        else:
            self.today = today
        self.today = pd.Timestamp(self.today)
        self.pilots_are_new_biz = pilots_are_new_biz
        self.sales_expansion_are_new_biz = sales_expansion_are_new_biz

    @ezr.pickle_cached_container()
    def df_raw(self):
        """
        A raw dump of the sfdc report
        """
        # Get raw sfdc dump
        sfdc = ezr.SalesForceReport()
        df = sfdc.get_report('00O0B000003xkOVUAY')

        # Slugify columns
        df = df.rename(columns=ezr.slugify)
        return df

    @ezr.cached_container
    def df_all(self):
        """
        The sfdc report with date fields transformed
        """
        # Get raw frame
        df = self.df_raw

        # Make dates appropriate
        date_cols = [
            'close_date',
            'created_date',
            'sql_date',
            'sal_date',
            'upside_forecast_status_date',
            'commit_forecast_status_date',
            'meeting_occurred_on',
        ]
        for col in date_cols:
            df.loc[:, col] = df.loc[:, col].astype(np.datetime64)

        # Ensure no future opps included
        df = df[df.created_date < (self.today + datetime.timedelta(days=1))]

        # Limit to standardized dates if requested
        if self.limit_to_standardized_dates:
            df = df[df.created_date >= self.EARLIEST_CREATION_DATE]

        # Make a status code field for determining if opp is opene or closed
        df['status_code'] = 0
        won_index = df.stage[df.stage == 'Closed Won'].index
        df.loc[won_index, 'status_code'] = 2

        lost_index = df.stage[df.stage == 'Closed Lost - Dead'].index
        df.loc[lost_index, 'status_code'] = 1

        # Compute the number of days that opp was/has-been open
        latest_open_dates = pd.Series(np.minimum(df.close_date.dt.to_pydatetime(), self.today), index=df.index)
        df['days_open'] = (latest_open_dates - df.created_date).dt.days

        df.loc[:, 'market_segment'] = df.market_segment.fillna('unknown').str.lower()

        return df

    @ezr.cached_container
    def df_new_biz(self):
        """
        Limited to only new business opps
        """
        df = self.df_all
        allowed_types = ['New Business']
        if self.pilots_are_new_biz:
            allowed_types.append('Pilot')
        if self.sales_expansion_are_new_biz:
            allowed_types.append('Sales Expansion')
        df = df[df.type.isin(allowed_types)]
        return df

    @ezr.cached_container
    def df_pipe(self):
        """
        Limit to only opps that hit sql
        """
        df = self.df_new_biz

        # All opps with sql date hit sql
        df = df[df.sql_date.notnull()]
        return df


class OppHistoryLoader(ezr.pickle_cache_mixin):
    pkc = ezr.pickle_cache_state('reset')

    def __init__(self, today=None):
        if today is None:
            today = fleming.floor(datetime.datetime.now(), day=1)
        self.today = pd.Timestamp(today)

    @ezr.pickle_cached_container()
    def df_raw(self):
        # Get raw sfdc dump
        sfdc = ezr.SalesForceReport()
        df = sfdc.get_report('00O0B000003xkP4UAI', slugify=True, date_fields=['last_modified'])
        return df

    @ezr.cached_container
    def df_all(self):
        df = self.df_raw
        df = df[df.last_modified < self.today + datetime.timedelta(days=1)]
        return df

    @ezr.cached_container
    def df_new_biz(self):
        df = self.df_all
        df = df[df.type == 'New Business']

        df = df[df.last_modified < self.today + datetime.timedelta(days=1)]
        return df


class PipeStats(ezr.pickle_cache_mixin):

    DEFAULT_HISTORY_DELTA = relativedelta(years=1)

    pkc = ezr.pickle_cache_state('reset')

    def __init__(
            self,
            opp_loader=None,
            opp_history_loader=None,
            order_product_obj=None,
            starting_date=None,
            horizon_date=None,
            today=None,
            pilots_are_new_biz=False,
            sales_expansion_are_new_biz=False):
        """
        Add docstrings later
        """
        if opp_loader is None:
            opp_loader = OppLoader(
                today=today,
                pilots_are_new_biz=pilots_are_new_biz,
                sales_expansion_are_new_biz=sales_expansion_are_new_biz)

        if opp_history_loader is None:
            opp_history_loader = OppHistoryLoader(today=today)

        if order_product_obj is None:
            order_product_obj = OrderProducts()

        self.loader = opp_loader
        self.op = order_product_obj
        self.opp_history_loader = opp_history_loader
        if today is None:
            self.today = self.loader.today
        else:
            self.today = pd.Timestamp(today)
        if horizon_date is None:
            horizon_date = self.today + relativedelta(years=1)
        self.horizon_date = horizon_date

        if starting_date is not None:
            self.history_delta = (self.today - starting_date)
        else:
            self.history_delta = self.DEFAULT_HISTORY_DELTA

    @classmethod
    def enable_pickle_cache(cls):
        import gtmarket as gtm
        gtm.OppLoader.enable_pickle_cache()
        gtm.OppHistoryLoader.enable_pickle_cache()
        gtm.OrderProducts.enable_pickle_cache()
        for base in cls.__bases__:
            if hasattr(base, 'enable_pickle_cache'):
                base.enable_pickle_cache()


    @classmethod
    def disable_pickle_cache(cls):
        import gtmarket as gtm
        gtm.OppLoader.disable_pickle_cache()
        gtm.OppHistoryLoader.disable_pickle_cache()
        gtm.OrderProducts.disable_pickle_cache()

        for base in cls.__bases__:
            if hasattr(base, 'disable_pickle_cache'):
                base.disable_pickle_cache()


    def _process_opp_frame(self, df):
        # Fix open opps with past close dates
        ind = df[(df.status_code == 0) & (df.close_date < self.today)].index
        if len(ind) > 0:
            df.loc[ind, 'close_date'] = self.today + relativedelta(days=1)

        # Fix closed opps with future close dates
        run_time_today = fleming.floor(datetime.datetime.now(), day=1)
        ind = df[(df.status_code != 0) & (df.close_date > run_time_today)].index
        if len(ind) > 0:
            df.loc[ind, 'close_date'] = run_time_today - relativedelta(days=1)

        # Ignore all opps with close date past horizon
        df = df[df.close_date < self.horizon_date].copy()

        df.loc[:, 'market_segment'] = df.market_segment.fillna('unknown')
        return df.copy()

    @ezr.cached_container
    def df_pipeline(self):
        """
        This will be all new-business sql ops with ACV filled-in for "all time"
        """
        df = self._process_opp_frame(self.loader.df_pipe)
        return df

    @ezr.cached_container
    def df_new_biz(self):
        """
        This will be all new-business sql ops with ACV filled-in for "all time"
        """
        df = self._process_opp_frame(self.loader.df_new_biz)
        return df

    @ezr.cached_container
    def df_won(self):
        df = self.df_pipeline
        df = df[df.stage == 'Closed Won']
        return df

    def get_df_orders(self, starting=None, ending=None):
        """
        Get a dataframe of orders with order_start_date
        defaulting to history_delta days ago
        """

        # Set default starting/ending
        if starting is None:
            starting = self.today - self.history_delta
        if ending is None:
            ending = self.today

        # Get all orders
        df = self.op.df_orders

        # Only get "New Business" orders (Not sure if "committed" belongs in here)
        df = df[df.order_type.isin(['New Business', 'Committed'])]

        # Order nicely
        df = df.sort_values(by='order_start_date')

        # Limit to time frame of interest
        df = df[(df.order_start_date >= starting) & (df.order_start_date < ending)].reset_index(drop=True)

        # Compute ACV and return
        df['acv'] = df.mrr.values * np.minimum(df.months, 12)
        return df

    @property
    def _stage_translator(self):
        return {
            "Champion's Choice Confirmed": 'Proposal',
            'Contract + Procurement + Legal': 'Negotiation',
            'Demo Complete + Qualification': 'Demo',
            'Sandbox + IT Call': 'Negotiation',

            'Closed Lost - Dead': 'Closed Lost - Dead',
            'Closed Won': 'Closed Won',
            'Demo': 'Demo',
            'Discovery': 'Discovery',
            'Negotiation': 'Negotiation',
            'Proposal': 'Proposal',
            'SAL': 'SAL',
        }

    def _translate_stage_name(self, stage):
        return self._stage_translator.get(stage, 'invalid')

    def translate_stage_in_frame(self, df, stage_column_name):
        df = df.copy()
        df.loc[:, stage_column_name] = df[stage_column_name].map(self._translate_stage_name)
        df = df[df[stage_column_name] != 'invalid']
        return df

    @ezr.cached_container
    def df_all_opp_segment_allocation(self):
        df = self.loader.df_new_biz
        df = df[df.market_segment != 'unknown']
        df = df[df.status_code == 0]
        df = df[['opportunity_id', 'status_code', 'market_segment']]
        df['num_opps'] = 1
        df = df.groupby(by='market_segment')[['num_opps']].sum()
        df = df / df.num_opps.sum()
        df = df.rename(columns={'num_opps': 'segment_allocation'})
        return df

    def get_opp_timeseries(self, value='pipeline_acv', interval_days=30, cumulative_since=None):
        """
        allowed_values = [
            'pipeline_acv',
            'num_sals',
            'num_sqls',
            'num_deals',
            'deal_acv',
        ]
        """
        allowed_values = [
            'pipeline_acv',
            'num_sals',
            'num_sqls',
            'num_deals',
            'deal_acv',
        ]
        if value not in allowed_values:
            raise ValueError(f'value must be in {allowed_values}')

        if cumulative_since is not None:
            cumulative_since = pd.Timestamp(cumulative_since)

        frame_mapper = {
            'num_sals': self.df_new_biz,
            'num_sqls': self.df_pipeline,
            'pipeline_acv': self.df_pipeline,
            'num_deals': self.df_won,
            'deal_acv': self.df_won,
        }
        df = frame_mapper[value]

        # # Set the appropriate source dataframe based on value
        # if value in ['pipeline_acv', 'num_sqls']:
        #     df = self.df_pipeline
        # elif value in ['num_sals']:
        #     df = self.df_new_biz
        # elif value in ['num_deals', 'deal_acv']:
        #     df = self.df_won

        if df.empty:
            return df

        # Fake ACV to be opp_count based on value
        if value in ['num_sals', 'num_sqls', 'num_deals']:
            df.loc[:, 'acv'] = 1

        # Set the date based on value
        date_field_mapper = {
            'num_sals': 'created_date',
            'num_sqls': 'sql_date',
            'pipeline_acv': 'sql_date',
            'num_deals': 'close_date',
            'deal_acv': 'close_date',
        }
        date_field = date_field_mapper[value]

        # # Set the date based on value
        # if value in ['num_sals']:
        #     date_field = 'created_date'
        # elif value in ['num_sqls', 'pipeline_acv']:
        #     date_field = 'sql_date'
        # elif value in ['num_deals', 'deal_acv']:
        #     date_field = 'close_date'
        # # Don't need an else clause here because I've already ensured only valid values for this

        df['date'] = df[date_field]
        df.loc[:, 'market_segment'] = ezr.slugify(df.market_segment)

        # Get a reference to all possible market segments
        dfseg = self.df_new_biz
        dfseg = dfseg[dfseg.market_segment != 'unknown']
        market_segments_columns = dfseg.market_segment.apply(ezr.slugify).value_counts().sort_index().index

        df = df.copy()

        # Collapse all pipe created in a day to a single record
        df = df[['date', 'market_segment', 'acv']].groupby(by=['date', 'market_segment'])[['acv']].sum()
        df = df.unstack('market_segment').sort_index().fillna(0)
        df.columns = df.columns.get_level_values(1)
        df.columns.name = None
        df.index.name = None

        df = df.reindex(market_segments_columns, axis=1).fillna(0)

        # Create an even sequence of days to index by
        min_date, max_date = df.index[0], df.index[-1]
        new_ind = pd.date_range(min_date, max_date, freq='D', name='date')

        # Make sure the frame has an entry for every day (zeros where nothing happened)
        df = df.reindex(new_ind).fillna(0., )

        if cumulative_since:
            if cumulative_since < min_date + relativedelta(days=1):
                raise ValueError(f'Can only go back as far as {min_date + relativedelta(days=1)}')
            dfg = df.cumsum()
            dfg = dfg.subtract(dfg.loc[cumulative_since - relativedelta(days=1), :])
            dfg = dfg.loc[cumulative_since:, :]
        else:
            dfg = df.rolling(interval_days, min_periods=1).sum().dropna()

            # Limit to the appropriate ealiest date
            starting = self.today - self.history_delta
            dfg = dfg.loc[pd.IndexSlice[starting:], :]

        if 'unknown' in dfg.columns:
            dfg = dfg.drop('unknown', axis=1)

        return dfg

    def get_conversion_timeseries(
            self, value, interval_days, bake_days=None, prior_pseudo_count=3):
        """
        """
        allowed_values = [
            'sal2sql_opps',
            'sal2won_opps',
            'sql2won_opps',

            'sal2sql_time',
            'sal2lost_time',
            'sal2won_time',

            'sql2won_time',
            'sql2lost_time',
        ]
        if value not in allowed_values:
            raise ValueError(f'value must be in {allowed_values}')

        # Set the appropriate source dataframe based on value
        if value in ['sal2sql_opps', 'sal2won_opps', 'sal2lost_time']:
            df = self.df_new_biz

        elif value in ['sql2won_opps', 'sal2sql_time', 'sal2won_time', 'sql2won_time', 'sql2lost_time']:
            df = self.df_pipeline

        if df.empty:
            return df

        if value in ['sal2sql_opps']:
            # For sal->sql only consider opps that have either been closed or hit sql
            df = df[(df.status_code != 0) | df.sql_date.notnull()]

            # The decision date for sql is either sql_date or close_date depending on outcome
            df['date'] = [
                close_date if pd.isnull(sql_date) else sql_date
                for (close_date, sql_date) in zip(df.sql_date, df.close_date)
            ]

        elif ('won' in value) or ('time' in value):
            # Only consider closed opps
            df = df[df.status_code != 0]

            # The decision date is the close date
            df['date'] = df.close_date

        if df.empty:
            return df

        df.loc[:, 'market_segment'] = ezr.slugify(df.market_segment)

        # Get a reference to all possible market segments
        dfseg = self.df_new_biz
        dfseg = dfseg[dfseg.market_segment != 'unknown']
        market_segments_colums = dfseg.market_segment.apply(ezr.slugify).value_counts().sort_index().index

        # Create an even sequence of days to index by
        min_date, max_date = df.date.min(), df.date.max()
        date_ind = pd.date_range(min_date, max_date, freq='D', name='date')

        # Set the outcome for the conversions
        if value in ['sal2sql_opps']:
            df['outcome'] = ['lost' if died_sal else 'won' for died_sal in df.sql_date.isnull()]

        elif value in ['sql2won_opps']:
            df['outcome'] = ['won' if closed_won else 'lost' for closed_won in (df.status_code == 2)]

        elif value in ['sal2won_opps'] or ('time' in value):
            df['outcome'] = ['won' if closed_won else 'lost' for closed_won in (df.status_code == 2)]

        if value in ['sal2sql_time']:
            df['days'] = (df.sql_date - df.created_date).dt.days

        elif value in ['sal2lost_time', 'sal2won_time']:
            df['days'] = (df.close_date - df.created_date).dt.days

        elif value in ['sql2lost_time', 'sql2won_time']:
            df['days'] = (df.close_date - df.sql_date).dt.days

        def extract_conversion_timeseries(df):
            df = df.groupby(by=['date', 'market_segment'])[['num']].sum().unstack('market_segment').fillna(0)
            dfnum = df.loc[:, 'num']
            dfnum = dfnum.reindex(market_segments_colums, axis=1).fillna(0)
            dfnum = dfnum.reindex(date_ind).fillna(0., )
            dfnum = dfnum.rolling(interval_days).sum().dropna()
            return dfnum

        def extract_duration_timeseries(df):
            df = df.groupby(by=['date', 'market_segment'])[['days', 'num']].sum().unstack('market_segment')
            df = df.loc[:, ['days', 'num']].fillna(1e-12)
            df = df.reindex(date_ind).fillna(1e-12)
            dfn = df.loc[:, 'num']
            dfd = df.loc[:, 'days']
            dfd = dfd.rolling(interval_days).sum().dropna()
            dfn = dfn.rolling(interval_days).sum().dropna()

            for col in dfd.columns:
                dfd.loc[:, col] = dfd.loc[:, col].mask(dfd.loc[:, col] < 1, 0)

            dft = dfd / dfn
            return dft

        df['num'] = 1

        dfw = df[df.outcome == 'won']
        dfl = df[df.outcome == 'lost']

        dfw_count = extract_conversion_timeseries(dfw)
        dfl_count = extract_conversion_timeseries(dfl)

        if 'time' in value:
            dfw_days = extract_duration_timeseries(dfw)
            dfl_days = extract_duration_timeseries(dfl)
            dfsal_sql_days = extract_duration_timeseries(df)

        # I'm going to assume a beta_binomial distribution here.  To do so, I want
        # to set reasonable priors based on data set as a whole, so here is how I do that.
        # First I make sure each market segment has at least one (possibly fake) count.
        df_prior = pd.DataFrame({'won': dfw_count.sum(), 'lost': dfl_count.sum()}) + 1

        # Add up all counts over the whole dataset by market segment
        df_prior = df_prior.rename(columns={'won': 'a', 'lost': 'b'})

        # Normalize to have the smallest "prior count" set to a specified number
        df_prior = prior_pseudo_count * df_prior.divide(df_prior.min(axis=1), axis=0).T

        # Create dataframes of a and b parameters for beta distribution
        dfa = dfw_count + df_prior.loc['a', :]
        dfb = dfl_count + df_prior.loc['b', :]

        # Make a beta distribution from these parameters
        dist = stats.beta(a=dfa.values, b=dfb.values)

        df_conv = pd.DataFrame(dist.mean(), index=dfa.index, columns=dfa.columns)

        if 'opps' in value:
            dfout = df_conv
        elif 'time' in value and 'won' in value:
            dfout = dfw_days
        elif 'time' in value and 'lost' in value:
            dfout = dfl_days
        elif value == 'sal2sql_time':
            dfout = dfsal_sql_days

        starting = self.today - self.history_delta
        dfout = dfout.loc[pd.IndexSlice[starting:self.today], :].copy()

        def remove_unbacked(df, bake_days):
            if bake_days is not None:
                bake_date = df.index[-1] - relativedelta(days=bake_days)
                df.loc[bake_date:, :] = np.NaN
                df = df.fillna(method='ffill')
            return df

        dfout = remove_unbacked(dfout, bake_days)
        dfout = dfout[market_segments_colums]

        return dfout

    def get_stage_win_rates_timeseries(
            self, interval_days, today=None, bake_days=0, prior_pseudo_count=5, prior_win_rate=1 / 3):
        # Get a frame of closed-won opps along with their close date
        dfo = self.loader.df_new_biz[['opportunity_id', 'status_code', 'close_date']]

        # Only consider closed opps
        dfo = dfo[dfo.status_code != 0]

        # Get a record of all the stages changes
        dfh = self.opp_history_loader.df_new_biz
        dfh = self.translate_stage_in_frame(dfh, 'to_stage')

        # Don't care about changed to close status
        dfh = dfh[~dfh.to_stage.isin(['Closed Lost - Dead', 'Closed Won'])]

        # Keep only the latest time an opp passed through a stage
        dfh = dfh.groupby(by=['opportunity_id', 'to_stage'])[['last_modified']].max().reset_index()

        # Limit to fields I need
        dfh = dfh[['opportunity_id', 'to_stage']]

        # Join to keep only won opps having stage and close dates
        df = pd.merge(dfh, dfo, on='opportunity_id', how='inner')
        df.loc[:, 'status_code'] = df.status_code.map({1: 'lost', 2: 'won'})
        df['num'] = 1

        # Add pseudocounts to bias small results
        pseudo_won_count = prior_pseudo_count * prior_win_rate
        pseudo_lost_count = prior_pseudo_count - pseudo_won_count

        # Internal function to compute the win rate by stage for a batch of dates
        def compute_win_rate(batch):
            dfg = batch.groupby(by=['to_stage', 'status_code'])[['num']].sum().unstack('status_code')
            dfg.columns = dfg.columns.get_level_values(1)
            dfg.index.name = None
            dfg.loc[:, 'lost'] += pseudo_lost_count
            dfg.loc[:, 'won'] += pseudo_won_count
            dfg['win_rate'] = dfg.won / dfg.sum(axis=1)
            rec = dfg.win_rate.to_dict()
            return rec

        # Create an array of dates extending back to history delta
        if today is None:
            ending_exclusive = self.today
        else:
            ending_exclusive = pd.Timestamp(today)

        starting = ending_exclusive - self.history_delta
        dates = pd.date_range(starting, ending_exclusive, inclusive='left')

        # This will hold output
        rec_list = []

        # Create batch frame ending at each date
        for date in dates:
            interval_start = date - datetime.timedelta(days=interval_days)
            batch = df[df.close_date.between(interval_start, date)]
            rec = {'date': date}

            # Compute the win rate for that batch and save result
            rec.update(compute_win_rate(batch))
            rec_list.append(rec)

        # Create a frame out of the results
        dfout = pd.DataFrame(rec_list)

        # New stages won't have early values, so just back fill them
        dfout = dfout.fillna(method='bfill')

        # Flatten out results from bake_days ago to end
        dfout = dfout.set_index('date')
        dfout.loc[ending_exclusive - datetime.timedelta(days=bake_days):, :] = np.NaN
        dfout = dfout.fillna(method='ffill')
        return dfout

    def get_active_pipe_timeseries(self):
        df = self.loader.df_pipe
        df = df[['opportunity_id', 'market_segment', 'sql_date', 'close_date', 'acv']].copy()
        df = df[df.acv.notnull()]
        df = df[df.market_segment != 'unknown']

        dates = pd.date_range('1/1/2021', datetime.datetime.now())

        rec_list = []
        for date in dates:
            batch = df[(df.sql_date <= date) & (df.close_date > date)].copy()
            batch.loc[:, 'market_segment'] = [ezr.slugify(s) for s in batch.market_segment]
            rec = {'date': date}
            rec.update(batch.groupby(by='market_segment').acv.sum().to_dict())
            rec_list.append(rec)
        dfo = pd.DataFrame(rec_list).set_index('date')
        return dfo

    def get_mean_deal_size_timeseries(self, starting=None):
        now = fleming.floor(datetime.datetime.now(), day=1)

        if starting is None:
            then = pd.Timestamp('1/1/2020')
        else:
            then = pd.Timestamp(starting)

        dfo = self.op.df_orders[['account_id', 'market_segment', 'mrr', 'order_start_date', 'order_ends']]

        dates = pd.date_range(then, now, freq='D')
        rec_list = []
        for date in dates:
            batch = dfo[(dfo.order_start_date <= date) & (dfo.order_ends > date)].copy()
            batch.loc[:, 'market_segment'] = [ezr.slugify(s) for s in batch.market_segment]
            batch = batch.groupby(by=['account_id', 'market_segment'])[['mrr']].sum().reset_index()
            batch['arr'] = 12 * batch.mrr
            deal_size = batch.groupby(by='market_segment').arr.mean()
            rec = {'date': date}
            rec.update(deal_size.to_dict())
            rec_list.append(rec)
        df = pd.DataFrame(rec_list).set_index('date')
        return df


class TestPipeStats(TestCase):
    def test_nothing(self):
        print('testing nothing')


class OrderProducts(ezr.pickle_cache_mixin):

    GRACE_PERIOD_DAYS = 15
    EPOCH = pd.Timestamp('1/1/2020')

    ORDER_REPORT_ID = '00O4O0000049xifUAA'

    SEGMENT_MAPPING = {
        'Mid-Market': 'mid_market',
        'Tech Touch': 'smb',
        'Enterprise': 'enterprise',
        'SMB': 'smb',
        'Strategic': 'enterprise',
        'Unkown': 'unkown'
    }

    pkc = ezr.pickle_cache_state('reset')

    def __init__(self):
        self.today = fleming.floor(datetime.datetime.now())

    @ezr.pickle_cached_container()
    def df_order_products_raw(self):
        sfdc = ezr.SalesForceReport()
        df = sfdc.get_report(
            self.ORDER_REPORT_ID,
            slugify=True,
            date_fields=[
                'order_start_date',
                'order_ends',
                'last_modified_date',
                'created_date',
            ]
        )
        df = df.rename(columns={'sentiment': 'health_risk'})

        return df

    @ezr.cached_container
    def df_raw(self):
        """
        Just an alias to be consistent with other loaders
        """
        return self.df_order_products_raw

    @ezr.cached_container
    def df_order_products_all(self):
        df = self.df_order_products_raw

        # Only care about recurring revenue
        df = df[df.revenue_type == 'Recurring']

        # Completely ignore pilots
        df = df[df.pilot == 0]

        # Completely ignore "write offs"
        df = df[df.order_type != 'Write Off']

        # Only need these specific columns
        df = df[[
            'account_name',
            'account_id',
            'health_risk',
            'order_id',
            'order_type',
            'pilot',
            'termination_status',
            'order_start_date',
            'order_ends',
            'last_modified_date',
            'created_date',
            'months',
            'mrr',
            'payment_terms',
            'market_segment',
            'product_name',
            'unit_price',
            'quantity',
        ]]
        df.loc[:, 'termination_status'] = df.termination_status.fillna('Active')
        df.loc[:, 'health_risk'] = df.health_risk.fillna('Unkown')
        return df

    @ezr.cached_container
    def df_orders(self):
        df = self.df_order_products_all

        # Collapse order product down into orders by summing revenue
        df = df.groupby(
            by=[
                'account_name',
                'account_id',
                'health_risk',
                'order_id',
                'order_type',
                'pilot',
                'termination_status',
                'order_start_date',
                'order_ends',
                'last_modified_date',
                'created_date',
                'months',
                'payment_terms',
                'market_segment',
            ]
        ).sum().reset_index()

        # Uncommitted orders don't mean anything
        df = df[df.order_type != 'Uncommitted']

        # Change pilots to int
        df.loc[:, 'pilot'] = df.pilot.astype(int)

        return df

    def get_ndr_metrics(self, months=12, now=None):
        """
        Returns comparison metrics for the state of orders "now" compared to "months" ago.
        This is used to come up with NDR metrics

        Args:
            now: set "today" to this date (defaults to datetime.datetime.now())
            months: Compare "today" with this many months ago to compute retention stats

        In the description below we will use the term "reference time" to mean "months" prior to "now"

        market_segment:  The market segment for the metric
        value: The value of the metric
        variable: A tag identifying which metric it is.
            base: The ARR of as it existed at the reference time
            expanded: The total amount the logos contracted at reference time have
                    expanded with us since reference time
            reduced: The total amount logos contracted at reference time have
                    reduced with us since reference time
            churned: The amount of dollars that have churned from the logos contracted
                    at reference time
            renewed: The amount of dollars logos contracted at reference time have
                    renewed with us
                ndr: The comprehensive net-dollar retention including expansion, reduction
                    churn and renewal for all logos that were under contract at reference time
            *_pct: The percentage of base for each of the named metrics
        """
        if now is None:
            now = fleming.floor(datetime.datetime.now(), day=1)

        # Now is the date for which you want to compute metrics
        now = pd.Timestamp(now)

        # You will be comparing ARR between now and this many months ago to compute metrics
        then = now - relativedelta(months=months)

        # Get all orders and standardize them
        df = self.df_orders[['account_id', 'order_start_date', 'order_ends', 'mrr', 'market_segment']]
        df.loc[:, 'market_segment'] = [ezr.slugify(s) for s in df.market_segment]

        # Create two frames.  One for "now" and one for "then"
        df_now = df[(df.order_start_date <= now) & (df.order_ends >= now)]
        df_then = df[(df.order_start_date <= then) & (df.order_ends >= then)]

        # Combine all revenue for a given account into a single record
        def agg_by_account(df):
            df = df.groupby(by=['account_id', 'market_segment'])[['mrr']].sum().reset_index()
            return df

        df_now = agg_by_account(df_now)
        df_then = agg_by_account(df_then)

        # Join the accounts that existed "then" with what exists "now".  Fill revenue with 0 if they don't exist "now"
        dfj = pd.merge(
            df_then, df_now, on=['account_id', 'market_segment'], how='left', suffixes=['_ref', '_ret']).fillna(0)

        # This is the base from which we will compute all metrics
        dfj['base'] = dfj.mrr_ref

        # Exppanded revenue is any revenue over and above the base an org had "back then"
        dfj['expanded'] = np.maximum(0, dfj.mrr_ret - dfj.mrr_ref)

        # Reduction revenue is any deficit below the base that an org had "back then"
        dfj['reduced'] = np.maximum(0, dfj.mrr_ref - dfj.mrr_ret)

        # Renewed is the amount of revenue we have "now" that we also had "back then"
        dfj['renewed'] = np.minimum(dfj.mrr_ref, dfj.mrr_ret)

        # If there is no revenue "now", that means the or churned
        dfj['churned'] = [ref if int(round(ret)) == 0 else 0 for (ref, ret) in zip(dfj.mrr_ref, dfj.mrr_ret)]

        # Reduced included churned revenue in the way computed it.  So remove that churned revenue
        dfj.loc[:, 'reduced'] = dfj.reduced - dfj.churned

        # Sum all revenues by market segment and convert MRR to ARR
        dfg = dfj.drop(['account_id'], axis=1).groupby(by=['market_segment']).sum()
        dfg = dfg * 12

        # Add a fake new market segment of "all"
        dfg = dfg.T
        dfg['combined'] = dfg.sum(axis=1)
        dfg = dfg.T

        # Net dollar is looking at the sum of all companies we had under contract "then" and
        # looking at the percent increase/decrease to everything they are paying us now
        dfg['ndr'] = 100 * dfg.mrr_ret / dfg.mrr_ref

        # I'd also like to know more granular percentages with respect to the base
        for metric in ['expanded', 'reduced', 'renewed', 'churned']:
            dfg[f'{metric}_pct'] = 100 * dfg[metric] / dfg.base

        # Clean up some variables I don't care about
        dfg = dfg.drop(['mrr_ref', 'mrr_ret'], axis=1).reset_index()

        # Transform the data into a melted format and insert date
        dfg = pd.melt(dfg, id_vars=['market_segment'])
        return dfg

    def orders_to_events(self, df, limit_fields=True):
        df = df.copy()

        # Make a dataframe of all starting events
        df_start = df.copy()
        df_start['date'] = df_start.order_start_date
        df_start['event'] = 'order_created'

        # Make a dataframe of all ending events
        df_end = df.copy()
        df_end['date'] = df_end.order_ends
        df_end['event'] = 'order_expired'

        # Concat starting and ending to be one dataframe
        df = pd.concat([df_start, df_end], ignore_index=True, sort=False)

        # Limit the the fields I care about
        if limit_fields:
            df = df[[
                'event',
                'date',
                'account_name',
                'account_id',
                'health_risk',
                'order_id',
                'order_type',
                'pilot',
                'termination_status',
                'order_start_date',
                'order_ends',
                'last_modified_date',
                'created_date',
                'months',
                'mrr',
                'payment_terms',
                'market_segment',
            ]]

        # Ending orders need these fields negated
        neg_fields = [
            'mrr',
        ]

        # Negate appropriate fields for events of expiring non-pending orders
        ind = df.event[(df.event == 'order_expired')].index
        for field in neg_fields:
            df.loc[ind, field] = -1 * df.loc[ind, field]

        return df

    def _compute_day_diff(self, batch):
        """
        A utility method to use in pandas groupby for computing day diffs
        """
        batch.loc[:, 'day_delta'] = batch.date.diff().dt.days.fillna(0)
        return batch

    def _iterative_collapse(self, batch):
        """
        Collapse all dates within an interval to a single date. I need to account
        for the situation where an order is placed today.  A second order tomorrow,
        a third and fourth order the day after.  In this case, all four orders
        will get collapsed to the same date.  I need to iterate the collapsing procedure
        to catch multiple small intervals.
        """
        # Loop until convergence or timeout
        for nn in range(50):
            # Compute date diffs by taking differences between consecutive event dates
            batch.loc[:, 'day_delta'] = batch.date.diff().dt.days

            # Non-simultaneous event less than a grace period apart in time need to get collapsed
            needs_collapsing = ((batch.day_delta < self.GRACE_PERIOD_DAYS) & (batch.day_delta > 0))
            needs_collapsing = needs_collapsing[needs_collapsing]

            # If nothing needs collapsing we are done
            if needs_collapsing.empty:
                break

            # Get indexes needed to collapse the dates
            late_index = needs_collapsing.index
            early_index = pd.Index([v - 1 for v in late_index])

            # Collapse by setting late dates to early dates
            batch.loc[late_index, 'date'] = batch.loc[early_index, 'date'].values

        # If the for loop didn't exit because of a break, then collapsing didn't converge
        else:
            raise RuntimeError('Recursive collapsing timed out')

        return batch

    @ezr.cached_container
    def df_events(self):
        return self.get_df_events(collapse=True)

    def get_df_events(self, collapse=True, df=None):
        # Get the orders
        limit_fields = False
        if df is None:
            limit_fields = True
            df = self.df_orders

        # Turn them into events
        df = self.orders_to_events(df, limit_fields=limit_fields)

        # Sort the events by account and then by time
        df = df.sort_values(by=['account_name', 'account_id', 'date', 'event']).reset_index(drop=True)

        # This will be used to hold the number of days between consecutive events
        df.insert(2, 'day_delta', np.NaN)

        # Make sure index is contiguous integers
        df = df.reset_index(drop=True)

        # Collapse all events that occure within a grace period of one another
        if collapse:
            df = df.groupby(by='account_id').apply(self._iterative_collapse)

            # Patch up start/end dates to match with collapse
            ind = df[(df.event == 'order_created') & (df.date != df.order_start_date)].index
            df.loc[ind, 'order_start_date'] = df.loc[ind, 'date']

            ind = df[(df.event == 'order_expired') & (df.date != df.order_ends)].index
            df.loc[ind, 'order_ends'] = df.loc[ind, 'date']

        df = df.drop('day_delta', axis=1)

        return df

    @ezr.cached_container
    def df_changes(self):
        df = self.df_events

        # Don't consider any created order events except follow-ons, since these
        # Are the only created orders that will impact expanion/reduction
        # However, keep their expired events because that will contribute
        # We also want to pretend that pilots never existed.

        # ##########################################################################33
        # HERE IS A WEIRD SITUAION.  "UNCOMMITTED" ORDERS ARE CURRENTLY NOT SHOWING
        # UP FOR CS.  THIS HAS SOME WEIRD IMPLICATIONS.  IF A COMMITTED ORDER IS CREATED,
        # IT CURRENTLY IS FULLY COUNTING AGAINST CS.  I NEED TO PUT MORE THOUGHT INTO
        # HOW THIS IS BEING COMPUTED.
        # ##########################################################################33
        ignore_these = (df.event == 'order_created')
        ignore_these = ignore_these & (df.order_type != 'Follow-On')
        ignore_these = ignore_these | (df.pilot == 1)
        df = df[~ignore_these]

        df = df.sort_values(by=['account_name', 'account_id', 'date', ])

        def process_batch(batch):
            batch['mod_date'] = batch.last_modified_date.max()
            return batch

        # For each account, you care about the latest time any corresponding
        # order was modified
        df = df.groupby(by='account_id').apply(process_batch)

        def summarize_batch(batch):
            """
            This will summarize everything that happened during a given revenue-changing
            event.
            """
            mrr = batch.mrr.sum()
            mrr_eligible = -batch.mrr[batch.event == 'order_expired'].sum()
            has_churned = 'Churn' in batch.termination_status.values

            out = {
                'mrr': mrr,
                'outcome_status': np.NaN,
                'health_risk': batch.health_risk.iloc[0],
                'mrr_pending': np.NaN,
                'mrr_churned': np.NaN,
                'mrr_renewed': 0,
                'mrr_eligible': mrr_eligible,
                'has_churned': has_churned,
            }

            if 'Decision Pending' in batch.termination_status.values:
                cond = (batch.termination_status == 'Decision Pending') & (batch.event == 'order_expired')
                out['mrr_pending'] = -batch[cond].mrr.sum()
            elif 'Churn' in batch.termination_status.values:
                cond = (batch.termination_status == 'Churn') & (batch.event == 'order_expired')
                out['mrr_churned'] = -batch[cond].mrr.sum()

            # Compute the MRR after the event occures
            post_mrr = out['mrr_eligible'] + out['mrr']

            # Renewal and possible expansion/contraction happened if there is still mrr after the event
            if np.abs(post_mrr) > .01:
                # If straight-up renewal or expansion, the eligible got renewed
                if out['mrr'] >= 0:
                    out['mrr_renewed'] = out['mrr_eligible']
                # If a reduction, then we "renewed" the post_mrr
                else:
                    out['mrr_renewed'] = post_mrr

            return pd.Series(out)

        # Summarize everything that happened during a single revene event, filling nans when done and sorting
        df = df.groupby(by=['account_name', 'account_id', 'market_segment', 'date', 'mod_date']).apply(
            summarize_batch).reset_index()
        cols = ['mrr_pending', 'mrr_churned']
        df.loc[:, cols] = df.loc[:, cols].fillna(0)
        df = df.sort_values(by=['account_name', 'market_segment', 'account_id', 'date', ]).reset_index()

        # Only care aboute revenue events since the epoch
        df = df[df.date >= self.EPOCH]

        # Compute extra fields
        df['net_arr'] = df.mrr * 12
        df['arr_churned'] = df.mrr_churned * 12
        df['arr_pending'] = df.mrr_pending * 12
        df['arr_renewed'] = df.mrr_renewed * 12
        df['arr_eligible'] = df.mrr_eligible * 12

        # Make the default outcomes
        df['outcome'] = df.mrr.apply(np.sign).astype(int).map({-1: 'reduced', 0: 'renewed', 1: 'expanded'})

        # Correctly label the terminal events
        def label_terminal_event(batch):
            latest_event = batch.iloc[-1, :]
            if latest_event['date'] > self.today:
                latest_event['outcome'] = 'eligible'
                batch.iloc[-1, :] = latest_event
            elif latest_event['has_churned']:
                latest_event['outcome'] = 'churned'
                batch.iloc[-1, :] = latest_event
            return batch
        df = df.groupby(by=['account_name', 'market_segment', 'account_id']).apply(label_terminal_event)

        # Don't want future reduced or churned events marked as such until they actually happen
        ind = df[df.outcome.isin(['churned', 'reduced']) & (df.date > self.today)].index
        if len(ind) > 0:
            df.loc[ind, 'outcome'] = 'eligible'

        # Override pending outcomes for those that have pending mrr
        ind = df.date[df.mrr_pending.abs() > 0].index
        df.loc[ind, 'outcome'] = 'pending'

        # Compute expansions/reductions
        df['arr_reduced'] = -np.minimum(df.net_arr, 0) - df['arr_churned']
        df['arr_expanded'] = np.maximum(df.net_arr, 0)

        # Set order of output columns
        df = df[[
            'date',
            'account_name',
            'market_segment',
            'net_arr',
            'arr_eligible',
            'arr_renewed',
            'arr_expanded',
            'arr_reduced',
            'arr_churned',
            'arr_pending',
            'account_id',
            'health_risk',
            'outcome',
            'mod_date',
        ]]

        return df

    @ezr.cached_container
    def df_starting(self):
        """
        Gets a frame of starting revenue
        """
        # Based on events
        df = self.get_df_events(collapse=True)

        # Convert to ARR
        df['arr'] = df.mrr * 12

        # Collapse all events that happened on a day to a single entry
        df = df.groupby(by='date')[['arr']].sum().sort_index().reset_index()

        # Take the cumulative sum of all arr.  ARR is actually arr-delta in this frame
        df.loc[:, 'arr'] = df.arr.cumsum()

        # Find start/end dates floored to month
        starting = fleming.floor(self.EPOCH, month=1)
        ending = fleming.floor(datetime.datetime.now(), month=1)

        # Create a frame of just month epochs
        dfm = pd.DataFrame({'date': pd.date_range(starting, ending, freq='MS', closed='left').values})

        # Perform an "as_of" merge of cumulative ARR with month epochs.
        dfj = pd.merge_asof(dfm, df, on='date', allow_exact_matches=True)
        return dfj

    @ezr.cached_container
    def df_starting_segment(self):
        return self.get_df_starting_segment()

    def get_df_starting_segment(self, include_current_period=False, pandas_frequency='MS', **kwargs):
        """
        Gets a frame of starting revenue broken down by market segment
        """
        if 'include_current_month' in kwargs:
            warnings.warn('include_current_month deprecated.  All the cool kids use include_current_period')
            include_current_period = kwargs['include_current_month']

        # Based on events
        df = self.get_df_events(collapse=True)

        # Convert to ARR
        df['arr'] = df.mrr * 12

        # Collapse all events of the same kind that happened on a day to a single entry
        df = df.groupby(by=['market_segment', 'date'])[['arr']].sum().sort_index().reset_index()

        starting = self.EPOCH
        ending = fleming.floor(datetime.datetime.now(), day=1)

        if pandas_frequency != 'D':
            # Find start/end dates floored to month
            starting = fleming.floor(starting, month=1)
            ending = fleming.floor(ending, month=1)

        # Create a frame of just month epochs
        closed = None if include_current_period else 'left'
        dfm = pd.DataFrame({'date': pd.date_range(starting, ending, freq=pandas_frequency, closed=closed).values})

        # Function to get ARR of segment for relevant dates
        def agg(batch):
            batch.loc[:, 'arr'] = batch.arr.cumsum()
            batch = pd.merge_asof(dfm, batch, on='date', allow_exact_matches=True)
            batch = batch.set_index('date')

            return batch

        # Take the cumulative sum of all arr for each segment.  ARR is actually arr-delta in this frame
        df = df.groupby(by=['market_segment']).apply(agg).drop('market_segment', axis=1)

        # Unstack to get starting by market_segment
        df = df.unstack('market_segment').fillna(0)
        df.columns = list(df.columns.get_level_values(1))
        df['total'] = df.sum(axis=1)
        df = df.reset_index()
        return df


class AccountLoader(ezr.pickle_cache_mixin):
    pkc = ezr.pickle_cache_state('reset')

    @ezr.pickle_cached_container()
    def df_raw(self):
        sfdc = ezr.SalesForceReport()
        df = sfdc.get_report(
            '00O4O000004AjoX',
            slugify=True,
            date_fields=[
                'last_modified_date',
                'date_account_was_round_robined',
                'last_sales_activity_date',
            ])
        return df

    @ezr.cached_container
    def df(self):
        return self.df_raw


if __name__ == '__main__':
    #############################################################################################
    # TO RUN THESE TEST:
    #     coverage erase &&  coverage run --source='.'  test_predictor.py -v && coverage report
    #     python -m unittest live_opps.TestNothing.test_nothing -v   # for a single test
    #############################################################################################

    main()
