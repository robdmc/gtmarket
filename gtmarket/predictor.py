from copy import deepcopy
import datetime
import itertools
import json
import warnings

from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
import easier as ezr
import fleming
from .core import OppLoader, PipeStats, OrderProducts

USE_PG = True


# Utility function to make sure to blobs are equal
def equal(v1, v2):
    if type(v1) != type(v2):
        raise ValueError(f'Bad type for v1={v1}')
    if isinstance(v1, dict):
        for key, val1 in v1.items():
            val2 = v2[key]
            equal(val1, val2)

        for key, val2 in v2.items():
            val1 = v1[key]
            equal(val1, val2)

    if isinstance(v1, list):
        if len(v1) != len(v2):
            raise ValueError(f'bad length for v1={v1}')

        for a, b in zip(v1, v2):
            equal(a, b)
    if v1 != v2:
        raise ValueError(f'unequal {v1} != {v2}')


def spread_values(y, spread_days):
    """
    Uses convolution to spread out contributions over a time window.
    """
    import numpy as np
    import pandas as pd
    from scipy.signal import convolve

    # Determine the length of the array
    ylen = len(y)
    ylen_orig = ylen

    # Save the index and convert to array if this is a pandas series
    is_series = isinstance(y, pd.Series)
    if is_series:
        ind = y.index
        y = y.values

    # If the array is less then the length of days to convert, pad it with zeros
    if ylen < spread_days:
        missing_days = spread_days - ylen + 1
        missing_pad = np.zeros(missing_days)
        y = np.concatenate((y, missing_pad))
        ylen = len(y)

    # Start with a zero mask with the same length as the array
    m = np.zeros(ylen)

    # Make a constant hazard spreading function
    m[:spread_days] = np.ones(spread_days)

    # Normalize the spreading function
    m = m / np.sum(m)

    # Create a zero pad to avoid fft roll-around
    p = np.zeros_like(y)
    m = np.concatenate((m, p))
    y = np.concatenate((y, p))

    # Run the scipy convolution of mask on data
    c = convolve(y, m, mode='full')

    # Limit to the original length of the data
    c = c[:ylen_orig]

    # Attach original index if a series was passed
    if is_series:
        c = pd.Series(c, index=ind)

    # Return answer
    return c


class DiscountedSalesExpansion:
    def __init__(self, model_params):
        self.mp = model_params
        self.loader = OppLoader()
        self.op = OrderProducts()

    @ezr.cached_container
    def _df_ungrouped_expansion_opps(self):
        today = fleming.floor(datetime.datetime.now(), day=1)
        df = self.loader.df_all
        df = df[df.status_code == 0]
        df = df[df.close_date >= today]
        df = df[df.type == 'Sales Expansion']

        if df.empty:
            return df

        df = df[['account_id', 'close_date', 'acv', 'market_segment']]
        df = df[df.acv.notnull()]
        return df

    @ezr.cached_container
    def _account_ids_with_expansion_opps(self):
        df = self._df_ungrouped_expansion_opps
        return set(df.account_id)

    @ezr.cached_container
    def df_discounted_existing_expansion_opps(self):
        import pandas as pd
        # Get a reference to all market segments to use as column names
        columns = list(self.mp.segment_allocation.keys())

        # Get the expansion opps
        df = self._df_ungrouped_expansion_opps

        # Return empty frame with proper columns if no expansion opps
        if df.empty:
            return pd.DataFrame(columns=columns)

        # Group by date and market segment
        df = df.groupby(by=['close_date', 'market_segment'])[['acv']].sum().unstack('market_segment')

        # Make sure frame has proper columns
        df.columns = df.columns.get_level_values(1)
        df = df.reindex(columns, axis=1)
        df.index.name = 'date'

        # Apply the win rate for expansion opps
        df = df * self.mp.sales_expansion_win_rate
        return df

    @ezr.cached_container
    def df_discounted_eligible(self):
        import pandas as pd

        # Get a frame of all orders
        dfo = self.op.df_orders

        # Only care about the acv of new business orders properly marked with market segment
        dfo = dfo[dfo.order_type == 'New Business']
        dfo['acv'] = 12 * dfo.mrr
        dfo.loc[:, 'market_segment'] = [ezr.slugify(s) for s in dfo.market_segment]

        # Compute the discounted expansion acv from the order acv
        dfo['expansion_acv'] = self.mp.sales_expansion_eligible_acv_frac * dfo.acv

        dfo = dfo[['account_id', 'order_start_date', 'expansion_acv', 'market_segment']]

        dfo = dfo.groupby(by=['order_start_date', 'market_segment'])[['expansion_acv']].sum().unstack('market_segment')
        dfo.columns = dfo.columns.get_level_values(1)
        ind = pd.date_range(dfo.index[0], datetime.datetime.now() + relativedelta(years=1))
        dfo = dfo.reindex(ind).fillna(0)

        for col in dfo.columns:
            days = self.mp.sales_expansion_eligibility_days[col]
            dfo.loc[:, col] = spread_values(dfo[col], days)

        dfo = dfo.cumsum()

        return dfo


class ModelParams(ezr.BlobMixin):
    # Don't delete blob versions here. Just comment out and add new
    # so that there is an easily visible history of versions
    # blob_version = ezr.BlobAttr('2022-02-04.01')
    # blob_version = ezr.BlobAttr('2022-04-22.01')
    blob_version = ezr.BlobAttr('2022-05-19.01')

    rep_sal_rate_per_day = ezr.BlobAttr(10 / 30.3)
    ramp_time_constant_days = ezr.BlobAttr(13 * 7)

    hiring_plan = ezr.BlobAttr([
        {'date': parse('1/1/2018'), 'num_hires': 8 + 2.2},
        {'date': parse('11/15/2021'), 'num_hires': 5},
        {'date': parse('11/29/2021'), 'num_hires': 4},
        {'date': parse('2/1/2022'), 'num_hires': 3},
    ])

    # Constructor overrides this to be "today"
    fit_date = ezr.BlobAttr(parse('1/1/1999'))

    segment_allocation = ezr.BlobAttr({
        'enterprise': .45,
        'commercial': .40,
        'velocity': .15,
    })

    segment_days_to_win = ezr.BlobAttr({
        'velocity': 42,
        'commercial': 97,
        'enterprise': 136,

    })

    segment_deal_size = ezr.BlobAttr({
        'velocity': 15177,
        'commercial': 27465,
        'enterprise': 70710,

    })

    segment_win_rate = ezr.BlobAttr({
        'commercial': 0.058,
        'enterprise': 0.032,
        'velocity': 0.052,
    })

    stage_win_rate = ezr.BlobAttr({
        'SAL': .04,
        'Discovery': .08,
        'Demo': .12,
        'Proposal': .45,
        'Negotiation': .7,
    })

    # The rate at which we win opened expansion opps
    sales_expansion_win_rate = ezr.BlobAttr(.8)

    # The fraction of eligible ACV that gets sales-expanded
    sales_expansion_eligible_acv_frac = ezr.BlobAttr(.1)

    sales_expansion_eligibility_days = ezr.BlobAttr({
        'velocity': 90,
        'commercial': 90,
        'enterprise': 365,
    })

    existing_pipe_model = ezr.BlobAttr({
        'commercial': {},
        'enterprise': {},
        'velocity': {},
    })

    existing_pipe_model_with_expansion = ezr.BlobAttr({
        'commercial': {},
        'enterprise': {},
        'velocity': {},
    })

    existing_stage_stats = ezr.BlobAttr({
        key: {
            'SAL': 0,
            'Discovery': 0,
            'Demo': 0,
            'Proposal': 0,
            'Negotiation': 0,
        } for key in ['acv', 'discounted_acv', 'num_opps']
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_segment_allocation = self.segment_allocation
        if self.fit_date < parse('1/1/2000'):
            self.fit_date = datetime.datetime.now()

    def __str__(self):
        return f'ModelParams(fit_date={self.fit_date.date()})'

    def __repr__(self):
        return self.__str__()

    def _attribute_from_timeseries(self, attribute_name, df):
        ser = df.iloc[-1, :]
        att = getattr(self, attribute_name)
        for key in att.keys():
            att[key] = ser[key]
        return self

    def _fit_segment_days_to_win(self, pipe_stats_obj):
        df = pipe_stats_obj.get_conversion_timeseries('sal2won_time', interval_days=360, bake_days=60)
        self._attribute_from_timeseries('segment_days_to_win', df)

    def _fit_segment_win_rate(self, pipe_stats_obj):
        df = pipe_stats_obj.get_conversion_timeseries('sal2won_opps', interval_days=360, bake_days=60)
        self._attribute_from_timeseries('segment_win_rate', df)

    def _fit_stage_win_rate(self, pipe_stats_obj):
        df = pipe_stats_obj.get_stage_win_rates_timeseries(interval_days=360, bake_days=60)
        self._attribute_from_timeseries('stage_win_rate', df)

    def _fit_segment_deal_size(self, pipestats_obj):
        df = pipestats_obj.get_mean_deal_size_timeseries(starting=pipestats_obj.today - relativedelta(days=5))
        self.segment_deal_size = df.iloc[-1, :].to_dict()

    def _fit_existing_pipe_model_and_stage_stats(
            self, pipe_stats_obj, model_attribute_name, populate_stage_stats=True, include_discounted_eligible=False):
        import numpy as np
        import pandas as pd
        if model_attribute_name not in ['existing_pipe_model', 'existing_pipe_model_with_expansion']:
            raise ValueError('Bad pipe model name')

        # Get all new biz opps with known market seg,ents
        df = pipe_stats_obj.loader.df_new_biz
        df = df[df.market_segment != 'unknown']

        # Limit to only open opps
        df = df[df.status_code == 0]

        # If there are any legacy stage names, translate them
        df.loc[:, 'stage'] = pipe_stats_obj.translate_stage_in_frame(df, 'stage')

        # Get a frame of pre-pipe opps and set their ACV to segment-based avg deal size
        # NOTE: This object better have the properly fit segment deal size computed
        df_pre = df[df.stage.isin(['SAL', 'Discovery'])].copy()
        df_pre.loc[:, 'acv'] = df_pre.market_segment.map(self.segment_deal_size)

        # Overwrite any ACV for pre-pipe opps with their segment aveage
        df.loc[df_pre.index, 'acv'] = df_pre.acv
        df = df[['close_date', 'acv', 'stage', 'market_segment']]

        # Use a stage-based win rate to compute a discounted ACV for each opp
        df['win_rate'] = df.stage.map(self.stage_win_rate)
        df['discounted_acv'] = df.win_rate * df.acv

        # Compute the acv, discounted acv and number of opps in each stage
        dfg = df.groupby(by='stage').aggregate({'acv': [np.sum, len], 'discounted_acv': [np.sum]})
        dfg.columns = ['_'.join(t) for t in dfg.columns]
        dfg = dfg.rename(columns={'acv_sum': 'acv', 'acv_len': 'opps', 'discounted_acv_sum': 'discounted_acv'})

        if populate_stage_stats:
            self.existing_stage_stats = {
                'acv': dfg.acv.to_dict(),
                'discounted_acv': dfg.discounted_acv.to_dict(),
                'num_opps': dfg.opps.to_dict()
            }

        # Force past close dates into the future (this happens with bad rep hygiene)
        df_past = df[df.close_date < pipe_stats_obj.today]
        if not df_past.empty:
            df.loc[df_past.index, 'close_date'] = pipe_stats_obj.today + relativedelta(days=90)

        # I just don't believe close dates more than a year into the future
        df = df[df.close_date <= pipe_stats_obj.today + relativedelta(years=1)]

        # Find the total amount expected to be one in each segment for each day
        # I deliberately leave NaNs for missing days
        df = df.groupby(by=['close_date', 'market_segment'])[['discounted_acv']].sum().unstack('market_segment')
        df.columns = df.columns.get_level_values(1)
        df.index.name = 'date'

        # If requested add fake ACV to the new business pipeline that reflects what we expect to win
        # from expansion opps and from new-business contracts still eligible for sales-expansion
        if include_discounted_eligible:
            df_list = [df.reset_index()]
            expansion_obj = DiscountedSalesExpansion(self)

            for attr in ['df_discounted_eligible', 'df_discounted_existing_expansion_opps']:
                dfa = getattr(expansion_obj, attr)
                if not dfa.empty:
                    df_list.append(dfa.reset_index())

            dfc = pd.concat(df_list, axis=0, ignore_index=True, sort=False)
            df = dfc.groupby(by='date').sum().sort_index()

        # I will need a reference to "yesterday" in the fitting
        yesterday = pipe_stats_obj.today - relativedelta(days=1)

        # Run a Bernstein fitter over the expected revenue for each segment
        existing_pipe_model = {}
        for seg in df.columns:
            # Get rid of nans
            y = df[seg].dropna()

            # Outliers cause the fitter to freak out, so ignore a slice of upper percentiles
            x = (y.index - np.min(y.index)).days
            ind = x < np.percentile(x, 90)
            y = y[ind]
            x = x[ind]

            y = y.resample('D').asfreq().fillna(0)
            y[yesterday] = 0
            y = y.sort_index()

            y = y.cumsum()
            x = (y.index - y.index[0]).days

            x = x.values
            y = y.values

            fitter = ezr.BernsteinFitter()
            fitter.fit(x, y, degree=10)
            existing_pipe_model[seg] = fitter.to_blob()

            # Debug code for plotting fits
            if False:
                import holoviews as hv
                yf = fitter.predict(x)
                display(hv.Curve((x, y), label=seg) * hv.Curve((x, yf), label='fit'))  # noqa

        setattr(self, model_attribute_name, existing_pipe_model)

    def _fit_segment_allocation(self, pipe_stats_obj):
        new_alloc = pipe_stats_obj.df_all_opp_segment_allocation.segment_allocation.to_dict()
        self.segment_allocation = new_alloc

    def fit(self):
        # Most of the stats are for new business only opps excluding pilot and expansion
        ps = PipeStats()
        self._fit_segment_allocation(ps)
        self._fit_segment_days_to_win(ps)
        self._fit_segment_win_rate(ps)
        self._fit_stage_win_rate(ps)
        self._fit_segment_deal_size(ps)

        # Fit an existing pipe model (including stage stats) from actual new business only
        self._fit_existing_pipe_model_and_stage_stats(
            ps, 'existing_pipe_model', populate_stage_stats=True, include_discounted_eligible=False)

        # Now fit a pipe model that includes sales expansion and pilots.  Don't however update the stage stats.
        ps = PipeStats(pilots_are_new_biz=True, sales_expansion_are_new_biz=True)
        self._fit_existing_pipe_model_and_stage_stats(
            ps, 'existing_pipe_model_with_expansion', populate_stage_stats=False, include_discounted_eligible=True)
        self.fit_date = datetime.datetime.now()

    def to_blob(self):
        import numpy as np
        # Run the standard blob creator to get initial blob
        blob = super().to_blob()

        # Manually change dates to what I want
        for entry in blob['hiring_plan']:
            entry['date'] = str(entry['date'].date())

        blob['fit_date'] = str(blob['fit_date'].date())

        # json.dumps doesn't like numpy types.  This is a dict that maps
        # unkown types to a lambda that will convert them to known types
        type_map = {
            np.int64: lambda v: int(v),
            np.float64: lambda v: float(v),
        }

        # An inner function that knows how to transform types
        def mapper(val):
            # If this is a dictionary, apply the mapper to all of it's values recursively
            if isinstance(val, dict):
                for k, v in val.items():
                    val[k] = mapper(v)
            # If this is a list, apply the mapper to all of it's values recursively
            if isinstance(val, list):
                for ind, v in enumerate(val):
                    val[ind] = mapper(v)

            # If not a list or a dict, search the type_map to see if this value
            # is of a type that needs translating, and if it is, do the translation
            for kind, func in type_map.items():
                if isinstance(val, kind):
                    val = func(val)

            # return the transformed value
            return val

        # Run the type mapper of the blob to make sure json will like it
        blob = mapper(blob)
        return blob

    def from_blob(self, blob, strict=False):
        blob = deepcopy(blob)
        for entry in blob['hiring_plan']:
            entry['date'] = parse(entry['date'])
        blob['fit_date'] = parse(blob.get('fit_date', '1/1/1999'))
        super().from_blob(blob, strict=strict)
        return self

    def to_json(self):
        blob = self.to_blob()
        blob = self._serialize_keys(blob)
        return json.dumps(blob)

    def from_json(self, serialized, strict=False):
        blob = json.loads(serialized)
        blob = self._deserialize_keys(blob)
        self.from_blob(blob, strict=strict)
        return self

    def copy(self):
        mp = ModelParams()
        mp.from_blob(self.to_blob())
        return mp

    def _serialize_keys(self, blob):
        import numpy as np
        new_blob = {}
        for key, val in blob.items():
            str_key = json.dumps(key)
            if isinstance(val, dict):
                new_blob[str_key] = self._serialize_keys(val)
            if isinstance(val, np.int64):
                val = int(val)
            else:
                new_blob[str_key] = val
        return new_blob

    def _deserialize_keys(self, blob):
        new_blob = {}
        for key, val in blob.items():
            try:
                new_key = json.loads(key)
            except json.JSONDecodeError:
                new_key = key
            if isinstance(new_key, list):
                new_key = tuple(new_key)
            if isinstance(val, dict):
                new_blob[new_key] = self._deserialize_keys(val)
            else:
                new_blob[new_key] = val
        return new_blob


class ModelParamsHist:
    DEFAULT_HISTORY_FILE = (
        '/Users/rob/Dropbox/ambition_stuff/sales_predictor_param_history/param_history2.sqlite'
    )

    def __init__(self, sqlite_file=None, use_pg=USE_PG):
        if sqlite_file is None:
            sqlite_file = self.DEFAULT_HISTORY_FILE
        self.sqlite_file = sqlite_file
        self.use_pg = use_pg

    def get_mini_model(self):
        if self.use_pg:
            return ezr.MiniModelPG(overwrite=False, read_only=False)
        else:
            return ezr.MiniModelSqlite(self.sqlite_file)

    def store(self, model_params_obj):
        import pandas as pd
        now = datetime.datetime.now()
        utc_seconds = int(ezr.pandas_time_to_utc_seconds(pd.Series([now])).iloc[0])
        data = model_params_obj.to_json()
        cols = {
            'utc_seconds': [utc_seconds],
            'data': [data],
        }
        df = pd.DataFrame(cols)
        mm = self.get_mini_model()
        mm.upsert('model_params', ['utc_seconds'], df)

        return self

    @ezr.cached_container
    def _df_history(self):
        mm = self.get_mini_model()
        df = mm.tables.model_params.df
        return df

    def get_history(self):
        df = self._df_history
        if not df.empty:
            df['time'] = ezr.pandas_utc_seconds_to_time(df.utc_seconds)
        return df

    def get_latest(self, as_of=None, strict=False, use_last_of_day=True):
        import pandas as pd

        if as_of is None:
            as_of = datetime.datetime.now()

        as_of = pd.Timestamp(as_of)

        if use_last_of_day:
            # Set "as_of" to be the final second of the day
            as_of_day = fleming.floor(as_of, day=1)
            as_of = as_of_day + relativedelta(days=1) - datetime.timedelta(seconds=1)

        # df = self.get_history()
        df = self.get_history()
        if as_of < df.time.min():
            raise ValueError(f'Can only go back to {df.time.min().date()}')
        df = df[df.time <= as_of]
        mp = ModelParams()
        mp.from_json(df.data.iloc[-1], strict=strict)

        delta = abs((as_of - pd.Timestamp(mp.fit_date)).days)
        if delta > 7:
            warnings.warn(
                f'Greater than 7 days gap between as_of={as_of.date()} and latest_fit_time={mp.fit_date.date()}')

        return mp


class SDRTeam:
    def __init__(self, model_params=None, model_params_hist=None, use_default_segment_allocation=False, use_pg=USE_PG):
        self._supplied_model_params_hist = model_params_hist
        if model_params is None:
            model_params = self.model_params_hist.get_latest()
        self.params = deepcopy(model_params)
        if use_default_segment_allocation:
            self.params.segment_allocation = self.params.default_segment_allocation
        self.start_date = min([r['date'] for r in self.hiring_plan])
        self.naive_start_date = self.start_date.replace(tzinfo=None)
        self.use_pg = use_pg

    @ezr.cached_property
    def model_params_hist(self):
        if self._supplied_model_params_hist:
            return self._supplied_model_params_hist
        else:
            return ModelParamsHist(use_pg=self.use_pg)



    @property
    def hiring_plan(self):
        return self.params.hiring_plan

    @property
    def rep_sal_rate_per_day(self):
        return self.params.rep_sal_rate_per_day

    @property
    def ramp_time_constant_days(self):
        return self.params.ramp_time_constant_days

    def get_rep_time_series(self, starting, ending):
        import pandas as pd
        df = pd.DataFrame(self.hiring_plan).set_index('date').cumsum()
        minned_starting = min(pd.Timestamp(starting), min(df.index))
        dates = pd.date_range(minned_starting, ending)
        df = df.reindex(dates, method='ffill')
        df = df.loc[starting:ending, :]
        df = df.rename(columns={'num_hires': 'num_reps'})
        return df

    def get_sal_rate_function(self, starting_date=None):
        import numpy as np

        def component_factory(component_start_time, delta_reps):
            tk = component_start_time
            ds = delta_reps

            def f(t):
                tau = self.ramp_time_constant_days
                return self.rep_sal_rate_per_day * ds * (t >= tk) * (1 - np.exp(-(t - tk) / tau))

            return f

        func_components = []
        for rec in self.hiring_plan:
            tk = float((rec['date'] - self.start_date).days)
            ds = float(rec['num_hires'])
            func_components.append(component_factory(tk, ds))

        if starting_date is None:
            delta_t = 0
        else:
            delta_t = (starting_date - self.start_date).days

        def gamma(t):
            t = t + delta_t
            g = sum([f(t) for f in func_components])
            return g

        return gamma

    def get_sal_rate_time_series(self, starting, ending):
        import pandas as pd
        dates = pd.Series(pd.date_range(starting, ending).values)
        t = (dates - self.naive_start_date).dt.days
        rates = self.get_sal_rate_function()(t.values)
        df = pd.DataFrame({
            'date': dates,
            'time': t,
            'sals_per_day': rates,
        })
        df = df.set_index('date')
        df = df.join(self.get_rep_time_series(starting, ending))
        return df

    def plot(self, starting=None, ending=None, unit='day'):  # pragma: no cover This is not accuracy critical
        import easier as ezr
        import fleming
        from dateutil.relativedelta import relativedelta
        lookup = {
            'day': 1,
            'week': 7,
            'month': 30.4,
            'quarter': 91.25
        }
        if unit not in lookup:
            raise ValueError(f'unit must be in {list(lookup.keys())}')
        if starting is None:
            starting = fleming.floor(datetime.datetime.now(), day=1)
        if ending is None:
            ending = starting + relativedelta(years=1)

        days = lookup[unit]

        df = self.get_sal_rate_time_series(starting, ending)
        ax1 = ezr.figure()
        plot1 = ax1.plot(df.index, df.sals_per_day * days, ezr.cc[0], label=f'SALS Per {unit.title()}', alpha=.8)
        ax2 = ax1.twinx()
        plot2 = ax2.plot(df.index, df.num_reps, ezr.cc[1], label='Num Reps', alpha=.8)
        ax1.set_ylabel(f'Sals Per {unit.title()}', color=ezr.cc[0])
        for t in ax1.get_yticklabels():
            t.set_color(ezr.cc[0])
        ax2.set_ylabel('Num Reps', color=ezr.cc[1])
        for t in ax2.get_yticklabels():
            t.set_color(ezr.cc[1])

        plots = plot1 + plot2
        labels = [p.get_label() for p in plots]
        ax1.legend(plots, labels, loc=4)
        ezr.date_formatter(ax2)
        return ax1, ax2


class Deals:
    def __init__(
            self,
            starting=None,
            ending_exclusive=None,
            include_sales_expansion=True,
            model_params=None,
            use_pg=USE_PG,
            model_params_hist=None):
        import pandas as pd
        today = fleming.floor(datetime.datetime.now(), day=1)
        next_year = today + relativedelta(years=1)
        if starting is None:
            starting = today
        if ending_exclusive is None:
            ending_exclusive = next_year

        self.starting = pd.Timestamp(starting)
        self.ending_exclusive = pd.Timestamp(ending_exclusive)
        self._supplied_model_params = model_params
        self._supplied_model_params_hist = model_params_hist
        self.include_sales_expansion = include_sales_expansion
        self.use_pg = use_pg

    @ezr.cached_property
    def model_params_hist(self):
        if self._supplied_model_params_hist:
            return self._supplied_model_params_hist
        else:
            return ModelParamsHist(use_pg=self.use_pg)

    @ezr.cached_property
    def model_params(self):
        if self._supplied_model_params is not None:
            mp = self._supplied_model_params
        else:
            mp = self.model_params_hist.get_latest(as_of=self.starting)

        return mp

    def _get_expected_revenue_from_current_pipe(self):
        import pandas as pd
        import numpy as np
        ending_inclusive = self.ending_exclusive - relativedelta(days=1)

        # Get the latest saved model_params as of the starting date
        mp = self.model_params

        # Get a string of dates you care about anchored to when model was fit
        dates = pd.date_range(mp.fit_date, ending_inclusive)

        # Loop over the parameters for existing pipe model and generate predictions
        # into the future
        col_dict = {}
        if self.include_sales_expansion:
            model = mp.existing_pipe_model_with_expansion
        else:
            model = mp.existing_pipe_model
        for key, blob in model.items():
            x = np.arange(len(dates))
            fitter = ezr.BernsteinFitter()
            fitter.from_blob(blob)
            col_dict[key] = fitter.predict(x)
        df = pd.DataFrame(col_dict, index=dates)

        # Compute total pipe
        df['total_current_pipe'] = df.sum(axis=1)

        # Chop off any early predictions and renormalize
        if self.starting > df.index[0]:
            baseline_row = df.loc[self.starting - relativedelta(days=1), :]
            df = df - baseline_row

        # Limit to the requested dates
        df = df.loc[self.starting:ending_inclusive, :]
        return df

    def _get_creation_functions(self):
        creation_function_list = []
        sdr = SDRTeam(self.model_params, use_pg=self.use_pg, model_params_hist=self.model_params_hist)

        def non_zero_factory(w, alloc):
            def gamma_w(t):
                g = w * alloc * sdr.get_sal_rate_function(starting_date=self.starting)(t)
                return g
            return gamma_w

        def zero_factory():
            def gamma_w(t):
                return 0
            return gamma_w

        for (seg, metric) in self._y_index:
            w = self.win_rates[seg]
            alloc = self.segment_allocation[seg]

            if metric == 'N':
                gamma_w = non_zero_factory(w, alloc)
            else:
                gamma_w = zero_factory()

            creation_function_list.append(gamma_w)
        return creation_function_list

    @ezr.cached_container
    def _decay_matrix(self):
        import numpy as np
        from scipy import linalg
        block_list = []
        for seg in self.segments:
            days_to_win = self.days_to_win[seg]
            block = np.array([[-1 / days_to_win, 0], [1 / days_to_win, 0]])
            block_list.append(block)
        M = linalg.block_diag(*block_list)
        return M

    @ezr.cached_container
    def segments(self):
        return list(self.model_params.segment_allocation.keys())

    @ezr.cached_container
    def days_to_win(self):
        return self.model_params.segment_days_to_win

    @ezr.cached_container
    def win_rates(self):
        return self.model_params.segment_win_rate

    @ezr.cached_container
    def segment_allocation(self):
        return self.model_params.segment_allocation

    @ezr.cached_container
    def mean_deal_size(self):
        return self.model_params.segment_deal_size

    @ezr.cached_container
    def _y_index(self):
        tups = list(itertools.product(self.segments, ['N', 'D']))
        return tups

    def _get_future_pipe_times_series_frames(self):
        """
        Returns two dataframes (df, and df_rev)
        df contains information about the number of opps
        df_rev contains information about the number of deals

        Both of these frames contain numbers for open opps (suffix _n) and
        won deals (suffix _d).
        """
        import numpy as np
        import pandas as pd
        from scipy.integrate import solve_ivp
        starting, ending = self.starting, self.ending_exclusive
        # ending = ending - relativedelta(days=1)
        # days = (ending - self.sdr_team.start_date).days
        num_days = (ending - starting).days

        def dNdt(t, Y):
            creation_vec = np.array([f(t) for f in self._get_creation_functions()])
            creation_vec = creation_vec.reshape(len(creation_vec), -1)
            decay_matrix = self._decay_matrix

            dn_dt = creation_vec + decay_matrix @ Y.reshape(len(Y), -1)
            return dn_dt.flatten()

        res = solve_ivp(
            dNdt, (0, num_days), np.zeros(len(self._y_index)), vectorized=False, t_eval=np.arange(0, num_days))

        df = pd.DataFrame(res.y.T, columns=[ezr.slugify(str(t)) for t in self._y_index])
        df['date'] = [starting + datetime.timedelta(days=int(d)) for d in res.t]
        df = df.set_index('date')

        df_rev = df.copy()

        for col in df_rev.columns:
            seg = col.split('_')[0]
            df_rev.loc[:, col] = self.mean_deal_size[seg] * df_rev.loc[:, col]

        return df, df_rev

    @ezr.cached_container
    def df_revenue_from_future_pipe(self):
        df = self._get_future_pipe_times_series_frames()[1]
        df = df[list(c for c in df.columns if c.endswith('_d'))]
        df = df.rename(columns={c: c.split('_')[0] for c in df.columns})
        return df

    @ezr.cached_container
    def df_revenue_from_current_pipe(self):
        return self._get_expected_revenue_from_current_pipe()

    @ezr.cached_container
    def df_deals_from_future_pipe(self):
        df = self._get_future_pipe_times_series_frames()[0]
        df = df[list(c for c in df.columns if c.endswith('_d'))]
        df = df.rename(columns={c: c.split('_')[0] for c in df.columns})
        return df

    @ezr.cached_container
    def df_predicted(self):
        # This includes both new business opps and sales expansion from existing orders
        dfc = self.df_revenue_from_current_pipe.drop('total_current_pipe', axis=1)

        # This includes revenue from pipe we have yet to create
        dff = self.df_revenue_from_future_pipe

        # This includes sales expansion we expect by closing future deals
        dffe = self.df_sales_expansion_revenue_from_future_pipe
        df = (dff + dfc + dffe).round()
        return df

    @ezr.cached_container
    def df_sales_expansion_revenue_from_future_pipe(self):
        # Get the future sales pipe
        df = self._get_future_pipe_times_series_frames()[1]
        df = df[list(c for c in df.columns if c.endswith('_d'))]
        df = df.rename(columns={c: c.split('_')[0] for c in df.columns})

        # Diff it to obtain sales for each day
        df = df.diff().fillna(0)

        # Estimate how much each day's revenue will contribute to sales expansion
        dfe = df * self.model_params.sales_expansion_eligible_acv_frac

        for col in df.columns:
            spread_days = self.model_params.sales_expansion_eligibility_days[col]
            dfe.loc[:, col] = spread_values(dfe.loc[:, col], spread_days)

        dfo = dfe.cumsum()
        return dfo
