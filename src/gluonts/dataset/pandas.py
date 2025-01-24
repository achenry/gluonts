# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations

import logging
from dataclasses import dataclass, field, InitVar
from typing import Any, Iterable, Optional, Type, Union, get_args

import numpy as np
import pandas as pd
import torch

import polars as pl
import polars.polars as plr
import polars.selectors as cs
from polars._typing import ConcatMethod
from polars._utils.wrap import wrap_ldf
from polars import functions as F
from polars.exceptions import InvalidOperationError
from polars._utils.various import ordered_unique

from functools import reduce
from itertools import chain

from pandas.core.indexes.datetimelike import DatetimeIndexOpsMixin
from toolz import first

from gluonts import maybe
from gluonts.dataset.common import DataEntry
from gluonts.itertools import Map, StarMap, SizedIterable

logger = logging.getLogger(__name__)

@dataclass
class DataFrameDataset:
    dataframes: InitVar[
        Union[
            pd.DataFrame,
            pd.Series,
            pl.LazyFrame,
            pl.DataFrame,
            pl.Series,
            Iterable[pd.DataFrame],
            Iterable[pd.Series],
            Iterable[tuple[Any, pd.DataFrame]],
            Iterable[tuple[Any, pd.Series]],
            Iterable[pl.LazyFrame],
            Iterable[pl.DataFrame],
            Iterable[pl.Series],
            Iterable[tuple[Any, pl.LazyFrame]],
            Iterable[tuple[Any, pl.DataFrame]],
            Iterable[tuple[Any, pl.Series]],
            dict[str, pd.DataFrame],
            dict[str, pd.Series],
            dict[str, pl.LazyFrame],
            dict[str, pl.DataFrame],
            dict[str, pl.Series],
        ]
    ]

    target: Union[str, list[str]] = "target"
    feat_dynamic_real: Optional[list[str]] = None
    past_feat_dynamic_real: Optional[list[str]] = None
    timestamp: Optional[str] = None
    freq: Optional[str] = None
    future_length: int = 0
    unchecked: bool = False
    assume_sorted: bool = False
    assume_resampled: bool = False
    dtype: Type = np.float32
    _data_entries: SizedIterable = field(init=False)

    def __iter__(self):
        yield from self._data_entries
        self.unchecked = True

    def __len__(self) -> int:
        return len(self._data_entries)

    def __repr__(self) -> str:
        info = ", ".join(
            [
                f"size={len(self)}",
                f"freq={self.freq}",
                f"num_feat_dynamic_real={self.num_feat_dynamic_real}",
                f"num_past_feat_dynamic_real={self.num_past_feat_dynamic_real}",
                f"num_feat_static_real={self.num_feat_static_real}",
                f"num_feat_static_cat={self.num_feat_static_cat}",
                f"static_cardinalities={self.static_cardinalities}",
            ]
        )
        return f"{self.__class__.__name__}<{info}>"
    
    @property
    def num_feat_static_cat(self) -> int:
        return len(self._static_cats)

    @property
    def num_feat_static_real(self) -> int:
        return len(self._static_reals)

    @property
    def num_feat_dynamic_real(self) -> int:
        return maybe.map_or(self.feat_dynamic_real, len, 0)

    @property
    def num_past_feat_dynamic_real(self) -> int:
        return maybe.map_or(self.past_feat_dynamic_real, len, 0)

    @property
    def static_cardinalities(self):
        return self._static_cats.max(axis=1).values + 1
    
@dataclass
class PandasDataset(DataFrameDataset):
    """
    A dataset type based on ``pandas.DataFrame``.

    This class is constructed with a collection of ``pandas.DataFrame``
    objects where each ``DataFrame`` is representing one time series.
    Both ``target`` and ``timestamp`` columns are essential. Dynamic features
    of a series can be specified with together with the series' ``DataFrame``,
    while static features can be specified in a separate ``DataFrame`` object
    via the ``static_features`` argument.

    Parameters
    ----------
    dataframes
        Single ``pd.DataFrame``/``pd.Series`` or a collection as list or dict
        containing at least ``timestamp`` and ``target`` values.
        If a dict is provided, the key will be the associated ``item_id``.
    target
        Name of the column that contains the ``target`` time series.
        For multivariate targets, a list of column names should be provided.
    timestamp
        Name of the column that contains the timestamp information.
    freq
        Frequency of observations in the time series. Must be a valid pandas
        frequency.
    feat_dynamic_real
        List of column names that contain dynamic real features.
    past_feat_dynamic_real
        List of column names that contain dynamic real features only available
        in the past.
    static_features
        ``pd.DataFrame`` containing static features for the series. The index
        should contain the key of the series in the ``dataframes`` argument.
    future_length
        For target and past dynamic features last ``future_length``
        elements are removed when iterating over the data set.
    unchecked
        Whether consistency checks on indexes should be skipped.
        (Default: ``False``)
    assume_sorted
        Whether to assume that indexes are sorted by time, and skip sorting.
        (Default: ``False``)
    """

    static_features: InitVar[Optional[pd.DataFrame]] = None
    _static_reals: pd.DataFrame = field(init=False)
    _static_cats: pd.DataFrame = field(init=False)

    def __post_init__(self, dataframes, static_features):
        # assert isinstance(dataframes, (
        #     pd.Series, pd.DataFrame, 
        #     Iterable[pd.DataFrame], Iterable[pd.Series], 
        #     Iterable[tuple[Any, pd.DataFrame]], Iterable[tuple[Any, pd.Series]],
        #     dict[str, pd.DataFrame], dict[str, pd.Series]
        # )), "dataframes arguement must be of type 'pd.Series', 'pd.DataFrame', or iterables/iterables of tuples, dicts of the above"
        

        if isinstance(dataframes, dict):
            pairs = dataframes.items()
        elif isinstance(dataframes, (pd.Series, pd.DataFrame)):
            pairs = [(None, dataframes)]
        else:
            assert isinstance(dataframes, SizedIterable)
            pairs = Map(pair_with_item_id, dataframes)

        self._data_entries = StarMap(self._pair_to_dataentry, pairs)

        if self.freq is None:
            assert (
                self.timestamp is None
            ), "You need to provide `freq` along with `timestamp`"

            self.freq = infer_freq(first(pairs)[1].index)

        static_features = maybe.unwrap_or_else(static_features, pd.DataFrame)

        object_columns = static_features.select_dtypes(
            "object"
        ).columns.tolist()
        if object_columns:
            logger.warning(
                f"Columns {object_columns} in static_features "
                f"have 'object' as data type and will be ignored; "
                f"consider setting this to 'category' using pd.DataFrame.astype, "
                f"if you wish to use them as categorical columns."
            )

        self._static_reals = (
            static_features.select_dtypes("number").astype(self.dtype).T
        )
        self._static_cats = (
            static_features.select_dtypes("category")
            .apply(lambda col: col.cat.codes)
            .astype(self.dtype)
            .T
        )

    def _pair_to_dataentry(self, item_id, df) -> DataEntry:
        if isinstance(df, pd.Series):
            df = df.to_frame(name=self.target)

        if self.timestamp:
            df.index = pd.DatetimeIndex(df[self.timestamp]).to_period(
                freq=self.freq
            )

        if not isinstance(df.index, pd.PeriodIndex):
            df = df.to_period(freq=self.freq)

        if not self.assume_sorted:
            df.sort_index(inplace=True)

        if not self.unchecked:
            assert is_uniform(df.index), (
                "Dataframe index is not uniformly spaced. "
                "If your dataframe contains data from multiple series in the "
                'same column ("long" format), consider constructing the '
                "dataset with `PandasDataset.from_long_dataframe` instead."
            )

        entry = {
            "start": df.index[0],
        }

        target = df[self.target].values
        target = target[: len(target) - self.future_length]
        entry["target"] = target.T # shape (num_features, time)

        if item_id is not None:
            entry["item_id"] = item_id

        if self.num_feat_static_cat > 0:
            entry["feat_static_cat"] = self._static_cats[item_id].values

        if self.num_feat_static_real > 0:
            entry["feat_static_real"] = self._static_reals[item_id].values

        if self.num_feat_dynamic_real > 0:
            entry["feat_dynamic_real"] = df[self.feat_dynamic_real].values.T

        if self.num_past_feat_dynamic_real > 0:
            past_feat_dynamic_real = df[self.past_feat_dynamic_real].values
            past_feat_dynamic_real = past_feat_dynamic_real[
                : len(past_feat_dynamic_real) - self.future_length
            ]
            entry["past_feat_dynamic_real"] = past_feat_dynamic_real.T

        return entry

    @classmethod
    def from_long_dataframe(
        cls,
        dataframe: pd.DataFrame,
        item_id: str,
        timestamp: Optional[str] = None,
        static_feature_columns: Optional[list[str]] = None,
        static_features: pd.DataFrame = pd.DataFrame(),
        **kwargs,
    ) -> "PandasDataset":
        """
        Construct ``PandasDataset`` out of a long data frame.

        A long dataframe contains time series data (both the target series and
        covariates) about multiple items at once. An ``item_id`` column is used
        to distinguish the items and ``group_by`` accordingly.

        Static features can be included in the long data frame as well (with
        constant value), or be given as a separate data frame indexed by the
        ``item_id`` values.

        Note: on large datasets, this constructor can take some time to complete
        since it does some indexing and groupby operations on the data, and caches
        the result.

        Parameters
        ----------
        dataframe
            pandas.DataFrame containing at least ``timestamp``, ``target`` and
            ``item_id`` columns.
        item_id
            Name of the column that, when grouped by, gives the different time
            series.
        static_feature_columns
            Columns in ``dataframe`` containing static features.
        static_features
            Dedicated ``DataFrame`` for static features. If both ``static_features``
            and ``static_feature_columns`` are specified, then the two sets of features
            are appended together.
        **kwargs
            Additional arguments. Same as of PandasDataset class.

        Returns
        -------
        PandasDataset
            Dataset containing series data from the given long dataframe.
        """
        if timestamp is not None:
            logger.info(f"Indexing data by '{timestamp}'.")
            dataframe.index = pd.to_datetime(dataframe[timestamp])

        if not isinstance(dataframe.index, DatetimeIndexOpsMixin):
            logger.info("Converting index into DatetimeIndex.")
            dataframe.index = pd.to_datetime(dataframe.index)

        if static_feature_columns is not None:
            logger.info(
                f"Collecting features from columns {static_feature_columns}."
            )
            other_static_features = (
                dataframe[[item_id] + static_feature_columns]
                .drop_duplicates()
                .set_index(item_id)
            )
            assert len(other_static_features) == len(
                dataframe[item_id].unique()
            )
        else:
            other_static_features = pd.DataFrame()

        logger.info(f"Grouping data by '{item_id}'; this may take some time.")
        pairs = list(dataframe.groupby(item_id))

        return cls(
            dataframes=pairs,
            static_features=pd.concat(
                [static_features, other_static_features], axis=1
            ),
            **kwargs,
        )

@dataclass
class PolarsDataset(DataFrameDataset):
    """_summary_

    Args:
        PandasDataset (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """

    static_features: InitVar[Optional[Union[pl.DataFrame, pl.LazyFrame]]] = None
    _static_reals: Union[pl.DataFrame, pl.LazyFrame] = field(init=False)
    _static_cats: Union[pl.DataFrame, pl.LazyFrame] = field(init=False)

    def __post_init__(self, dataframes, static_features):
        # assert isinstance(dataframes, (
        #     pl.Series, pl.DataFrame, pl.LazyFrame,
        #     Iterable[pl.DataFrame], Iterable[pl.LazyFrame], Iterable[pl.Series], 
        #     Iterable[tuple[Any, pl.DataFrame]], Iterable[tuple[Any, pl.LazyFrame]], Iterable[tuple[Any, pl.Series]],
        #     dict[str, pl.DataFrame], dict[str, pl.LazyFrame], dict[str, pl.Series]
        # )), "dataframes arguement must be of type 'pl.Series', 'pl.DataFrame', 'pl.LazyFrame', or iterables/iterables of tuples, dicts of the above"
        assert self.timestamp is not None, "Must provide timestamp column name for polars DataFrame/LazyFrame"
        if isinstance(dataframes, dict):
            pairs = dataframes.items()
        elif isinstance(dataframes, (pl.Series, pl.DataFrame, pl.LazyFrame)):
            pairs = [(None, dataframes)]
        else:
            assert isinstance(dataframes, SizedIterable)
            pairs = Map(pair_with_item_id, dataframes)

        self._data_entries = StarMap(self._pair_to_dataentry, pairs)

        if self.freq is None:
            assert (
                self.timestamp is None
            ), "You need to provide `freq` along with `timestamp`"

            self.freq = infer_freq(first(pairs)[1].select("time"))

        static_features = maybe.unwrap_or_else(static_features, pd.DataFrame)

        object_columns = static_features.select_dtypes(
            "object"
        ).columns.tolist()
        if object_columns:
            logger.warning(
                f"Columns {object_columns} in static_features "
                f"have 'object' as data type and will be ignored; "
                f"consider setting this to 'category' using pd.DataFrame.astype, "
                f"if you wish to use them as categorical columns."
            )

        self._static_reals = (
            static_features.select_dtypes("number").astype(self.dtype).T
        )
        self._static_cats = (
            static_features.select_dtypes("category")
            .apply(lambda col: col.cat.codes)
            .astype(self.dtype)
            .T
        )

    def _pair_to_dataentry(self, item_id, df) -> DataEntry:

        # if isinstance(df, pl.Series):
        #     df = df.to_frame(name=self.target).lazy()

        if not self.assume_resampled and self.timestamp:
            df = df.with_columns(pl.col(self.timestamp).dt.round(self.freq))

        if not self.assume_sorted:
            df = df.sort(by=self.timestamp)

        if not self.unchecked:
            assert is_uniform(df.select(pl.col(self.timestamp))), (
                "Dataframe index is not uniformly spaced. "
                "If your dataframe contains data from multiple series in the "
                'same column ("long" format), consider constructing the '
                "dataset with `PandasDataset.from_long_dataframe` instead."
            )

        entry = {
            "start": pd.Period(df.select(pl.col(self.timestamp).first()).collect().item(), freq=self.freq),
        }

        target_len = df.select(pl.len()).collect().item() 
        target = df.select(pl.col(self.target).slice(0, target_len - self.future_length))
        
        # entry["target"] = np.squeeze(target.collect().to_numpy().T)
        # entry["target"] = self.iterate_df(target, target_len - self.future_length)
        entry["target"] = target
        
        if item_id is not None:
            entry["item_id"] = item_id

        if self.num_feat_static_cat > 0:
            entry["feat_static_cat"] = self._static_cats[item_id].values

        if self.num_feat_static_real > 0:
            entry["feat_static_real"] = self._static_reals[item_id].values

        if self.num_feat_dynamic_real > 0:
            feat_dynamic_real = df.select(pl.col(self.feat_dynamic_real))
            # entry["feat_dynamic_real"] = feat_dynamic_real.collect().to_numpy().T
            # entry["feat_dynamic_real"] = self.iterate_df(feat_dynamic_real, target_len)
            # entry["feat_dynamic_real"] = self.iterate_df(feat_dynamic_real, target_len)
            entry["feat_dynamic_real"] = feat_dynamic_real

        if self.num_past_feat_dynamic_real > 0:
            past_feat_dynamic_real_len = df.select(pl.len()).collect().item() # TODO same as target_len?
            past_feat_dynamic_real = df.select(pl.col(self.past_feat_dynamic_real).slice(0, past_feat_dynamic_real_len - self.future_length))
            # entry["past_feat_dynamic_real"] = past_feat_dynamic_real.collect().to_numpy().T
            # entry["past_feat_dynamic_real"] = self.iterate_df(past_feat_dynamic_real, past_feat_dynamic_real_len - self.future_length) 
            entry["past_feat_dynamic_real"] = past_feat_dynamic_real

        return entry

    def iterate_df(self, df, length):
        i = 0
        while i < length:
            yield df.select(pl.all().slice(i, 1)).collect().to_numpy().T
            if i == length - 1:
                i = 0
            else:
                i += 1

    @classmethod
    def from_long_dataframe(
        cls,
        dataframe: Union[pl.DataFrame, pl.LazyFrame],
        item_id: str,
        timestamp: Optional[str] = None,
        static_feature_columns: Optional[list[str]] = None,
        static_features: pd.DataFrame = pd.DataFrame(),
        **kwargs,
    ) -> "PandasDataset":
        """
        Construct ``PandasDataset`` out of a long data frame.

        A long dataframe contains time series data (both the target series and
        covariates) about multiple items at once. An ``item_id`` column is used
        to distinguish the items and ``group_by`` accordingly.

        Static features can be included in the long data frame as well (with
        constant value), or be given as a separate data frame indexed by the
        ``item_id`` values.

        Note: on large datasets, this constructor can take some time to complete
        since it does some indexing and groupby operations on the data, and caches
        the result.

        Parameters
        ----------
        dataframe
            pandas.DataFrame containing at least ``timestamp``, ``target`` and
            ``item_id`` columns.
        item_id
            Name of the column that, when grouped by, gives the different time
            series.
        static_feature_columns
            Columns in ``dataframe`` containing static features.
        static_features
            Dedicated ``DataFrame`` for static features. If both ``static_features``
            and ``static_feature_columns`` are specified, then the two sets of features
            are appended together.
        **kwargs
            Additional arguments. Same as of PandasDataset class.

        Returns
        -------
        PandasDataset
            Dataset containing series data from the given long dataframe.
        """
        if timestamp is not None:
            logger.info(f"Indexing data by '{timestamp}'.")
            dataframe = dataframe.with_columns(pl.col(timestamp).str.to_datetime())

        if static_feature_columns is not None:
            logger.info(
                f"Collecting features from columns {static_feature_columns}."
            )
            other_static_features = (
                dataframe.select([[item_id] + static_feature_columns])
                .unique()
                .collect()
                .to_pandas()
                .set_index(item_id)
            )
            assert len(other_static_features) == len(
                dataframe.select(pl.col(item_id)).unique()
            )
        else:
            other_static_features = pd.DataFrame()

        logger.info(f"Grouping data by '{item_id}'; this may take some time.")
        pairs = list(dataframe.group_by(item_id))

        return cls(
            dataframes=pairs,
            static_features=pd.concat(
                [static_features, other_static_features], axis=1
            ),
            **kwargs,
        )


def pair_with_item_id(obj: Union[tuple, pd.DataFrame, pd.Series, pl.DataFrame, pl.LazyFrame]):
    if isinstance(obj, tuple) and len(obj) == 2:
        return obj
    if isinstance(obj, (pd.DataFrame, pd.Series, pl.DataFrame, pl.LazyFrame)):
        return (None, obj)
    raise ValueError("input must be a pair, or a pandas Series or DataFrame.")


def infer_freq(index: Union[pd.Index, pl.DataFrame, pl.LazyFrame]) -> str:
    if isinstance(index, pd.PeriodIndex):
        return index.freqstr
    elif isinstance(index, pl.LazyFrame):
        freq = index.select(pl.all().diff().last()).collect().item()
        freq = f"{freq.seconds}s" 
    elif isinstance(index, pl.DataFrame):
        freq = index.select(pl.all().diff().last()).item()
        freq = f"{freq.seconds}s"
    else:
        freq = pd.infer_freq(index)
    # pandas likes to infer the `start of x` frequency, however when doing
    # df.to_period("<x>S"), it fails, so we avoid using it. It's enough to
    # remove the trailing S, e.g `MS` -> `M
    if len(freq) > 1 and freq.endswith("S"):
        return freq[:-1]

    return freq


def is_uniform(index: Union[pd.PeriodIndex, pl.DataFrame, pl.LazyFrame]) -> bool:
    """
    Check if ``index`` contains monotonically increasing periods, evenly spaced
    with frequency ``index.freq``.

        >>> ts = ["2021-01-01 00:00", "2021-01-01 02:00", "2021-01-01 04:00"]
        >>> is_uniform(pd.DatetimeIndex(ts).to_period("2H"))
        True
        >>> ts = ["2021-01-01 00:00", "2021-01-01 04:00"]
        >>> is_uniform(pd.DatetimeIndex(ts).to_period("2H"))
        False
    """
    if isinstance(index, pl.DataFrame):
        freq = index.select(pl.all().diff().slice(1).first()).item()
        return index.select((pl.all().diff().slice(1) == freq).all()).item()
    elif isinstance(index, pl.LazyFrame): 
        freq = index.select(pl.all().diff().slice(1).first()).collect().item()
        return index.select((pl.all().diff().slice(1) == freq).all()).collect().item()
    else:
        return bool(np.all(np.diff(index.asi8) == index.freq.n))

class IterableLazyFrame:
    def __init__(self, data=None, data_path=None, schema=None, target_cols=None, dtype=None):
        
        if data_path is not None and data is None:
            self._df = pl.scan_parquet(data_path, schema=schema)
        elif data is not None and data_path is None:
            self._df = pl.LazyFrame(data, schema=schema)
        else:
            raise Exception("Must pass either argument 'data' or 'data_path', but not both.")
        # TODO make sure data types here match what is set in estimator add time features etc 
        # self.dtype = list(self._df.select(cs.float()).collect_schema().values())[0]
        if dtype is not None:
            self._df = self._df.with_columns(cs.float().cast(dtype))
        # self.dtype = dtype
        self.target_cols = target_cols
        # self.i = 0
        self._length = self._df.select(pl.len()).collect().item()
        self._shape = (len(self._df.collect_schema().names()), self._length)
        
    
    def __getattr__(self, name):
        # Delegate attribute access to the underlying LazyFrame
        attr = getattr(self._df, name)
        
        # If the attribute is a callable (a method), we need to bind it
        # to the MyLazyFrame instance to maintain the correct context
        if callable(attr):
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, pl.LazyFrame):
                    return IterableLazyFrame._from_lazyframe(result, self.target_cols) # maintain iterableLazyFrame type
                else:
                    return result
            return wrapper
        else:
            return attr
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or self.length
            return self._df.slice(start, stop - start).collect().to_numpy().T
            # return torch.from_numpy(self._df.slice(start, stop - start).collect().to_numpy().T)
        elif isinstance(key[1], slice):
            start = key[1].start or 0
            stop = key[1].stop or self.length
            return self._df.slice(start, stop - start).collect().to_numpy().T # to avoid ellipsis
            # return torch.from_numpy(self._df.slice(start, stop - start).collect().to_numpy().T) # to avoid ellipsis
        else:
            return self._df.slice(key, 1).collect().to_numpy().T
            # return torch.from_numpy(self._df.slice(key, 1).collect().to_numpy().T)
    
    # def __setitem__(self, key, value):
    #     if isinstance(key, slice):
    #         self._df = self._df
    
    @classmethod
    def _from_lazyframe(cls, df: pl.LazyFrame, target_cols):
        inst = cls.__new__(cls)
        inst._df = df
        inst._length = df.select(pl.len()).collect().item()
        inst._shape = (len(df.collect_schema().names()), inst._length)
        # inst.i = 0 # TODO should it be set to old index...
        # inst.dtype = list(df.select(cs.float()).collect_schema().values())[0]
        inst.target_cols = target_cols
        return inst
    
    # def __iter__(self):
    #     while self.i < self.length:
    #         yield self.select(pl.all().slice(self.i, 1)).collect().to_numpy().T
    #         if self.i == self.length - 1:
    #             self.i = 0
    #         else:
    #             self.i += 1
    
    @property
    def length(self):
        return self._length

    @property
    def shape(self):
        return self._shape
    
    @property
    def dtype(self):
        return list(self._df.select(cs.float()).collect_schema().values())[0]
    
    # def __len__(self):
    #     return self.length

def concat_lazyframes(items: Iterable[IterableLazyFrame],
                        *,
                        how: ConcatMethod = "vertical",
                        rechunk: bool = False,
                        parallel: bool = True):
    
    elems = list(items)

    if not elems:
        msg = "cannot concat empty list"
        raise ValueError(msg)
    elif len(elems) == 1 and isinstance(
        elems[0], (pl.DataFrame, pl.Series, pl.LazyFrame)
    ):
        return elems[0]

    if how == "align":
        if not isinstance(elems[0], (pl.DataFrame, pl.LazyFrame)):
            msg = f"'align' strategy is not supported for {type(elems[0]).__name__!r}"
            raise TypeError(msg)

        # establish common columns, maintaining the order in which they appear
        all_columns = list(chain.from_iterable(e.collect_schema() for e in elems))
        key = {v: k for k, v in enumerate(ordered_unique(all_columns))}
        common_cols = sorted(
            reduce(
                lambda x, y: set(x) & set(y),  # type: ignore[arg-type, return-value]
                chain(e.collect_schema() for e in elems),
            ),
            key=lambda k: key.get(k, 0),
        )
        # we require at least one key column for 'align'
        if not common_cols:
            msg = "'align' strategy requires at least one common column"
            raise InvalidOperationError(msg)

        # align the frame data using a full outer join with no suffix-resolution
        # (so we raise an error in case of column collision, like "horizontal")
        lf: pl.LazyFrame = reduce(
            lambda x, y: (
                x.join(y, how="full", on=common_cols, suffix="_PL_CONCAT_RIGHT")
                # Coalesce full outer join columns
                .with_columns(
                    F.coalesce([name, f"{name}_PL_CONCAT_RIGHT"])
                    for name in common_cols
                )
                .drop([f"{name}_PL_CONCAT_RIGHT" for name in common_cols])
            ),
            [df.lazy() for df in elems],
        ).sort(by=common_cols)

        eager = isinstance(elems[0], pl.DataFrame)
        return lf.collect() if eager else IterableLazyFrame(data=lf.collect())  # type: ignore[return-value]

    first = elems[0]
    
    if how in ("vertical", "vertical_relaxed"):
        return IterableLazyFrame(data=wrap_ldf(
            plr.concat_lf(
                elems,
                rechunk=rechunk,
                parallel=parallel,
                to_supertypes=how.endswith("relaxed"),
            )
        ).collect())
    elif how in ("diagonal", "diagonal_relaxed"):
        return IterableLazyFrame(data=wrap_ldf(
            plr.concat_lf_diagonal(
                elems,
                rechunk=rechunk,
                parallel=parallel,
                to_supertypes=how.endswith("relaxed"),
            )
        ).collect())
    elif how == "horizontal":
        return IterableLazyFrame(data=wrap_ldf(
            plr.concat_lf_horizontal(
                elems,
                parallel=parallel,
            )).collect()
        )
    else:
        allowed = ", ".join(repr(m) for m in get_args(ConcatMethod))
        msg = f"LazyFrame `how` must be one of {{{allowed}}}, got {how!r}"
        raise ValueError(msg)