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

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.distributions import Distribution

from gluonts.model.forecast import Forecast, Quantile, SampleForecast


class DistributionForecast(Forecast):
    """
    A `Forecast` object that uses a distribution directly.

    This can for instance be used to represent marginal probability
    distributions for each time point -- although joint distributions are
    also possible, e.g. when using MultiVariateGaussian).

    Parameters
    ----------
    distribution
        Distribution object. This should represent the entire prediction
        length, i.e., if we draw `num_samples` samples from the distribution,
        the sample shape should be

            samples = trans_dist.sample(num_samples)
            samples.shape -> (num_samples, prediction_length)

    start_date
        start of the forecast
    info
        additional information that the forecaster may provide e.g. estimated
        parameters, number of iterations ran etc.
    """

    def __init__(
        self,
        distribution: Distribution,
        start_date: pd.Period,
        item_id: Optional[str] = None,
        info: Optional[Dict] = None,
    ) -> None:
        self.distribution = distribution
        self.shape = distribution.batch_shape + distribution.event_shape
        self.prediction_length = self.shape[0]
        self.item_id = item_id
        self.info = info
        self._dim: Optional[int] = None

        assert isinstance(
            start_date, pd.Period
        ), "start_date should be a pandas Period object"
        self.start_date = start_date

        self._mean = None

    #CHANGE
    # @property
    # def dim(self) -> tuple:
    #     return self.distribution.event_shape

    @property
    def mean(self) -> np.ndarray:
        """
        Forecast mean.
        """
        if self._mean is not None:
            return self._mean
        else:
            _mean = self.distribution.mean.cpu().numpy()
            self._mean = _mean
            return _mean

    @property
    def mean_ts(self) -> pd.Series:
        """
        Forecast mean, as a pandas.Series object.
        """
        return pd.Series(data=self.mean, index=self.index)

    def quantile(self, level: Union[float, str]) -> np.ndarray:
        level = Quantile.parse(level).value
        if self.distribution.event_shape == 1:
            return (
                self.distribution.icdf(
                    torch.tensor([level], device=self.distribution.mean.device)
                )
                .cpu()
                .numpy()
            )
        else:
            # Note: computes quantile on each dimension of the target independently.
            # `sample_idx` would be same for each element of the batch, time point and dimension.
            num_samples = 200 # TODO QUESTION should be argument
            samples = self.distribution.sample(torch.Size((num_samples,)))
            sorted_samples = torch.sort(samples, axis=0).values
            # num_samples = sorted_samples.shape[0]
            sample_idx = int(np.round(num_samples * level)) - 1

            return sorted_samples[sample_idx, :].cpu().numpy()

    def to_sample_forecast(self, num_samples: int = 200) -> SampleForecast:
        return SampleForecast(
            samples=self.distribution.sample(torch.Size((num_samples,)))
            .cpu()
            .numpy(),
            start_date=self.start_date,
            item_id=self.item_id,
            info=self.info,
        )

    # CHANGE
    def copy_dim(self, dim: int) -> "SampleForecast":
        """
        Returns a new Forecast object with only the selected sub-dimension.

        Parameters
        ----------
        dim
            The returned forecast object will only represent this dimension.
        """
        if self.distribution.event_shape == 1:
            distribution = self.distribution
        else:
            target_dim = self.distribution.event_shape[0]
            assert dim < target_dim, (
                f"must set 0 <= dim < target_dim, but got dim={dim},"
                f" target_dim={target_dim}"
            )
            # distribution = self.distribution[:, :, dim]
            distribution = self.distribution.__class__(
                **{param_key: getattr(self.distribution, param_key)[:, dim] for param_key in self.distribution.arg_constraints.keys()}) 

        return DistributionForecast(
            distribution=distribution,
            start_date=self.start_date,
            item_id=self.item_id,
            info=self.info,
        )

    def dim(self) -> int:
        """
        Returns the dimensionality of the forecast object.
        """
        if self._dim is not None:
            return self._dim
        else:
            return self.distribution.event_shape[0]
            # if len(self.samples.shape) == 2:
            #     # univariate target
            #     # shape: (num_samples, prediction_length)
            #     return 1
            # else:
            #     # multivariate target
            #     # shape: (num_samples, prediction_length, target_dim)
            #     return self.samples.shape[2]