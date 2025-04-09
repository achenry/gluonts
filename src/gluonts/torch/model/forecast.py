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

            # --- Debugging Cholesky ---
            # import logging
            # import torch
            # try:
            required_params = self.distribution.arg_constraints.keys()
            original_params = {
                param_key: getattr(self.distribution, param_key)
                for param_key in required_params
                if hasattr(self.distribution, param_key) # Check if attribute exists
            }
            # Slice parameters *before* passing them to the constructor
            sliced_params = {}
            for param_key in original_params:
                if original_params[param_key].ndim > 1 and original_params[param_key].shape[-1] == original_params[param_key].shape[-2] == target_dim:
                    sliced_params[param_key] = original_params[param_key][..., dim, dim]
                else:
                    sliced_params[param_key] = original_params[param_key][..., dim]

                # logging.info(f"copy_dim(dim={dim}): Reconstructing {self.distribution.__class__.__name__}")
                # for param_key, tensor in original_params.items():
                #     logging.info(f"  Original {param_key} shape: {tensor.shape}, has NaNs: {torch.isnan(tensor).any()}, has Infs: {torch.isinf(tensor).any()}")
                # for param_key, tensor in sliced_params.items():
                #     logging.info(f"  Sliced {param_key} shape: {tensor.shape}, has NaNs: {torch.isnan(tensor).any()}, has Infs: {torch.isinf(tensor).any()}")

                # Specifically log cov_diag if it exists, as it's crucial for positive-definiteness
                # if 'cov_diag' in sliced_params:
                #     diag_tensor = sliced_params['cov_diag']
                #     logging.info(f"  Sliced cov_diag values (shape {diag_tensor.shape}): {diag_tensor.flatten()}")
                #     if torch.any(diag_tensor <= 1e-6): # Check for non-positive or very small values
                #          logging.warning(f"  WARNING: Sliced cov_diag contains non-positive or near-zero values!")

            # except Exception as log_ex:
            #     logging.error(f"copy_dim(dim={dim}): Error during debug logging: {log_ex}")
            # --- End Debugging ---

            # Pass the pre-sliced parameters
            if self.distribution.__class__.__name__ == "MultivariateNormal":
                del sliced_params["precision_matrix"], sliced_params["scale_tril"]
                sliced_params["covariance_matrix"] = torch.diag(sliced_params["covariance_matrix"])
                distribution = self.distribution.__class__(**sliced_params)
            else:
                distribution = self.distribution.__class__(**sliced_params)

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