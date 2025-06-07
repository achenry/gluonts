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

import logging
import os
from typing import Callable, Iterable, Optional

import numpy as np

from gluonts.dataset import DataBatch, Dataset
from gluonts.itertools import (
    Cyclic,
    IterableSlice,
    PseudoShuffled,
    batcher,
    rows_to_columns,
)
from gluonts.pydantic import BaseModel
from gluonts.transform import (
    AdhocTransform,
    Identity,
    SelectFields,
    Transformation,
    Valmap,
)

logger = logging.getLogger(__name__)


def _detect_distributed_early() -> tuple[bool, int, int]:
    """
    Detect distributed training from environment variables.
    
    Distinguishes between two scenarios:
    1. Independent tuning workers (WORKER_RANK set) - NO data sharding
    2. Cooperative training workers (DDP via srun) - YES data sharding
    
    Returns: (is_distributed_sharding_enabled, world_size, rank)
    """
    # Check SLURM environment (most common for HPC)
    if "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ:
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            
            # Only enable data sharding for cooperative DDP workers
            # If WORKER_RANK is set, these are independent tuning workers
            is_independent_worker = "WORKER_RANK" in os.environ
            enable_sharding = world_size > 1 and not is_independent_worker
            
            if enable_sharding:
                logger.info(f"Cooperative distributed training detected: rank={rank}, world_size={world_size} (data sharding enabled)")
            elif world_size > 1 and is_independent_worker:
                logger.info(f"Independent worker detected: SLURM_PROCID={rank}, WORKER_RANK={os.environ['WORKER_RANK']} (data sharding disabled)")
            
            return enable_sharding, world_size, rank
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to parse SLURM environment variables: {e}")
    
    # Check PyTorch distributed environment variables (fallback)
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        try:
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ["RANK"])
            
            # Same logic: disable sharding for independent workers
            is_independent_worker = "WORKER_RANK" in os.environ
            enable_sharding = world_size > 1 and not is_independent_worker
            
            if enable_sharding:
                logger.info(f"PyTorch distributed training detected: rank={rank}, world_size={world_size} (data sharding enabled)")
            elif world_size > 1 and is_independent_worker:
                logger.info(f"Independent worker detected: RANK={rank}, WORKER_RANK={os.environ['WORKER_RANK']} (data sharding disabled)")
            
            return enable_sharding, world_size, rank
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to parse PyTorch environment variables: {e}")
    
    # Single process or no distributed training detected
    return False, 1, 0


class DistributedShardedIterable:
    """
    Wraps an iterable to provide distributed data sharding.    
    Each process gets a different subset of the data.
    """
    
    def __init__(self, iterable, world_size: int = 1, rank: int = 0):
        self.iterable = iterable
        self.world_size = world_size
        self.rank = rank
        
    def __iter__(self):
        """Yield every world_size-th item starting from rank."""
        for i, item in enumerate(self.iterable):
            if i % self.world_size == self.rank:
                yield item


DataLoader = Iterable[DataBatch]


# TODO: the following are for backward compatibility
# and could eventually be removed


class Batch(Transformation, BaseModel):
    batch_size: int

    def __call__(self, data, is_train):
        yield from batcher(data, self.batch_size)


class Stack(Transformation, BaseModel):
    def __call__(self, data, is_train):
        for batch in data:
            yield rows_to_columns(batch, np.array)



def as_stacked_batches(
    dataset: Dataset,
    *,
    batch_size: int,
    output_type: Optional[Callable] = None,
    num_batches_per_epoch: Optional[int] = None,
    shuffle_buffer_length: Optional[int] = None,
    field_names: Optional[list] = None,
    distributed: bool = True,
):
    """
    Prepare data in batches to be passed to a network.

    Input data is collected into batches of size ``batch_size`` and then
    columns are stacked on top of each other. In addition, the result is
    wrapped in ``output_type`` if provided.

    If ``num_batches_per_epoch`` is provided, only those number of batches are
    effectively returned. This is especially useful for training when
    providing a cyclic dataset.

    To pseudo shuffle data, ``shuffle_buffer_length`` can be set to collect
    inputs into a buffer first, from which we then randomly sample.

    Setting ``field_names`` will only consider those columns in the input data
    and discard all other values.
    
    Parameters
    ----------
    distributed : bool, default True
        Whether to enable distributed training support. When True and running
        in a cooperative distributed environment (DDP), data will be automatically 
        sharded across processes. Independent workers (tuning mode) will not have 
        data sharding applied.
    """

    # Only detect distributed training if distributed=True
    if distributed:
        enable_sharding, world_size, rank = _detect_distributed_early()
    else:
        enable_sharding, world_size, rank = False, 1, 0
    
    if distributed and enable_sharding:
        # Cooperative distributed training - apply data sharding
        logger.info(f"Applying distributed data sharding: rank={rank}, world_size={world_size}")
        
        # Adjust num_batches_per_epoch for distributed training
        if num_batches_per_epoch is not None:
            original_batches = num_batches_per_epoch
            num_batches_per_epoch = num_batches_per_epoch // world_size
            logger.info(f"Adjusted batches per epoch for rank {rank}: {original_batches} -> {num_batches_per_epoch}")
    else:
        # Single process or independent workers - no data sharding
        world_size = 1
        rank = 0
        if "WORKER_RANK" in os.environ:
            logger.info(f"Independent worker mode: WORKER_RANK={os.environ['WORKER_RANK']} (full dataset access)")
        else:
            logger.info("Single process mode detected")

    if shuffle_buffer_length:
        dataset = PseudoShuffled(dataset, shuffle_buffer_length)

    transform: Transformation = Identity()

    if field_names is not None:
        transform += SelectFields(field_names)

    transform += Batch(batch_size=batch_size)
    transform += Stack()

    if output_type is not None:
        transform += Valmap(output_type)

    # Note: is_train needs to be provided but does not have an effect
    transformed_dataset = transform.apply(dataset, is_train=True)
    
    # Apply distributed sharding only for cooperative workers
    if distributed and enable_sharding and world_size > 1:
        transformed_dataset = DistributedShardedIterable(
            transformed_dataset, 
            world_size=world_size, 
            rank=rank
        )
        logger.info(f"Applied distributed sharding for rank {rank}/{world_size}")
    
    return IterableSlice(transformed_dataset, num_batches_per_epoch)


def TrainDataLoader(
    dataset: Dataset,
    *,
    transform: Transformation = Identity(),
    batch_size: int,
    stack_fn: Callable,
    num_batches_per_epoch: Optional[int] = None,
    shuffle_buffer_length: Optional[int] = None,
):
    """
    Construct an iterator of batches for training purposes.

    This function wraps around ``DataLoader`` to offer training-specific
    behaviour and options, as follows:

        1. The provided dataset is iterated cyclically, so that one can go over
        it multiple times in a single epoch. 2. A transformation must be
        provided, that is lazily applied as the dataset is being iterated;
        this is useful e.g. to slice random instances of fixed length out of
        each time series in the dataset. 3. The resulting batches can be
        iterated in a pseudo-shuffled order.

    The returned object is a stateful iterator, whose length is either
    ``num_batches_per_epoch`` (if not ``None``) or infinite (otherwise).

    Parameters
    ----------
    dataset
        Data to iterate over.
    transform
        Transformation to be lazily applied as data is being iterated.
        The transformation is applied in "training mode" (``is_train=True``).
    batch_size
        Number of entries to include in a batch.
    stack_fn
        Function to use to stack data entries into batches.
        This can be used to set a specific array type or computing device
        the arrays should end up onto (CPU, GPU).
    num_batches_per_epoch
        Length of the iterator. If ``None``, then the iterator is endless.
    shuffle_buffer_length
        Size of the buffer used for shuffling. Default: None, in which case no
        shuffling occurs.

    Returns
    -------
    Iterator[DataBatch]
        An iterator of batches.
    """
    dataset: Dataset = Cyclic(dataset)

    if shuffle_buffer_length:
        dataset = PseudoShuffled(dataset, shuffle_buffer_length)

    transform += Batch(batch_size=batch_size) + AdhocTransform(stack_fn)
    transformed_dataset = transform.apply(dataset, is_train=True)

    batches = iter(transformed_dataset)
    return IterableSlice(batches, num_batches_per_epoch)


def ValidationDataLoader(
    dataset: Dataset,
    *,
    transform: Transformation = Identity(),
    batch_size: int,
    stack_fn: Callable,
):
    """
    Construct an iterator of batches for validation purposes.

    Parameters
    ----------
    dataset
        Data to iterate over.
    transform
        Transformation to be lazily applied as data is being iterated.
        The transformation is applied in "training mode" (``is_train=True``).
    batch_size
        Number of entries to include in a batch.
    stack_fn
        Function to use to stack data entries into batches.
        This can be used to set a specific array type or computing device
        the arrays should end up onto (CPU, GPU).

    Returns
    -------
    Iterable[DataBatch]
        An iterable sequence of batches.
    """

    transform += Batch(batch_size=batch_size) + AdhocTransform(stack_fn)
    return transform.apply(dataset, is_train=True)


def InferenceDataLoader(
    dataset: Dataset,
    *,
    transform: Transformation = Identity(),
    batch_size: int,
    stack_fn: Callable,
):
    """
    Construct an iterator of batches for inference purposes.

    Parameters
    ----------
    dataset
        Data to iterate over.
    transform
        Transformation to be lazily applied as data is being iterated.
        The transformation is applied in "inference mode" (``is_train=False``).
    batch_size
        Number of entries to include in a batch.
    stack_fn
        Function to use to stack data entries into batches.
        This can be used to set a specific array type or computing device
        the arrays should end up onto (CPU, GPU).

    Returns
    -------
    Iterable[DataBatch]
        An iterable sequence of batches.
    """
    transform += Batch(batch_size=batch_size) + AdhocTransform(stack_fn)
    return transform.apply(dataset, is_train=False)
