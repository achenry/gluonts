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

from typing import NamedTuple, Optional, Iterable, Dict, Any
import logging

import inspect
import numpy as np
# Use the newer namespace consistent with Lightning > v2.0
import lightning.pytorch as pl
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.env import env
from gluonts.itertools import Cached
from gluonts.model import Estimator, Predictor
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import Transformation

logger = logging.getLogger(__name__)


class TrainOutput(NamedTuple):
    transformation: Transformation
    trained_net: nn.Module
    trainer: pl.Trainer
    predictor: PyTorchPredictor


class PyTorchLightningEstimator(Estimator):
    """
    An `Estimator` type with utilities for creating PyTorch-Lightning-based
    models.

    To extend this class, one needs to implement three methods:
    `create_transformation`, `create_training_network`, `create_predictor`,
    `create_training_data_loader`, and `create_validation_data_loader`.
    """

    @validated()
    def __init__(
        self,
        trainer_kwargs: Dict[str, Any],
        lead_time: int = 0,
    ) -> None:
        super().__init__(lead_time=lead_time)
        self.trainer_kwargs = trainer_kwargs

    def create_transformation(self) -> Transformation:
        """
        Create and return the transformation needed for training and inference.

        Returns
        -------
        Transformation
            The transformation that will be applied entry-wise to datasets,
            at training and inference time.
        """
        raise NotImplementedError

    def create_lightning_module(self) -> pl.LightningModule:
        """
        Create and return the network used for training (i.e., computing the
        loss).

        Returns
        -------
        pl.LightningModule
            The network that computes the loss given input data.
        """
        raise NotImplementedError

    def create_predictor(
        self,
        transformation: Transformation,
        module,
        **kwargs # CHANGE
    ) -> PyTorchPredictor:
        """
        Create and return a predictor object.

        Parameters
        ----------
        transformation
            Transformation to be applied to data before it goes into the model.
        module
            A trained `pl.LightningModule` object.

        Returns
        -------
        Predictor
            A predictor wrapping a `nn.Module` used for inference.
        """
        raise NotImplementedError

    def create_training_data_loader(
        self, data: Dataset, module, **kwargs
    ) -> Iterable:
        """
        Create a data loader for training purposes.

        Parameters
        ----------
        data
            Dataset from which to create the data loader.
        module
            The `pl.LightningModule` object that will receive the batches from
            the data loader.

        Returns
        -------
        Iterable
            The data loader, i.e. and iterable over batches of data.
        """
        raise NotImplementedError

    def create_validation_data_loader(
        self, data: Dataset, module, **kwargs
    ) -> Iterable:
        """
        Create a data loader for validation purposes.

        Parameters
        ----------
        data
            Dataset from which to create the data loader.
        module
            The `pl.LightningModule` object that will receive the batches from
            the data loader.

        Returns
        -------
        Iterable
            The data loader, i.e. and iterable over batches of data.
        """
        raise NotImplementedError

    
    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        from_predictor: Optional[PyTorchPredictor] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> TrainOutput:
        # Check if this estimator wants to use PyTorch DataLoaders for distributed training
        use_pytorch_dataloader = getattr(self, 'use_pytorch_dataloader', False)
        
        # Always create transformation (needed for predictor even in PyTorch path)
        transformation = self.create_transformation()
        
        # Create lightning module
        training_network = self.create_lightning_module()
        
        if use_pytorch_dataloader:
            # PyTorch DataLoader path for distributed training
            logger.info("Using PyTorch DataLoader for distributed training")
            
            # Check if the estimator has PyTorch-specific data loader methods
            if not hasattr(self, 'create_pytorch_training_data_loader') and not hasattr(self, 'create_pytorch_data_module'):
                raise NotImplementedError(
                    f"{self.__class__.__name__} must implement create_pytorch_training_data_loader OR create_pytorch_data_module "
                    "when use_pytorch_dataloader=True"
                )
            
            training_data_loader = None
            # For PyTorch path, training_data should be a file path or similar identifier
            # Pass it directly to the PyTorch dataloader creation method
            # training_data_loader = self.create_pytorch_training_data_loader(
            #     training_data,
            #     training_network,
            #     **kwargs
            # )
            
            
            validation_data_loader = None
            if validation_data is not None:
                if not hasattr(self, 'create_pytorch_validation_data_loader') and not hasattr(self, 'create_pytorch_data_module'):
                    raise NotImplementedError(
                        f"{self.__class__.__name__} must implement create_pytorch_validation_data_loader OR create_pytorch_data_module "
                        "when use_pytorch_dataloader=True"
                    )
                
                # validation_data_loader = self.create_pytorch_validation_data_loader(
                #     validation_data,
                #     training_network,
                #     **kwargs
                # )
                
            logging.info(f"Creating LightningDataModule for distributed training with kwargs {kwargs}.")    
            
            data_module = self.create_pytorch_data_module(
                train_data_path=training_data,
                val_data_path=validation_data,
                **kwargs
            )
        else:
            # Original GluonTS data loading path
            with env._let(max_idle_transforms=max(len(training_data), 100)):
                transformed_training_data: Dataset = transformation.apply(
                    training_data, is_train=True
                )
                 
                if cache_data:
                    transformed_training_data = Cached(transformed_training_data)

                # {p: t.shape for p, t in training_network.named_parameters()}
                training_data_loader = self.create_training_data_loader(
                    transformed_training_data,
                    training_network,
                    shuffle_buffer_length=shuffle_buffer_length,
                )
                
                # x = next(iter(training_data_loader))
                # x = sum(1 for _ in training_data_loader)
            validation_data_loader = None

            if validation_data is not None:
                with env._let(max_idle_transforms=max(len(validation_data), 100)):
                    transformed_validation_data: Dataset = transformation.apply(
                        validation_data, is_train=True
                    )
                    if cache_data:
                        transformed_validation_data = Cached(
                            transformed_validation_data
                        )

                    
                    validation_data_loader = self.create_validation_data_loader(
                        transformed_validation_data,
                        training_network,
                    )

        if from_predictor is not None:
            training_network.load_state_dict(
                from_predictor.network.state_dict()
            )

        # Check if checkpointing is disabled
        enable_checkpointing = self.trainer_kwargs.get("enable_checkpointing", True)
        
        custom_callbacks = self.trainer_kwargs.pop("callbacks", [])

        # @boujuan Check if a ModelCheckpoint is already provided in custom_callbacks
        has_custom_checkpoint = any(isinstance(cb, pl.callbacks.ModelCheckpoint) for cb in custom_callbacks)
        
        # Create default checkpoint only if checkpointing is enabled and no custom checkpoint exists
        checkpoint = None
        if enable_checkpointing and not has_custom_checkpoint:
            monitor = "train_loss" if validation_data is None else "val_loss"
            checkpoint = pl.callbacks.ModelCheckpoint(
                monitor=monitor, mode="min", verbose=True
            )
        
        # @boujuan Construct the final list of callbacks for the Trainer
        if enable_checkpointing:
            final_callbacks = custom_callbacks if has_custom_checkpoint else [checkpoint] + custom_callbacks
        else:
            # Remove any ModelCheckpoint callbacks when checkpointing is disabled
            final_callbacks = [cb for cb in custom_callbacks if not isinstance(cb, pl.callbacks.ModelCheckpoint)]
            logger.info("Checkpointing disabled: Removed all ModelCheckpoint callbacks")
        if has_custom_checkpoint and enable_checkpointing:
            checkpoint = [cb for cb in custom_callbacks if cb.__class__.__name__ == "ModelCheckpoint"][0]
            
        logging.info(f"Final Trainer callbacks: {final_callbacks}")
        logging.info(f"Final Trainer kwargs: {self.trainer_kwargs}")
        trainer = pl.Trainer(
            **{
                # "accelerator": "auto",
                "callbacks": final_callbacks, # Use the combined list
                "num_sanity_val_steps": 0, # This disables the check that is crashing
                **self.trainer_kwargs,
            }
        )
        
        if not (training_data_loader is None):
            trainer.fit(
                model=training_network,
                train_dataloaders=training_data_loader,
                val_dataloaders=validation_data_loader,
                ckpt_path=ckpt_path,
            )
        else:
            trainer.fit(
                model=training_network,
                datamodule=data_module,
                ckpt_path=ckpt_path,
            )

        if checkpoint is not None and checkpoint.best_model_path != "":
            logger.info(
                f"Loading best model from {checkpoint.best_model_path}"
            )
            best_model = training_network.__class__.load_from_checkpoint(
                checkpoint.best_model_path,
                strict=False # Allow loading even if save_hyperparameters fails internally
            )
        else:
            best_model = training_network

        return TrainOutput(
            transformation=transformation,
            trained_net=best_model,
            trainer=trainer,
            predictor=self.create_predictor(transformation, best_model, 
                                            **{k: kwargs[k] for k in kwargs if k in inspect.signature(PyTorchPredictor).parameters.keys()}), # CHANGE
        )

    @staticmethod
    def _worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ) -> TrainOutput:
        return self.train_model(
            training_data,
            validation_data,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
            ckpt_path=ckpt_path,
            **kwargs # CHANGE
        ) # CHANGE

    def train_from(
        self,
        predictor: Predictor,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        ckpt_path: Optional[str] = None,
    ) -> TrainOutput:
        assert isinstance(predictor, PyTorchPredictor)
        return self.train_model(
            training_data,
            validation_data,
            from_predictor=predictor,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
            ckpt_path=ckpt_path,
        ) # CHANGE
