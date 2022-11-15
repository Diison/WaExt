# -*- coding: utf-8 -*-

"""Training KGE models based on the sLCWA."""
import logging
from typing import Optional

import torch.utils.data
from class_resolver import HintOrType, OptionalKwargs
from torch.utils.data import DataLoader

from pykeen.training.training_loop import TrainingLoop
from pykeen.sampling import NegativeSampler
from pykeen.triples import CoreTriplesFactory
from pykeen.triples.instances import SLCWABatch, SLCWASampleType
from pykeen.evaluation import Evaluator

import json
import random
import csv
import torch
from typing import Any, List, Mapping, Optional, Union
from pykeen.stoppers import Stopper
import pathlib
from pykeen.training.callbacks import (
    GradientAbsClippingTrainingCallback,
    GradientNormClippingTrainingCallback,
    MultiTrainingCallback,
    StopperTrainingCallback,
    TrackerTrainingCallback,
    TrainingCallbackHint,
    TrainingCallbackKwargsHint,
)
from pykeen.training.training_loop import get_preferred_device, get_batchnorm_modules, _get_optimizer_kwargs, _get_lr_scheduler_kwargs

import gc
import logging
import os
import pathlib
import random
import time
from datetime import datetime
from tempfile import NamedTemporaryFile

from class_resolver import HintOrType, OptionalKwargs
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange

from pykeen.training.callbacks import (
    GradientAbsClippingTrainingCallback,
    GradientNormClippingTrainingCallback,
    MultiTrainingCallback,
    StopperTrainingCallback,
    TrackerTrainingCallback,
    TrainingCallbackHint,
    TrainingCallbackKwargsHint,
)
from pykeen.constants import PYKEEN_CHECKPOINTS, PYKEEN_DEFAULT_CHECKPOINT
from pykeen.models import RGCN
from pykeen.stoppers import Stopper
from pykeen.triples import CoreTriplesFactory
from pykeen.utils import (
    format_relative_comparison,
    get_batchnorm_modules,
    get_preferred_device,
)

__all__ = [
    "SLCWATrainingLoopWaLPEval",
]

logger = logging.getLogger(__name__)

import random

class SLCWATrainingLoopWaLPEval(TrainingLoop[SLCWASampleType, SLCWABatch]):
    """A training loop that uses the stochastic local closed world assumption training approach.

    [ruffinelli2020]_ call the sLCWA ``NegSamp`` in their work.
    """

    def __init__(
        self,
        negative_sampler: HintOrType[NegativeSampler] = None,
        negative_sampler_kwargs: OptionalKwargs = None,
        lp_evaluator: Evaluator = None,
        walp_evaluator: Evaluator = None,
        save_location: str = "",
        swtc_evals = None,
        dataset: CoreTriplesFactory = None,
        step = 1,
        eval_batch: int = 8,
        **kwargs,
    ):
        """Initialize the training loop.

        :param negative_sampler: The class, instance, or name of the negative sampler
        :param negative_sampler_kwargs: Keyword arguments to pass to the negative sampler class on instantiation
            for every positive one
        :param kwargs:
            Additional keyword-based parameters passed to TrainingLoop.__init__
        """
        super().__init__(**kwargs)
        self.negative_sampler = negative_sampler
        self.negative_sampler_kwargs = negative_sampler_kwargs
        self.lp_evaluator = lp_evaluator
        self.walp_evaluator = walp_evaluator
        self.save_location = save_location
        self.swtc_evals = swtc_evals
        self.dataset = dataset
        self.eval_batch = eval_batch
        self.step = step

    def _create_training_data_loader(
        self,
        triples_factory: CoreTriplesFactory,
        batch_size: int,
        drop_last: bool,
        num_workers: int,
        pin_memory: bool,
        sampler: Optional[str],
    ) -> DataLoader[SLCWABatch]:  # noqa: D102
        return DataLoader(
            dataset=triples_factory.create_slcwa_instances(
                batch_size=batch_size,
                shuffle=True,
                drop_last=drop_last,
                negative_sampler=self.negative_sampler,
                negative_sampler_kwargs=self.negative_sampler_kwargs,
                sampler=sampler,
            ),
            num_workers=num_workers,
            pin_memory=pin_memory,
            # disable automatic batching
            batch_size=None,
            batch_sampler=None,
        )

    @staticmethod
    def _get_batch_size(batch: SLCWABatch) -> int:  # noqa: D102
        return batch[0].shape[0]

    def _process_batch(
        self,
        batch: SLCWABatch,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        # Slicing is not possible in sLCWA training loops
        if slice_size is not None:
            raise AttributeError("Slicing is not possible for sLCWA training loops.")

        # split batch
        positive_batch, negative_batch, positive_filter = batch

        # send to device
        positive_batch = positive_batch[start:stop].to(device=self.device)
        negative_batch = negative_batch[start:stop]
        if positive_filter is not None:
            positive_filter = positive_filter[start:stop]
            negative_batch = negative_batch[positive_filter]
        # Make it negative batch broadcastable (required for num_negs_per_pos > 1).
        negative_score_shape = negative_batch.shape[:-1]
        negative_batch = negative_batch.view(-1, 3)

        # Ensure they reside on the device (should hold already for most simple negative samplers, e.g.
        # BasicNegativeSampler, BernoulliNegativeSampler
        negative_batch = negative_batch.to(self.device)

        # Compute negative and positive scores
        self.model.current_epoch = self._epoch+1
        positive_scores = self.model.score_hrt(positive_batch, mode=self.mode)
        negative_scores = self.model.score_hrt(negative_batch, mode=self.mode).view(*negative_score_shape)
        # positive_scores = self.model.score_hrt(positive_batch, mode=True)
        # negative_scores = self.model.score_hrt(negative_batch, mode=False)
        # negative_scores = negative_scores.view(*negative_score_shape)

        # base_loss = self.loss.process_slcwa_scores(
        #         positive_scores=positive_scores,
        #         negative_scores=negative_scores,
        #         label_smoothing=label_smoothing,
        #         batch_filter=positive_filter,
        #         num_entities=self.model._get_entity_len(mode=self.mode),
        # ) + self.model.collect_regularization_term()
        
        # total_loss = (1-alpha-beta)*base_loss + alpha*pos_acc_loss + beta*neg_acc_loss

        # return total_loss

        return (
            self.loss.process_slcwa_scores(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                label_smoothing=label_smoothing,
                batch_filter=positive_filter,
                num_entities=self.model._get_entity_len(mode=self.mode),
            )
            + self.model.collect_regularization_term()
        )

    def _slice_size_search(
        self,
        *,
        triples_factory: CoreTriplesFactory,
        batch_size: int,
        sub_batch_size: int,
        supports_sub_batching: bool,
    ):  # noqa: D102
        # Slicing is not possible for sLCWA
        if supports_sub_batching:
            report = "This model supports sub-batching, but it also requires slicing, which is not possible for sLCWA"
        else:
            report = "This model doesn't support sub-batching and slicing is not possible for sLCWA"
        logger.warning(report)
        raise MemoryError("The current model can't be trained on this hardware with these parameters.")
















    def train(
        self,
        triples_factory: CoreTriplesFactory,
        num_epochs: int = 1,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        label_smoothing: float = 0.0,
        sampler: Optional[str] = None,
        continue_training: bool = False,
        only_size_probing: bool = False,
        use_tqdm: bool = True,
        use_tqdm_batch: bool = True,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
        stopper: Optional[Stopper] = None,
        sub_batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        clear_optimizer: bool = False,
        checkpoint_directory: Union[None, str, pathlib.Path] = None,
        checkpoint_name: Optional[str] = None,
        checkpoint_frequency: Optional[int] = None,
        checkpoint_on_failure: bool = False,
        drop_last: Optional[bool] = None,
        callbacks: TrainingCallbackHint = None,
        callback_kwargs: TrainingCallbackKwargsHint = None,
        gradient_clipping_max_norm: Optional[float] = None,
        gradient_clipping_norm_type: Optional[float] = None,
        gradient_clipping_max_abs_value: Optional[float] = None,
        pin_memory: bool = True,
    ) -> Optional[List[float]]:
        """Train the KGE model.

        .. note ::
            Gradient clipping is a technique to avoid the exploding gradient problem. Clip by norm and clip by value
            are two alternative implementations.


        :param triples_factory:
            The training triples.
        :param num_epochs:
            The number of epochs to train the model.
        :param batch_size:
            If set the batch size to use for mini-batch training. Otherwise find the largest possible batch_size
            automatically.
        :param slice_size: >0
            The divisor for the scoring function when using slicing. This is only possible for LCWA training loops in
            general and only for models that have the slicing capability implemented.
        :param label_smoothing: (0 <= label_smoothing < 1)
            If larger than zero, use label smoothing.
        :param sampler: (None or 'schlichtkrull')
            The type of sampler to use. At the moment sLCWA in R-GCN is the only user of schlichtkrull sampling.
        :param continue_training:
            If set to False, (re-)initialize the model's weights. Otherwise continue training.
        :param only_size_probing:
            The evaluation is only performed for two batches to test the memory footprint, especially on GPUs.
        :param use_tqdm: Should a progress bar be shown for epochs?
        :param use_tqdm_batch: Should a progress bar be shown for batching (inside the epoch progress bar)?
        :param tqdm_kwargs:
            Keyword arguments passed to :mod:`tqdm` managing the progress bar.
        :param stopper:
            An instance of :class:`pykeen.stopper.EarlyStopper` with settings for checking
            if training should stop early
        :param sub_batch_size:
            If provided split each batch into sub-batches to avoid memory issues for large models / small GPUs.
        :param num_workers:
            The number of child CPU workers used for loading data. If None, data are loaded in the main process.
        :param clear_optimizer:
            Whether to delete the optimizer instance after training (as the optimizer might have additional memory
            consumption due to e.g. moments in Adam).
        :param checkpoint_directory:
            An optional directory to store the checkpoint files. If None, a subdirectory named ``checkpoints`` in the
            directory defined by :data:`pykeen.constants.PYKEEN_HOME` is used. Unless the environment variable
            ``PYKEEN_HOME`` is overridden, this will be ``~/.pykeen/checkpoints``.
        :param checkpoint_name:
            The filename for saving checkpoints. If the given filename exists already, that file will be loaded and used
            to continue training.
        :param checkpoint_frequency:
            The frequency of saving checkpoints in minutes. Setting it to 0 will save a checkpoint after every epoch.
        :param checkpoint_on_failure:
            Whether to save a checkpoint in cases of a RuntimeError or MemoryError. This option differs from ordinary
            checkpoints, since ordinary checkpoints are only saved after a successful epoch. When saving checkpoints
            due to failure of the training loop there is no guarantee that all random states can be recovered correctly,
            which might cause problems with regards to the reproducibility of that specific training loop. Therefore,
            these checkpoints are saved with a distinct checkpoint name, which will be
            ``PyKEEN_just_saved_my_day_{datetime}.pt`` in the given checkpoint_root.
        :param drop_last:
            Whether to drop the last batch in each epoch to prevent smaller batches. Defaults to False, except if the
            model contains batch normalization layers. Can be provided explicitly to override.
        :param callbacks:
            An optional :class:`pykeen.training.TrainingCallback` or collection of callback instances that define
            one of several functionalities. Their interface was inspired by Keras.
        :param callback_kwargs:
            additional keyword-based parameter to instantiate the training callback.
        :param gradient_clipping_max_norm:
            The maximum gradient norm for use with gradient clipping. If None, no gradient norm clipping is used.
        :param gradient_clipping_norm_type:
            The gradient norm type to use for maximum gradient norm, cf. :func:`torch.nn.utils.clip_grad_norm_`
        :param gradient_clipping_max_abs_value:
            The maximum absolute value in gradients, cf. :func:`torch.nn.utils.clip_grad_value_`. If None, no
            gradient clipping will be used.
        :param pin_memory:
            whether to use memory pinning in the data loader, cf.
            https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning

        :return:
            The losses per epoch.
        """
        self._should_stop = False

        # In some cases, e.g. using Optuna for HPO, the cuda cache from a previous run is not cleared
        torch.cuda.empty_cache()

        # A checkpoint root is always created to ensure a fallback checkpoint can be saved
        if checkpoint_directory is None:
            checkpoint_directory = PYKEEN_CHECKPOINTS
        checkpoint_directory = pathlib.Path(checkpoint_directory)
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        logger.debug("using checkpoint_root at %s", checkpoint_directory)

        # If a checkpoint file is given, it must be loaded if it exists already
        save_checkpoints = False
        checkpoint_path = None
        best_epoch_model_file_path = None
        last_best_epoch = None
        if checkpoint_name:
            checkpoint_path = checkpoint_directory.joinpath(checkpoint_name)
            if checkpoint_path.is_file():
                best_epoch_model_file_path, last_best_epoch = self._load_state(
                    path=checkpoint_path,
                    triples_factory=triples_factory,
                )
                if stopper is not None:
                    stopper_dict = stopper.load_summary_dict_from_training_loop_checkpoint(path=checkpoint_path)
                    # If the stopper dict has any keys, those are written back to the stopper
                    if stopper_dict:
                        stopper._write_from_summary_dict(**stopper_dict)
                    else:
                        logger.warning(
                            "the training loop was configured with a stopper but no stopper configuration was "
                            "saved in the checkpoint",
                        )
                continue_training = True
            else:
                logger.info(f"=> no checkpoint found at '{checkpoint_path}'. Creating a new file.")
            # The checkpoint frequency needs to be set to save checkpoints
            if checkpoint_frequency is None:
                checkpoint_frequency = 30
            save_checkpoints = True
        elif checkpoint_frequency is not None:
            logger.warning(
                "A checkpoint frequency was set, but no checkpoint file was given. No checkpoints will be created",
            )

        checkpoint_on_failure_file_path = None
        if checkpoint_on_failure:
            # In case a checkpoint frequency was set, we warn that no checkpoints will be saved
            date_string = datetime.now().strftime("%Y%m%d_%H_%M_%S")
            # If no checkpoints were requested, a fallback checkpoint is set in case the training loop crashes
            checkpoint_on_failure_file_path = checkpoint_directory.joinpath(
                PYKEEN_DEFAULT_CHECKPOINT.replace(".", f"_{date_string}."),
            )

        # If the stopper loaded from the training loop checkpoint stopped the training, we return those results
        if getattr(stopper, "stopped", False):
            result: Optional[List[float]] = self.losses_per_epochs
        else:
            # send model to device before going into the internal training loop
            self.model = self.model.to(get_preferred_device(self.model, allow_ambiguity=True))
            result = self._train_eval(
                num_epochs=num_epochs,
                batch_size=batch_size,
                slice_size=slice_size,
                label_smoothing=label_smoothing,
                sampler=sampler,
                continue_training=continue_training,
                only_size_probing=only_size_probing,
                use_tqdm=use_tqdm,
                use_tqdm_batch=use_tqdm_batch,
                tqdm_kwargs=tqdm_kwargs,
                stopper=stopper,
                sub_batch_size=sub_batch_size,
                num_workers=num_workers,
                save_checkpoints=save_checkpoints,
                checkpoint_path=checkpoint_path,
                checkpoint_frequency=checkpoint_frequency,
                checkpoint_on_failure_file_path=checkpoint_on_failure_file_path,
                best_epoch_model_file_path=best_epoch_model_file_path,
                last_best_epoch=last_best_epoch,
                drop_last=drop_last,
                callbacks=callbacks,
                callback_kwargs=callback_kwargs,
                gradient_clipping_max_norm=gradient_clipping_max_norm,
                gradient_clipping_norm_type=gradient_clipping_norm_type,
                gradient_clipping_max_abs_value=gradient_clipping_max_abs_value,
                triples_factory=triples_factory,
                pin_memory=pin_memory,
            )

        # Ensure the release of memory
        torch.cuda.empty_cache()

        # Clear optimizer
        if clear_optimizer:
            self.optimizer = None
            self.lr_scheduler = None

        return result

















    def _train_eval(  # noqa: C901
        self,
        triples_factory: CoreTriplesFactory,
        num_epochs: int = 1,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        label_smoothing: float = 0.0,
        sampler: Optional[str] = None,
        continue_training: bool = False,
        only_size_probing: bool = False,
        use_tqdm: bool = True,
        use_tqdm_batch: bool = True,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
        stopper: Optional[Stopper] = None,
        sub_batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        save_checkpoints: bool = False,
        checkpoint_path: Union[None, str, pathlib.Path] = None,
        checkpoint_frequency: Optional[int] = None,
        checkpoint_on_failure_file_path: Union[None, str, pathlib.Path] = None,
        best_epoch_model_file_path: Optional[pathlib.Path] = None,
        last_best_epoch: Optional[int] = None,
        drop_last: Optional[bool] = None,
        callbacks: TrainingCallbackHint = None,
        callback_kwargs: TrainingCallbackKwargsHint = None,
        gradient_clipping_max_norm: Optional[float] = None,
        gradient_clipping_norm_type: Optional[float] = None,
        gradient_clipping_max_abs_value: Optional[float] = None,
        pin_memory: bool = True,
    ) -> Optional[List[float]]:
        """Train the KGE model, see docstring for :func:`TrainingLoop.train`."""
        if self.optimizer is None:
            raise ValueError("optimizer must be set before running _train()")
        # When using early stopping models have to be saved separately at the best epoch, since the training loop will
        # due to the patience continue to train after the best epoch and thus alter the model
        if (
            stopper is not None
            and not only_size_probing
            and last_best_epoch is None
            and best_epoch_model_file_path is None
        ):
            # Create a path
            best_epoch_model_file_path = pathlib.Path(NamedTemporaryFile().name)
        best_epoch_model_checkpoint_file_path: Optional[pathlib.Path] = None

        if isinstance(self.model, RGCN) and sampler != "schlichtkrull":
            logger.warning(
                'Using RGCN without graph-based sampling! Please select sampler="schlichtkrull" instead of %s.',
                sampler,
            )

        # Prepare all of the callbacks
        callback = MultiTrainingCallback(callbacks=callbacks, callback_kwargs=callback_kwargs)
        # Register a callback for the result tracker, if given
        if self.result_tracker is not None:
            callback.register_callback(TrackerTrainingCallback())
        # Register a callback for the early stopper, if given
        # TODO should mode be passed here?
        if stopper is not None:
            callback.register_callback(
                StopperTrainingCallback(
                    stopper,
                    triples_factory=triples_factory,
                    last_best_epoch=last_best_epoch,
                    best_epoch_model_file_path=best_epoch_model_file_path,
                )
            )
        if gradient_clipping_max_norm is not None:
            callback.register_callback(
                GradientNormClippingTrainingCallback(
                    max_norm=gradient_clipping_max_norm,
                    norm_type=gradient_clipping_norm_type,
                )
            )
        if gradient_clipping_max_abs_value is not None:
            callback.register_callback(GradientAbsClippingTrainingCallback(clip_value=gradient_clipping_max_abs_value))

        callback.register_training_loop(self)

        # Take the biggest possible training batch_size, if batch_size not set
        batch_size_sufficient = False
        if batch_size is None:
            if self.automatic_memory_optimization:
                # Using automatic memory optimization on CPU may result in undocumented crashes due to OS' OOM killer.
                if self.model.device.type == "cpu":
                    batch_size = 256
                    batch_size_sufficient = True
                    logger.info(
                        "Currently automatic memory optimization only supports GPUs, but you're using a CPU. "
                        "Therefore, the batch_size will be set to the default value '{batch_size}'",
                    )
                else:
                    batch_size, batch_size_sufficient = self.batch_size_search(triples_factory=triples_factory)
            else:
                batch_size = 256
                logger.info(f"No batch_size provided. Setting batch_size to '{batch_size}'.")

        # This will find necessary parameters to optimize the use of the hardware at hand
        if (
            not only_size_probing
            and self.automatic_memory_optimization
            and not batch_size_sufficient
            and not continue_training
        ):
            # return the relevant parameters slice_size and batch_size
            sub_batch_size, slice_size = self.sub_batch_and_slice(
                batch_size=batch_size, sampler=sampler, triples_factory=triples_factory
            )

        if sub_batch_size is None or sub_batch_size == batch_size:  # by default do not split batches in sub-batches
            sub_batch_size = batch_size
        elif get_batchnorm_modules(self.model):  # if there are any, this is truthy
            raise SubBatchingNotSupportedError(self.model)

        model_contains_batch_norm = bool(get_batchnorm_modules(self.model))
        if batch_size == 1 and model_contains_batch_norm:
            raise ValueError("Cannot train a model with batch_size=1 containing BatchNorm layers.")

        if drop_last is None:
            drop_last = model_contains_batch_norm

        # Force weight initialization if training continuation is not explicitly requested.
        if not continue_training:
            # Reset the weights
            self.model.reset_parameters_()
            # afterwards, some parameters may be on the wrong device
            self.model.to(get_preferred_device(self.model, allow_ambiguity=True))

            # Create new optimizer
            optimizer_kwargs = _get_optimizer_kwargs(self.optimizer)
            self.optimizer = self.optimizer.__class__(
                params=self.model.get_grad_params(),
                **optimizer_kwargs,
            )

            if self.lr_scheduler is not None:
                # Create a new lr scheduler and add the optimizer
                lr_scheduler_kwargs = _get_lr_scheduler_kwargs(self.lr_scheduler)
                self.lr_scheduler = self.lr_scheduler.__class__(self.optimizer, **lr_scheduler_kwargs)
        elif not self.optimizer.state:
            raise ValueError("Cannot continue_training without being trained once.")

        # Ensure the model is on the correct device
        self.model.to(get_preferred_device(self.model, allow_ambiguity=True))

        if num_workers is None:
            num_workers = 0

        _use_outer_tqdm = not only_size_probing and use_tqdm
        _use_inner_tqdm = _use_outer_tqdm and use_tqdm_batch

        # When size probing, we don't want progress bars
        if _use_outer_tqdm:
            # Create progress bar
            _tqdm_kwargs = dict(desc=f"Training epochs on {self.device}", unit="epoch")
            if tqdm_kwargs is not None:
                _tqdm_kwargs.update(tqdm_kwargs)
            epochs = trange(self._epoch + 1, 1 + num_epochs, **_tqdm_kwargs, initial=self._epoch, total=num_epochs)
        elif only_size_probing:
            epochs = range(1, 1 + num_epochs)
        else:
            epochs = range(self._epoch + 1, 1 + num_epochs)

        logger.debug(f"using stopper: {stopper}")

        train_data_loader = self._create_training_data_loader(
            triples_factory,
            batch_size,
            drop_last,
            num_workers,
            pin_memory,
            sampler=sampler,
        )
        if len(train_data_loader) == 0:
            raise NoTrainingBatchError()
        if drop_last and not only_size_probing:
            logger.info(
                "Dropping last (incomplete) batch each epoch (%s batches).",
                format_relative_comparison(part=1, total=len(train_data_loader)),
            )

        # Save the time to track when the saved point was available
        last_checkpoint = time.time()

        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)
        if not os.path.exists(self.save_location+"/tri_ranks"):
            os.makedirs(self.save_location+"/tri_ranks")
            # os.makedirs(self.save_location+"/walp_result")

        lp_f = open(self.save_location+"/_lp_result.csv", "w")
        lp_writer = csv.writer(lp_f)
        lp_writer.writerow(["Epoch", "MR", "MRR", "Hit1", "Hit3", "Hit5", "Hit10"])
        walp_f = open(self.save_location+"/_walp_result.csv", "w")
        walp_writer = csv.writer(walp_f)
        walp_writer.writerow(["Epoch", "MR", "MRR", "Hit1", "Hit3", "Hit5", "Hit10"])

        watc_writers={}
        for k, v in self.swtc_evals.items():
            watc_f = open(self.save_location+"/_watc_result_"+k+".csv", "w")
            watc_writer = csv.writer(watc_f)
            watc_writer.writerow(["Epoch", "F1", "WaF1", "Precision", "Recall", "WaPrecision", "WaRecall"])
            watc_writers[k]=watc_writer
            
        # Training Loop
        for epoch in epochs:
            # When training with an early stopper the memory pressure changes, which may allow for errors each epoch
            try:
                # Enforce training mode
                self.model.train()

                # Accumulate loss over epoch
                current_epoch_loss = 0.0

                # Batching
                # Only create a progress bar when not in size probing mode
                if _use_inner_tqdm:
                    batches = tqdm(
                        train_data_loader,
                        desc=f"Training batches on {self.device}",
                        leave=False,
                        unit="batch",
                    )
                else:
                    batches = train_data_loader

                # Flag to check when to quit the size probing
                evaluated_once = False

                num_training_instances = 0
                for batch in batches:
                    # Recall that torch *accumulates* gradients. Before passing in a
                    # new instance, you need to zero out the gradients from the old instance
                    self.optimizer.zero_grad()

                    # Get batch size of current batch (last batch may be incomplete)
                    current_batch_size = self._get_batch_size(batch)
                    _sub_batch_size = sub_batch_size or current_batch_size

                    # accumulate gradients for whole batch
                    for start in range(0, current_batch_size, _sub_batch_size):
                        stop = min(start + _sub_batch_size, current_batch_size)

                        # forward pass call
                        batch_loss = self._forward_pass(
                            batch,
                            start,
                            stop,
                            current_batch_size,
                            label_smoothing,
                            slice_size,
                        )
                        current_epoch_loss += batch_loss
                        num_training_instances += stop - start
                        callback.on_batch(epoch=epoch, batch=batch, batch_loss=batch_loss)

                    # when called by batch_size_search(), the parameter update should not be applied.
                    if not only_size_probing:
                        callback.pre_step()

                        # update parameters according to optimizer
                        self.optimizer.step()

                    # After changing applying the gradients to the embeddings, the model is notified that the forward
                    # constraints are no longer applied
                    self.model.post_parameter_update()

                    # For testing purposes we're only interested in processing one batch
                    if only_size_probing and evaluated_once:
                        break

                    callback.post_batch(epoch=epoch, batch=batch)

                    evaluated_once = True

                del batch
                del batches
                gc.collect()
                self.optimizer.zero_grad()
                self._free_graph_and_cache()

                # When size probing we don't need the losses
                if only_size_probing:
                    return None

                # Update learning rate scheduler
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(epoch=epoch)

                # Track epoch loss
                if self.model.loss.reduction == "mean":
                    epoch_loss = current_epoch_loss / num_training_instances
                else:
                    epoch_loss = current_epoch_loss / len(train_data_loader)
                self.losses_per_epochs.append(epoch_loss)

                # Print loss information to console
                if _use_outer_tqdm:
                    epochs.set_postfix(
                        {
                            "loss": self.losses_per_epochs[-1],
                            "prev_loss": self.losses_per_epochs[-2] if epoch > 1 else float("nan"),
                        }
                    )

                # Save the last successful finished epoch
                self._epoch = epoch

            # When the training loop failed, a fallback checkpoint is created to resume training.
            except (MemoryError, RuntimeError) as e:
                # During automatic memory optimization only the error message is of interest
                if only_size_probing:
                    raise e

                logger.warning(f"The training loop just failed during epoch {epoch} due to error {str(e)}.")
                if checkpoint_on_failure_file_path:
                    # When there wasn't a best epoch the checkpoint path should be None
                    if last_best_epoch is not None and best_epoch_model_file_path is not None:
                        best_epoch_model_checkpoint_file_path = best_epoch_model_file_path
                    self._save_state(
                        path=checkpoint_on_failure_file_path,
                        stopper=stopper,
                        best_epoch_model_checkpoint_file_path=best_epoch_model_checkpoint_file_path,
                        triples_factory=triples_factory,
                    )
                    logger.warning(
                        "However, don't worry we got you covered. PyKEEN just saved a checkpoint when this "
                        f"happened at '{checkpoint_on_failure_file_path}'. To resume training from the checkpoint "
                        f"file just restart your code and pass this file path to the training loop or pipeline you "
                        f"used as 'checkpoint_file' argument.",
                    )
                # Delete temporary best epoch model
                if best_epoch_model_file_path is not None and best_epoch_model_file_path.is_file():
                    os.remove(best_epoch_model_file_path)
                raise e

            
            
            if epoch==1 or epoch%self.step==0:

                lp_result = self.lp_evaluator.evaluate(self.model, self.dataset._testing.mapped_triples, batch_size=self.eval_batch, additional_filter_triples=[self.dataset._training.mapped_triples, self.dataset._validation.mapped_triples])
                lp_result=lp_result.to_dict()
                lp_writer.writerow([epoch, lp_result["both"]["realistic"]["arithmetic_mean_rank"], lp_result["both"]["realistic"]["inverse_arithmetic_mean_rank"], lp_result["both"]["realistic"]["hits_at_1"], lp_result["both"]["realistic"]["hits_at_3"], lp_result["both"]["realistic"]["hits_at_5"], lp_result["both"]["realistic"]["hits_at_10"]])
                print("\nlp: ", epoch, lp_result["both"]["realistic"]["arithmetic_mean_rank"], lp_result["both"]["realistic"]["inverse_arithmetic_mean_rank"])

                tri_ranks_f = open(self.save_location+"/tri_ranks/"+str(epoch)+".csv", "w")
                tri_ranks_writer = csv.writer(tri_ranks_f)
                tri_ranks_writer.writerow(["Rank", "Weight", "H", "R", "T"])
                self.walp_evaluator.outter = tri_ranks_writer
                walp_result = self.walp_evaluator.evaluate(self.model, self.dataset._testing.mapped_triples, batch_size=self.eval_batch, additional_filter_triples=[self.dataset._training.mapped_triples, self.dataset._validation.mapped_triples])
                tri_ranks_f.close()
                walp_result=walp_result.to_dict()
                walp_writer.writerow([epoch, walp_result["both"]["realistic"]["arithmetic_mean_rank"], walp_result["both"]["realistic"]["inverse_arithmetic_mean_rank"], walp_result["both"]["realistic"]["hits_at_1"], walp_result["both"]["realistic"]["hits_at_3"], walp_result["both"]["realistic"]["hits_at_5"], walp_result["both"]["realistic"]["hits_at_10"]])
                print("\nwalp: ", epoch, walp_result["both"]["realistic"]["arithmetic_mean_rank"], walp_result["both"]["realistic"]["inverse_arithmetic_mean_rank"])
                print(self.save_location, "\n")

                for k, evaluator in self.swtc_evals.items():
                    watc_result = evaluator.evaluate(self.model, self.dataset._testing.mapped_triples, batch_size=self.eval_batch, additional_filter_triples=[self.dataset._training.mapped_triples, self.dataset._validation.mapped_triples])
                    watc_writers[k].writerow([epoch, watc_result.data['f1'], watc_result.data['wa_f1'], watc_result.data['precision'], watc_result.data['recall'], watc_result.data["wa_precision"], watc_result.data["wa_recall"]])
                


            # Includes a call to result_tracker.log_metrics
            callback.post_epoch(epoch=epoch, epoch_loss=epoch_loss)

            # If a checkpoint file is given, we check whether it is time to save a checkpoint
            if save_checkpoints and checkpoint_path is not None:
                minutes_since_last_checkpoint = (time.time() - last_checkpoint) // 60
                # MyPy overrides are because you should
                if (
                    minutes_since_last_checkpoint >= checkpoint_frequency  # type: ignore
                    or self._should_stop
                    or epoch == num_epochs
                ):
                    # When there wasn't a best epoch the checkpoint path should be None
                    if last_best_epoch is not None and best_epoch_model_file_path is not None:
                        best_epoch_model_checkpoint_file_path = best_epoch_model_file_path
                    self._save_state(
                        path=checkpoint_path,
                        stopper=stopper,
                        best_epoch_model_checkpoint_file_path=best_epoch_model_checkpoint_file_path,
                        triples_factory=triples_factory,
                    )  # type: ignore
                    last_checkpoint = time.time()

            if self._should_stop:
                if last_best_epoch is not None and best_epoch_model_file_path is not None:
                    self._load_state(path=best_epoch_model_file_path)
                    # Delete temporary best epoch model
                    if pathlib.Path.is_file(best_epoch_model_file_path):
                        os.remove(best_epoch_model_file_path)
                return self.losses_per_epochs

        callback.post_train(losses=self.losses_per_epochs)

        # If the stopper didn't stop the training loop but derived a best epoch, the model has to be reconstructed
        # at that state
        if stopper is not None and last_best_epoch is not None and best_epoch_model_file_path is not None:
            self._load_state(path=best_epoch_model_file_path)
            # Delete temporary best epoch model
            if pathlib.Path.is_file(best_epoch_model_file_path):
                os.remove(best_epoch_model_file_path)

        return self.losses_per_epochs

