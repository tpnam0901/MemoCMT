import datetime
import inspect
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import optax
import tqdm
from flax import linen as nn
from flax.training import checkpoints, train_state

from . import optimizers
from .callbacks import Callback

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class TrainStateWithBatchStats(train_state.TrainState):
    batch_stats: Any


class FlaxTrainer(ABC, nn.Module):
    log_dir: str = "logs"

    def predict(self, inputs: Union[jnp.ndarray, Dict, List]) -> Union[jnp.ndarray, Dict, List]:
        """

        Args:
            inputs (Union[jnp.ndarray, Dict, List]): Inputs to the model

        Returns:
            Union[jnp.ndarray, Dict, List]: Predictions
        """

        # Use state.params to get the parameters of the model if available
        try:
            params = self.state.params
        except:
            params = self.params

        # Use state.batch_stats to get the batch stats of the model if available
        variables = {"params": params}
        if self.batch_stats is not None:
            try:
                variables["batch_stats"] = self.state.batch_stats
            except:
                variables["batch_stats"] = self.batch_stats
        logits = self.state.apply_fn(variables, inputs, training=False)
        return logits

    def train_epoch(
        self,
        epoch: int,
        train_data: Iterable,
        eval_data: Iterable = None,
        logger: logging.Logger = None,
        callbacks: List[Callback] = None,
    ):
        """Performs one epoch of training and validation.

        Args:
            epoch (int): Current epoch.
            train_data (Iterable): training data.
            eval_data (Iterable, optional): validation data. Defaults to None.
            logger (logging.Logger, optional): logger used for logging. Defaults to None.
            callbacks (List[Callback], optional): List of callbacks. Defaults to None.
        """
        if logger is None:
            logger = logging

        epoch_log = {}
        with tqdm.tqdm(total=len(train_data), ascii=True) as pbar:
            pbar.update(1)
            for batch in train_data:
                # Training step
                train_log = self.train_step(batch)
                assert isinstance(train_log, dict), "train_step should return a dict."
                # Add logs, update progress bar
                postfix = ""
                for key, value in train_log.items():
                    e_val = epoch_log.get(key, [])
                    e_val.append(value)
                    epoch_log.update({key: e_val})
                    postfix += f"{key}: {value:.4f} "
                    mlflow.log_metric(f"train_{key}", value)
                pbar.set_description(postfix)
                pbar.update(1)

                # Try to log learning rate if optax is implement with hyperparams injection
                try:
                    mlflow.log_metric(f"learning_rate", self.state.opt_state.hyperparams["learning_rate"])
                except:
                    pass

                # Callbacks
                if callbacks is not None:
                    for callback in callbacks:
                        callback(self, self.state, self.state.step, epoch, train_log, isValPhase=False)
        for key, value in epoch_log.items():
            logger.info(f"Epoch {epoch} - {key}: {np.mean(value):.4f}")
        if eval_data is not None:
            logger.info("Performing validation...")
            # First pass to retrieve keys
            val_log = self.test_step(batch)
            assert isinstance(val_log, dict), "val_step should return a dict."
            eval_logs = {key: [] for key in val_log.keys()}

            # Perform validation
            for batch in tqdm.tqdm(eval_data, ascii=True):
                val_log = self.test_step(batch)
                for key, value in val_log.items():
                    eval_logs[key].append(value)

            # Log validation metrics
            postfix = ""
            for key, value in eval_logs.items():
                postfix += f"{key}: {np.mean(value):.4f} "
                mlflow.log_metric(f"val_{key}", np.mean(value))
            logger.info("Validation: " + postfix)

            # Callbacks
            if callbacks is not None:
                eval_logs = {key: np.mean(value) for key, value in eval_logs.items()}
                for callback in callbacks:
                    callback(self, self.state, self.state.step, epoch, eval_logs, isValPhase=True)

    def evaluate(
        self,
        test_data: Iterable,
        logger: logging.Logger = None,
    ) -> Dict:
        """Performs evaluation on the test set.

        Args:
            test_data (Iterable): Test data.
            logger (logging.Logger, optional): Logger. Defaults to None.

        Returns:
            Dict: The evaluation metrics in a dictionary with the metric name as key and the metric value as value.
        """
        if logger is None:
            logger = logging
        test_logs = {}

        for batch in tqdm.tqdm(test_data, ascii=True):
            # Perform validation
            test_log = self.test_step(batch)
            for key, value in test_log.items():
                v = test_logs.get(key, [])
                v.append(value)
                test_logs.update({key: v})
        # Log validation metrics
        postfix = ""
        for key, value in test_logs.items():
            postfix += f"{key}: {np.mean(value):.4f} "
            try:
                mlflow.log_metric(f"test_{key}", np.mean(value))
            except:
                logger.warning(f"Could not log test metric {key} using mlflow.")
        logger.info("Test: " + postfix)

    def build(self, inputs: jnp.ndarray, seeds: int = 42):
        """Builds the model and initializes the parameters.

        Args:
            inputs (jnp.ndarray): Dummy inputs to initialize the model.
            seeds (int, optional): Random seeds. Defaults to 42.
        """
        args = inspect.signature(self.__call__).parameters
        assert "training" in args, "The model must have a training argument in the __call__ method"

        # Assign dummy inputs for summary
        self.dummy_inputs = inputs
        # Prepare random seeds for initialization
        self.root_rng, self.dropout_rng, self.batch_stats_rng = jax.random.split(jax.random.PRNGKey(seeds), 3)

        # Initialize the variables, check if batch stats are present
        variables = self.init(self.root_rng, inputs, training=False)
        self.params = variables["params"]
        self.batch_stats = variables.get("batch_stats", None)

    def summary(self):
        """Prints a summary of the model."""
        print(self.tabulate(self.root_rng, self.dummy_inputs, training=False))

    def save(self, path: str, step: int, keep: int = 3):
        """Save the model to the checkpoint directory.

        Args:
            path (str): Path to the checkpoint directory.
            step (int): The current step.
            keep (int, optional): Number of checkpoints to keep. Defaults to 3.
        """
        checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(path), target=self.state, step=step, keep=keep)

    def load(self, path: str):
        """Load the model from the checkpoint directory.

        Args:
            ckpt_dir (str): Path to the checkpoint directory.
        """
        assert os.path.exists(path), "Checkpoint directory is not found at {}".format(path)
        raw_state_dict = checkpoints.restore_checkpoint(path, None)
        if raw_state_dict.get("batch_stats", None) is not None:
            raise NotImplementedError("TODO load batch stats")
        else:
            checkpoints.restore_checkpoint(ckpt_dir=path, target=self.state)

    def save_weights(self, path: str):
        raise NotImplementedError("TODO save weights")

    def load_weights(self, path: str):
        raise NotImplementedError("TODO load weights")

    def compile(self, optimizer: Union[str, optax.GradientTransformation] = "sgd"):
        """Compile the model with the given optimizer.

        Args:
            optimizer (Union[str, optax.GradientTransformation], optional): The optimizer to use. Defaults to "sgd".

        Raises:
            AttributeError: This method must be called after the model is built.
            NotImplementedError: The given optimizer is not implemented.
        """
        try:
            self.params
        except AttributeError:
            raise AttributeError("Please build the model first!")
        assert isinstance(optimizer, (str, optax.GradientTransformation)), "Optimizer must be a string or optax object"

        if type(optimizer) == str:
            available_optimizers = {
                "sgd": optimizers.sgd(learning_rate=0.01),
                "adam": optimizers.adam(learning_rate=0.01),
                "rmsprop": optimizers.rmsprop(learning_rate=0.01),
                "adagrad": optimizers.adagrad(learning_rate=0.01),
                "adafactor": optimizers.adafactor(learning_rate=0.01),
                "adamw": optimizers.adamw(learning_rate=0.01, weight_decay=0.01),
            }
            optimizer = available_optimizers.get(optimizer, None)
            if optimizer is None:
                raise NotImplementedError(
                    "{} is not found. List of available optimizers: {}".format(optimizer, list(available_optimizers.keys()))
                )
        # Create flax train state
        if self.batch_stats is not None:
            self.state = TrainStateWithBatchStats.create(
                apply_fn=self.apply, params=self.params, tx=optimizer, batch_stats=self.batch_stats
            )
        else:
            self.state = train_state.TrainState.create(apply_fn=self.apply, params=self.params, tx=optimizer)

    def fit(
        self,
        train_data: Iterable,
        epochs: int,
        eval_data: Iterable = None,
        test_data: Iterable = None,
        callbacks: List[Callback] = None,
    ):
        """Hyper API for training the model.

        Args:
            epochs (int): Number of epochs to train.
            train_data (Iterable): Training data.
            eval_data (Iterable, optional): Evaluation data. Defaults to None.
            test_data (Iterable, optional): Test data. Defaults to None.
            callbacks (List[Callback], optional): List of callbacks which will be called during training. Defaults to None.
        Raises:
            AttributeError: This method must be called after the model is compiled.
        """
        try:
            self.state
        except AttributeError:
            raise AttributeError("Please compile the model first!")

        assert isinstance(callbacks, list) or callbacks is None, "Callbacks must be a list of Callback objects"

        # Logger
        logger = logging.getLogger("Training")

        # Init mlflow
        self.log_dir = os.path.join(self.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        mlflow.set_tracking_uri(uri=f'file://{os.path.abspath(os.path.join(self.log_dir, "mlruns"))}')

        # Start training
        with mlflow.start_run():
            for epoch in range(1, epochs + 1):
                logger.info(f"Epoch {epoch}/{epochs}")
                self.train_epoch(epoch, train_data, eval_data, logger, callbacks=callbacks)
                if test_data is not None:
                    self.evaluate(test_data)

    @abstractmethod
    def train_step(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Train step for the model.

        Args:
            batch (Dict[str, jnp.ndarray]): Your inputs should be compressed into a dictionary or list.
            For example, {"inputs_1": inputs_1, "inputs_2": inputs_2} or [inputs_1, inputs_2]

        Returns:
            Dict[str, jnp.ndarray]: The outputs must be a dictionary which contains the information that you want to log.
            For example, {"loss": loss, "metrics": metrics}, loss and metrics need to be a scalar.
        """
        pass

    @abstractmethod
    def test_step(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Test step for the model.

        Args:
            batch (Dict[str, jnp.ndarray]): Your inputs should be compressed into a dictionary or list.
            For example, {"inputs_1": inputs_1, "inputs_2": inputs_2} or [inputs_1, inputs_2]

        Returns:
            Dict[str, jnp.ndarray]: The outputs must be a dictionary which contains the information that you want to log.
            For example, {"loss": loss, "metrics": metrics}, loss and metrics need to be a scalar.
        """
        pass
