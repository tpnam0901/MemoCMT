import logging
import os
from abc import ABC, abstractmethod
from typing import Dict

from flax.training.train_state import TrainState


class Callback(ABC):
    @abstractmethod
    def __call__(
        self,
        trainer,  # trainer.FlaxTrainer
        state: TrainState,
        global_step: int,
        global_epoch: int,
        logs: Dict,
        isValPhase: bool = False,
    ):
        """Abstract method to be implemented by the user.

        Args:
            trainer (trainer.Trainer): trainer.FlaxTrainer module
            state (TrainState): The state of the training.
            global_step (int): The global step of the training.
            global_epoch (int): The global epoch of the training.
            logs (Dict): The logs of the training which contains the loss and the metrics. For example:
                                                            {
                                                                "loss": 0.1,
                                                                "accuracy": 0.9
                                                                "some_custom_metric": 0.5
                                                            }
            isValPhase (bool, optional): Whether the callback is called during the validation phase. Defaults to False.
        """
        pass


class CheckpointsCallback(Callback):
    def __init__(
        self,
        checkpoint_dir: str,
        save_freq: int = 1000,
        max_to_keep: int = 3,
        save_best_val: bool = False,
    ):
        """Callback to save checkpoints during training.

        Args:
            checkpoint_dir (str): Path to the directory where checkpoints will be saved.
            save_freq (int, optional): The frequency at which checkpoints will be saved. Defaults to 1000.
            max_to_keep (int, optional): The maximum number of checkpoints to keep. Defaults to 3.
            save_best_val (bool, optional): Whether to save the best model based on the validation loss. Defaults to False.
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        self.max_to_keep = max_to_keep
        self.save_best_val = save_best_val
        if save_best_val:
            logging.warning(
                "When save_best_val is True, please make sure that you pass the validation data to the trainer.fit() method.\n\
                            Otherwise, the best model will not be saved.\n\
                            The model will save the lowest validation value if the metric starts with 'loss' and the highest value otherwise."
            )
            self.best_val = {}

    def __call__(
        self,
        trainer,  # trainer.FlaxTrainer
        state: TrainState,
        global_step: int,
        global_epoch: int,
        logs: Dict,
        isValPhase: bool = False,
    ):
        """Abstract method to be implemented by the user.
        Args:
            trainer (trainer.Trainer): trainer.FlaxTrainer module
            state (TrainState): The train state of the training.
            global_step (int): The global step of the training.
            global_epoch (int): The global epoch of the training.
            logs (Dict): The logs of the training which contains the loss and the metrics. For example:
                                                            {
                                                                "loss": 0.1,
                                                                "accuracy": 0.9
                                                                "some_custom_metric": 0.5
                                                            }
            isValPhase (bool, optional): Whether the callback is called during the validation phase. Defaults to False.
        """
        if not isValPhase:
            if global_step % self.save_freq == 0:
                trainer.save(self.checkpoint_dir, global_step, self.max_to_keep)

        elif isValPhase and self.save_best_val:
            for k, v in logs.items():
                if k not in self.best_val:
                    self.best_val[k] = v
                    logging.info(f"Saving best model based on {k} = {v}")
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    trainer.save(os.path.join(self.checkpoint_dir, "best_{}".format(k)), global_step, 1)
                else:
                    if k.startswith("loss"):
                        if v < self.best_val[k]:
                            self.best_val[k] = v
                            logging.info(f"Saving best model based on {k} = {v}")
                            os.makedirs(self.checkpoint_dir, exist_ok=True)
                            trainer.save(os.path.join(self.checkpoint_dir, "best_{}".format(k)), global_step, 1)
                    else:
                        if v > self.best_val[k]:
                            self.best_val[k] = v
                            logging.info(f"Saving best model based on {k} = {v}")
                            os.makedirs(self.checkpoint_dir, exist_ok=True)
                            trainer.save(os.path.join(self.checkpoint_dir, "best_{}".format(k)), global_step, 1)
