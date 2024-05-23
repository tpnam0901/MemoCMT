import logging
import os
import shutil
from abc import ABC, abstractmethod
from typing import Dict


class Callback(ABC):
    @abstractmethod
    def __call__(
        self,
        trainer,  #: TorchTrainer,
        global_step: int,
        global_epoch: int,
        logs: Dict,
        isValPhase: bool = False,
        logger: logging = None,
    ):
        """Abstract method to be implemented by the user.

        Args:
            trainer (trainer.Trainer): trainer.TorchTrainer module
            global_step (int): The global step of the training.
            global_epoch (int): The global epoch of the training.
            logs (Dict): The logs of the training which contains the loss and the metrics. For example:
                                                            {
                                                                "loss": 0.1,
                                                                "accuracy": 0.9
                                                                "some_custom_metric": 0.5
                                                            }
            isValPhase (bool, optional): Whether the callback is called during the validation phase. Defaults to False.
            logger (logging, optional): The logger to be used. Defaults to None.
        """
        pass


class CheckpointsCallback(Callback):
    def __init__(
        self,
        checkpoint_dir: str,
        save_freq: int = 1000,
        max_to_keep: int = 3,
        save_best_val: bool = False,
        save_all_states: bool = False,
    ):
        """Callback to save checkpoints during training.

        Args:
            checkpoint_dir (str): Path to the directory where checkpoints will be saved.
            save_freq (int, optional): The frequency at which checkpoints will be saved. Defaults to 1000.
            keep_one_only (bool, optional): Whether to keep only the last checkpoint. Defaults to True.
            save_best_val (bool, optional): Whether to save the best model based on the validation loss. Defaults to False.
            save_all_states (bool, optional): Whether to save all the states of the model. Defaults to False.
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_freq = save_freq
        self.max_to_keep = max_to_keep
        self.keep = []

        self.save_best_val = save_best_val
        if save_best_val:
            logging.warning(
                "When save_best_val is True, please make sure that you pass the validation data to the trainer.fit() method.\n\
                            Otherwise, the best model will not be saved.\n\
                            The model will save the lowest validation value if the metric starts with 'loss' and the highest value otherwise."
            )
            self.best_val = {}

        self.save_all_states = save_all_states
        self.best_path = ""

    def __call__(
        self,
        trainer,  # trainer.TorchTrainer
        global_step: int,
        global_epoch: int,
        logs: Dict,
        isValPhase: bool = False,
        logger: logging = None,
    ):
        """Abstract method to be implemented by the user.

        Args:
            trainer (trainer.Trainer): trainer.TorchTrainer module
            global_step (int): The global step of the training.
            global_epoch (int): The global epoch of the training.
            logs (Dict): The logs of the training which contains the loss and the metrics. For example:
                                                            {
                                                                "loss": 0.1,
                                                                "accuracy": 0.9
                                                                "some_custom_metric": 0.5
                                                            }
            isValPhase (bool, optional): Whether the callback is called during the validation phase. Defaults to False.
            logger (logging, optional): The logger to be used. Defaults to None.
        """
        if not isValPhase:
            if global_step % self.save_freq == 0:
                logger.info("Saving model at step {}".format(global_step))
                if self.save_all_states:
                    ckpt_path = trainer.save_all_states(
                        self.checkpoint_dir, global_epoch, global_step
                    )
                else:
                    ckpt_path = trainer.save_weights(self.checkpoint_dir, global_step)
                self.keep.append(ckpt_path)
                if len(self.keep) > self.max_to_keep:
                    logger.info(f"Deleting checkpoint {self.keep[0]}")
                    ckpt_to_delete = self.keep.pop(0)
                    os.remove(ckpt_to_delete)

        elif isValPhase and self.save_best_val:
            for k, v in logs.items():
                if k not in self.best_val:
                    logger.info(
                        "Model {} improve from inf to {}, Saving model...".format(k, v)
                    )
                    self.best_val[k] = v
                    os.makedirs(
                        os.path.join(self.checkpoint_dir, "best_{}".format(k)),
                        exist_ok=True,
                    )
                    if self.save_all_states:
                        ckpt_path = trainer.save_all_states(
                            os.path.join(self.checkpoint_dir, "best_{}".format(k)), 0, 0
                        )
                    else:
                        ckpt_path = trainer.save_weights(
                            os.path.join(self.checkpoint_dir, "best_{}".format(k)), 0
                        )
                        self.best_path = ckpt_path
                else:
                    if k.startswith("loss"):
                        if v < self.best_val[k]:
                            logger.info(
                                "Model {} improve from {} to {}, Saving model...".format(
                                    k, self.best_val[k], v
                                )
                            )
                            self.best_val[k] = v
                            os.makedirs(
                                os.path.join(self.checkpoint_dir, "best_{}".format(k)),
                                exist_ok=True,
                            )
                            if self.save_all_states:
                                ckpt_path = trainer.save_all_states(
                                    os.path.join(
                                        self.checkpoint_dir, "best_{}".format(k)
                                    ),
                                    0,
                                    0,
                                )
                            else:
                                ckpt_path = trainer.save_weights(
                                    os.path.join(
                                        self.checkpoint_dir, "best_{}".format(k)
                                    ),
                                    0,
                                )
                    else:
                        if v > self.best_val[k]:
                            logger.info(
                                "Model {} improve from {} to {}, Saving model...".format(
                                    k, self.best_val[k], v
                                )
                            )
                            self.best_val[k] = v
                            os.makedirs(
                                os.path.join(self.checkpoint_dir, "best_{}".format(k)),
                                exist_ok=True,
                            )
                            if self.save_all_states:
                                ckpt_path = trainer.save_all_states(
                                    os.path.join(
                                        self.checkpoint_dir, "best_{}".format(k)
                                    ),
                                    0,
                                    0,
                                )
                            else:
                                ckpt_path = trainer.save_weights(
                                    os.path.join(
                                        self.checkpoint_dir, "best_{}".format(k)
                                    ),
                                    0,
                                )
                                self.best_path = ckpt_path
