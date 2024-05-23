import logging
import os
from typing import Dict

import torch
from torch import Tensor
from configs.base import Config
from models.networks import TestSER
from utils.torch.trainer import TorchTrainer


class Trainer(TorchTrainer):
    def __init__(
        self,
        cfg: Config,
        network: TestSER,
        criterion: torch.nn.CrossEntropyLoss = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.network = network
        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.train()
        self.optimizer.zero_grad()

        # Prepare batch
        input_text, input_audio, label = batch

        # Move inputs to cpu or gpu
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)
        input_text = input_text.to(self.device)

        # Forward pass
        output = self.network(input_text, input_audio)
        loss = self.criterion(output, label)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Calculate accuracy
        _, preds = torch.max(output[0], 1)
        accuracy = torch.mean((preds == label).float())
        return {
            "loss": loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        self.network.eval()
        # Prepare batch
        input_text, input_audio, label = batch

        # Move inputs to cpu or gpu
        input_audio = input_audio.to(self.device)
        label = label.to(self.device)
        input_text = input_text.to(self.device)
        with torch.no_grad():
            # Forward pass
            output = self.network(input_text, input_audio)
            loss = self.criterion(output, label)
            # Calculate accuracy
            _, preds = torch.max(output[0], 1)
            accuracy = torch.mean((preds == label).float())
        return {
            "loss": loss.detach().cpu().item(),
            "acc": accuracy.detach().cpu().item(),
        }
