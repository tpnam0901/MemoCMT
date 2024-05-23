import os
import torch
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from .utils import read_audio, pad_sequence
from glob import glob
import random


class BaseDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        sample_rate=None,
        max_audio_sec=5,
    ):
        super(BaseDataset, self).__init__()
        self.X = X
        self.y = y
        self.sample_rate = sample_rate
        self.max_audio_sec = max_audio_sec

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        waveform, sample_rate = read_audio(self.X[index], self.sample_rate)
        max_audio_len = sample_rate * self.max_audio_sec
        # random pick start point
        len_waveform = max(waveform.shape)
        if len_waveform > max_audio_len:
            start = random.choice(range(len_waveform - max_audio_len))
            waveform = waveform[:, start : start + max_audio_len]
        label = torch.tensor(self.y[index])
        return waveform, label

    def __len__(self):
        return len(self.X)


def collate_fn_base(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, label in batch:
        tensors += [waveform]
        targets += [label]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)[:, 0, :]
    targets = torch.stack(targets)

    return tensors, targets


def build_random_train_val_dataset(
    data_root: str = "reid_skip_test",
    data_type: str = "waveform",
    batch_size: int = 32,
    max_audio_sec: int = 5,
    seed=0,
):
    sample_paths = glob(os.path.join(data_root, "*", "*"))
    labels = [CLASSES[path.split("/")[-2]] for path in sample_paths]

    dataset = {
        "waveform": BaseDataset,
    }
    dataset_fn = dataset[data_type]

    # using the train test split function
    X_train, X_test, y_train, y_test = train_test_split(
        sample_paths, labels, random_state=seed, test_size=0.2, shuffle=True
    )

    training_data = dataset_fn(X_train, y_train, max_audio_sec=max_audio_sec)
    test_data = dataset_fn(X_test, y_test, max_audio_sec=max_audio_sec)

    def worker_init_fn(worker_id):
        os.sched_setaffinity(0, list(range(os.cpu_count())))

    collate_fn = collate_fn_base if data_type == "waveform" else None

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    return (train_dataloader, test_dataloader), len(set(labels))
