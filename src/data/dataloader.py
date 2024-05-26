import os
import pickle
import re
from typing import Tuple, Union
import logging
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
)

from models.networks import TestSER
from configs.base import Config
from torchvggish.vggish_input import waveform_to_examples
from tqdm.auto import tqdm
import pickle
import librosa


class BaseDataset(Dataset):
    def __init__(
        self,
        cfg: Config,
        data_mode: str = "train.pkl",
        encoder_model: Union[TestSER, None] = None,
    ):
        """Dataset for IEMOCAP

        Args:
            path (str, optional): Path to data.pkl. Defaults to "path/to/data.pkl".
            encoder_model (_4M_SER, optional): if want to pre-encoder dataset
        """
        super(BaseDataset, self).__init__()

        with open(os.path.join(cfg.data_root, data_mode), "rb") as train_file:
            self.data_list = pickle.load(train_file)

        if cfg.text_encoder_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif cfg.text_encoder_type == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        else:
            raise NotImplementedError(
                "Tokenizer {} is not implemented".format(cfg.text_encoder_type)
            )

        self.audio_max_length = cfg.audio_max_length
        self.text_max_length = cfg.text_max_length
        if cfg.batch_size == 1:
            self.audio_max_length = None
            self.text_max_length = None

        self.audio_encoder_type = cfg.audio_encoder_type

        self.encode_data = False
        self.list_encode_audio_data = []
        self.list_encode_text_data = []
        if encoder_model is not None:
            self._encode_data(encoder_model)
            self.encode_data = True

    def _encode_data(self, encoder):
        logging.info("Encoding data for training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.train()
        encoder.to(device)
        for index in tqdm(range(len(self.data_list))):
            audio_path, text, _ = self.data_list[index]

            samples = self.__paudio__(audio_path)
            audio_embedding = (
                encoder.encode_audio(samples.unsqueeze(0).to(device))
                .squeeze(0)
                .detach()
                .cpu()
            )
            self.list_encode_audio_data.append(audio_embedding)

            input_ids = self.__ptext__(text)
            text_embedding = (
                encoder.encode_text(input_ids.unsqueeze(0).to(device))
                .squeeze(0)
                .detach()
                .cpu()
            )
            self.list_encode_text_data.append(text_embedding)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        audio_path, text, label = self.data_list[index]
        input_audio = (
            self.list_encode_audio_data[index]
            if self.encode_data
            else self.__paudio__(audio_path)
        )
        input_text = (
            self.list_encode_text_data[index]
            if self.encode_data
            else self.__ptext__(text)
        )
        label = self.__plabel__(label)

        return input_text, input_audio, label

    def __paudio__(self, file_path: int) -> torch.Tensor:
        samples, sr = sf.read(file_path, dtype="int16")
        if (
            self.audio_max_length is not None
            and samples.shape[0] < self.audio_max_length
        ):
            samples = np.pad(
                samples, (0, self.audio_max_length - samples.shape[0]), "constant"
            )
        elif self.audio_max_length is not None:
            samples = samples[: self.audio_max_length]

        samples = waveform_to_examples(
            samples, sr, return_tensor=False
        )  # num_samples, 96, 64
        samples = np.expand_dims(samples, axis=1)  # num_samples, 1, 96, 64

        return torch.from_numpy(samples.astype(np.float32))

    def _text_preprocessing(self, text):
        """
        - Remove entity mentions (eg. '@united')
        - Correct errors (eg. '&amp;' to '&')
        @param    text (str): a string to be processed.
        @return   text (Str): the processed string.
        """
        # Remove '@name'
        text = re.sub("[\(\[].*?[\)\]]", "", text)

        # Replace '&amp;' with '&'
        text = re.sub(" +", " ", text).strip()

        # Normalize and clean up text; order matters!
        try:
            text = " ".join(text.split())  # clean up whitespaces
        except:
            text = "NULL"

        # Convert empty string to NULL
        if not text.strip():
            text = "NULL"

        return text

    def __ptext__(self, text: str) -> torch.Tensor:
        text = self._text_preprocessing(text)
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        if self.text_max_length is not None and len(input_ids) < self.text_max_length:
            input_ids = np.pad(
                input_ids,
                (0, self.text_max_length - len(input_ids)),
                "constant",
                constant_values=self.tokenizer.pad_token_id,
            )
        elif self.text_max_length is not None:
            input_ids = input_ids[: self.text_max_length]
        return torch.from_numpy(np.asarray(input_ids))

    def __plabel__(self, label: int) -> torch.Tensor:
        return torch.tensor(label)

    def __len__(self):
        return len(self.data_list)


class FocalNetDataset(BaseDataset):
    def __init__(
        self,
        cfg: Config,
        data_mode: str = "train.pkl",
        encoder_model: Union[TestSER, None] = None,
    ):
        super(FocalNetDataset, self).__init__(cfg, data_mode, encoder_model)
        self.audio_resize = torch.nn.Upsample(
            (cfg.audio_im_size, cfg.audio_im_size), mode="nearest"
        )

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        audio_path, text, label = self.data_list[index]
        # num_samples, 1, 96, 64
        log_mel = (
            self.list_encode_audio_data[index]
            if self.encode_data
            else self.__paudio__(audio_path)
        )
        log_mel_delta1 = np.zeros((0, *log_mel.shape[-3:]))
        for i in range(log_mel.size(0)):
            # 96,64
            sample = log_mel[i, 0]
            delta1 = librosa.feature.delta(sample.numpy())
            log_mel_delta1 = np.concatenate(
                [log_mel_delta1, delta1[None, None, ...]], axis=0
            )
        log_mel_delta2 = np.zeros((0, *log_mel.shape[-3:]))
        for i in range(log_mel_delta1.shape[0]):
            # 96,64
            sample = log_mel_delta1[i, 0]
            delta2 = librosa.feature.delta(sample, order=2)
            log_mel_delta2 = np.concatenate(
                [log_mel_delta2, delta2[None, None, ...]], axis=0
            )
        input_audio = torch.from_numpy(
            np.concatenate([log_mel.numpy(), log_mel_delta1, log_mel_delta2], axis=1)
        ).float()
        # input_audio = self.audio_resize(input_audio)

        input_text = (
            self.list_encode_text_data[index]
            if self.encode_data
            else self.__ptext__(text)
        )
        label = self.__plabel__(label)

        return input_text, input_audio, label


def build_train_test_dataset(cfg: Config, encoder_model: Union[TestSER, None] = None):
    DATASET_MAP = {
        "IEMOCAP": FocalNetDataset,
        "ESD": FocalNetDataset,
    }

    dataset = DATASET_MAP.get(cfg.data_name, None)
    if dataset is None:
        raise NotImplementedError(
            "Dataset {} is not implemented, list of available datasets: {}".format(
                cfg.data_name, DATASET_MAP.keys()
            )
        )
    if cfg.data_name in ["IEMOCAP_MSER", "MELD_MSER"]:
        return dataset(cfg)

    train_data = dataset(
        cfg,
        data_mode="train.pkl",
        encoder_model=encoder_model,
    )

    if encoder_model is not None:
        encoder_model.eval()
    test_set = cfg.data_valid if cfg.data_valid is not None else "test.pkl"
    test_data = dataset(
        cfg,
        data_mode=test_set,
        encoder_model=encoder_model,
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return (train_dataloader, test_dataloader)
