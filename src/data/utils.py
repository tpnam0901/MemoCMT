import torchaudio
import torch


def read_audio(path, new_sample_rate=None):
    waveform, sample_rate = torchaudio.load(path)
    if new_sample_rate is not None:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=new_sample_rate
        )(waveform)
        sample_rate = new_sample_rate

    return waveform, sample_rate


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)
