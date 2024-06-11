
<h1 align="center">
  Emolinse
  <br>
</h1>

<h4 align="center">Official code repository for paper "MemoCMT: Cross-Modal Transformer-Based} Multimodal Emotion Recognition System". Paper submitted to <a href="Scientific Reports"></a> Scientific Reports</h4>

<p align="center">
<a href=""><img src="https://img.shields.io/github/stars/namphuongtran9196/memocmt?" alt="stars"></a>
<a href=""><img src="https://img.shields.io/github/forks/namphuongtran9196/memocmt?" alt="forks"></a>
<a href=""><img src="https://img.shields.io/github/license/namphuongtran9196/memocmt?" alt="license"></a>
</p>

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citation">Citation</a> •
  <a href="#references">References</a> •
</p>

## Abstract
> 

## How To Use
- Clone this repository 
```bash
git clone https://github.com/namphuongtran9196/MemoCMT.git 
cd emolinse
```
- Create a conda environment and install requirements
```bash
conda create -n emolinse python=3.8 -y
conda activate emolinse
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

- Dataset used in this project is [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm) and [ESD](https://hltsingapore.github.io/ESD/). 

```bash
cd scripts && python preprocess.py -ds IEMOCAP --data_root ./data/IEMOCAP_full_release
```

- Before starting training, you need to modify the [config file](./src/configs/base.py) in the config folder. You can refer to the config file in the config folder for more details.

```bash
cd scripts && python train.py -cfg ../src/configs/hubert_base.py
```

- You can also find our pre-trained models in the [release](https://github.com/namphuongtran9196/emolinse/releases).

## Citation
```bibtex

```
---

> GitHub [@namphuongtran9196](https://github.com/namphuongtran9196) &nbsp;&middot;&nbsp;
