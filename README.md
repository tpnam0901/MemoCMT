
<h1 align="center">
  Emolinse
  <br>
</h1>

<h4 align="center">Official code repository for paper "". Paper submitted to <a href=""></a> </h4>

<p align="center">
<a href=""><img src="https://img.shields.io/github/stars/namphuongtran9196/emolinse?" alt="stars"></a>
<a href=""><img src="https://img.shields.io/github/forks/namphuongtran9196/emolinse?" alt="forks"></a>
<a href=""><img src="https://img.shields.io/github/license/namphuongtran9196/emolinse?" alt="license"></a>
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
git clone https://github.com/namphuongtran9196/emolinse.git 
cd emolinse
```
- Create a conda environment and install requirements
```bash
conda create -n emolinse python=3.8 -y
conda activate emolinse
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

- Dataset used in this project is IEMOCAP. You can download it [here](https://sail.usc.edu/iemocap/iemocap_release.htm). 
- Preprocess data or you can download our preprocessed dataset [here](https://github.com/namphuongtran9196/emolinse/releases) (this only include path to sample in dataset).

```bash
cd scripts && python preprocess.py -ds IEMOCAP --data_root ./data/IEMOCAP_full_release
```

- Before starting training, you need to modify the [config file](./src/configs/base.py) in the config folder. You can refer to the config file in the config folder for more details.

```bash
cd scripts && python train.py -cfg ../src/configs/focal_net_t.py
```

- You can also find our pre-trained models in the [release](https://github.com/namphuongtran9196/emolinse/releases).

## Citation
```bibtex

```
## References

[1] Phuong-Nam Tran, 3M-SER: Multi-modal Speech Emotion Recognition using Multi-head Attention Fusion of Multi-feature Embeddings (INISCOM), 2023. Available https://github.com/namphuongtran9196/3m-ser.git

[2] Nhat Truong Pham, SERVER: Multi-modal Speech Emotion Recognition using Transformer-based and Vision-based Embeddings (ICIIT), 2023. Available https://github.com/nhattruongpham/mmser.git

---

> GitHub [@namphuongtran9196](https://github.com/namphuongtran9196) &nbsp;&middot;&nbsp;
