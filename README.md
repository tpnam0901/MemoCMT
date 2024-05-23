
<h1 align="center">
  4M-SER
  <br>
</h1>

<h4 align="center">Official code repository for paper "Comprehensive Study of Multi-Feature Embeddings and Multi-Loss Functions with Multi-Head Self-Attention Fusion for Multi-Modal Speech Emotion Recognition". Paper submitted to <a href="https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?reload=true&punumber=5165369">IEEE Transactions on Affective Computing (2024)</a> </h4>

<p align="center">
<a href=""><img src="https://img.shields.io/github/stars/namphuongtran9196/4m-ser?" alt="stars"></a>
<a href=""><img src="https://img.shields.io/github/forks/namphuongtran9196/4m-ser?" alt="forks"></a>
<a href=""><img src="https://img.shields.io/github/license/namphuongtran9196/4m-ser?" alt="license"></a>
</p>

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Download</a> •
  <a href="#license">License</a> •
  <a href="#citation">Citation</a> •
  <a href="#references">References</a> •
</p>

## Abstract
> In recent years, multi-modal analysis has markedly improved the performance of speech emotion recognition (SER), advancing the realms of affective computing and human-computer interaction. However, current approaches often simply concatenate features extracted from audio/speech and text inputs, neglecting effective fusion and alignment. Furthermore, most studies primarily rely on the commonly used cross-entropy loss function for classification, overlooking the potential benefits of integrating feature loss functions to enhance SER performance. In this study, we propose a two-stage framework to bolster the effectiveness and robustness of multi-modal SER. Leveraging multi-feature embeddings and multi-loss functions with multi-head self-attention fusion, our approach applies transfer learning in the initial stage to train feature fusion, while fine-tuning in the subsequent stage facilitates feature alignment. Experimental results and performance analyses on the interactive emotional dyadic motion capture (IEMOCAP) and emotional speech database (ESD) datasets reveal that employing diverse feature embeddings can yield varying performance levels, and integrating different feature loss functions can significantly improve model performance. Additionally, we conduct correlation analysis between audio/speech and text features, alongside model interpretation, to gain insights into the model's behavior.

## How To Use
- Clone this repository 
```bash
git clone https://github.com/namphuongtran9196/4m-ser.git 
cd 4m-ser
```
- Create a conda environment and install requirements
```bash
conda create -n 4m-ser python=3.8 -y
conda activate 4m-ser
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

- Dataset used in this project is IEMOCAP. You can download it [here](https://sail.usc.edu/iemocap/iemocap_release.htm). 
- Preprocess data or you can download our preprocessed dataset [here](https://github.com/namphuongtran9196/4m-ser/releases) (this only include path to sample in dataset).

```bash
cd scripts && python preprocess.py -ds IEMOCAP --data_root ./data/IEMOCAP_full_release
```

- Before starting training, you need to modify the [config file](./src/configs/base.py) in the config folder. You can refer to the config file in the config folder for more details.

```bash
cd scripts && python train.py -cfg ../src/configs/4m-ser_bert_vggish.py
```

- The visualization of our figure in paper can be found in [notebook](./src/visualization/metrics.ipynb).

- You can also find our pre-trained models in the [release](https://github.com/namphuongtran9196/4m-ser/releases).

## Citation
```bibtex

```
## References

[1] Phuong-Nam Tran, 3M-SER: Multi-modal Speech Emotion Recognition using Multi-head Attention Fusion of Multi-feature Embeddings (INISCOM), 2023. Available https://github.com/namphuongtran9196/3m-ser.git

[2] Nhat Truong Pham, SERVER: Multi-modal Speech Emotion Recognition using Transformer-based and Vision-based Embeddings (ICIIT), 2023. Available https://github.com/nhattruongpham/mmser.git

---

> GitHub [@namphuongtran9196](https://github.com/namphuongtran9196) &nbsp;&middot;&nbsp;