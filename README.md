# Disentangled Global-Local Feature Learning with E-Branchformer for Audio Deepfake Detection

This repository contains the code and pretrained models for the following paper:

* **Title** : Disentangled Global-Local Feature Learning with E-Branchformer for Audio Deepfake Detection
* **Authors** : Phuong Tuan Dat, Ho Bao Thu, Nguyen Tran Trung, Pham Viet Hoang, Nguyen Thi Thu Trang

## Pretrained Model
The pretrained XLSR front-end can be found at [link](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt).

Our pretrained model for ASVspoof 5 is available on Hugging Face: [tuandattttt/Ebranchformer_LA5](https://huggingface.co/tuandattttt/Ebranchformer_LA5).

## Setting up environment
Python version: 3.7.16

Install PyTorch
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other libraries:
```bash
pip install -r requirements.txt
```

Install fairseq:
```bash
git clone https://github.com/facebookresearch/fairseq.git fairseq_dir
cd fairseq_dir
git checkout a54021305d6b3c
pip install --editable ./
```

## Training & Testing on fixed-length input
To train and produce the score for LA set evaluation, run:
```bash
python main.py --algo 5
```

To train and produce the score for DF set evaluation, run:
```bash
python main.py --algo 3
```

To train and produce the score on ASVspoof 5, run:
```bash
python main.py --dataset_train asv5 --algo 5
```
## Scoring
To get evaluation results of minimum t-DCF and EER (Equal Error Rate), follow these steps:
```bash
cd 2021/eval-package
python main.py --cm-score-file your_LA_score.txt --track LA --subset eval # For LA track evaluation
python main.py --cm-score-file your_DF_score.txt --track DF --subset eval # For DF track evaluation
```
## Inference
To run inference on a single wav file with the pretrained model, run:
```bash
python inference.py --ckpt_path=path_to/model.pth --threshold=-3.73 --wav_path=path_to/audio.flac
```
The threshold can be obtained when calculating EER on LA or DF set. In this example, the threshold is from DF set
evaluation.

## Citation
If you find our repository valuable for your work, please consider giving a star to this repo and citing our paper:
```
@inproceedings{dat_ebranchformer,
  title     = {Disentangled Global-Local Feature Learning with E-Branchformer for Audio Deepfake Detection},
  author    = {Phuong Tuan Dat and Ho Bao Thu and Nguyen Tran Trung and Pham Viet Hoang and Nguyen Thi Thu Trang},
}
```

### Acknowledge

Our work is built upon [Temporal-Channel Modeling in Multi-head Self-Attention for Synthetic Speech Detection](https://github.com/ductuantruong/tcm_add) and [conformer-based-classifier-for-anti-spoofing](https://github.com/ErosRos/conformer-based-classifier-for-anti-spoofing). We also follow some parts of the following codebases:

[SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing) (for training pipeline).

[conformer](https://github.com/lucidrains/conformer) (for Conformer model architechture).

[ESPnet](https://github.com/espnet/espnet) (for E-Branchformer model architechture).

[DHVT](https://github.com/ArieSeirack/DHVT) (for Head Token desgin).

Thanks for these authors for sharing their work!
