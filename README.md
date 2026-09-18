<h1 align="center">[A<sup>2</sup>L] Single-Round Active Manifold Calibration for Source-Free Domain Adaptation in Segmentation</h1>

## 📌 Abstract

Severe domain shifts and class imbalance make source-free domain adaptation (SFDA) for medical image segmentation prone to confirmation bias and feature entanglement. We propose Adapt-Label-Adapt (A<sup>2</sup>L), a single-round active SFDA framework. Unlike direct querying, A<sup>2</sup>L first improves class separability using unlabeled target data, reducing the impact of feature entanglement on sample assessment. We then assess segmentation difficulty across foreground classes and inter-image similarity to select a few representative images for one-time annotation. After annotation, we construct class-wise anchors from labeled pixel features and continue adapting on the remaining unlabeled data. Expert supervision thus not only corrects errors on hard samples but also guides target representation calibration and mitigates confirmation bias. Extensive experiments show that A<sup>2</sup>L achieves state-of-the-art (SOTA) performance with a single annotation round under an extremely low annotation budget.

## 🎯 Motivation

<p align="center">
  <img width="900" alt="Motivation for A2L" src="./Figures/Introduction.png">
</p>

## 🎇 Method Overview

<p align="center">
  <img width="1200" alt="A2L framework" src="./Figures/Method.png">
</p>
## 💡 Key Features

- We propose A<sup>2</sup>L, a single-round active SFDA framework that uses UPA to mitigate sampling bias under domain shifts and identifies hard samples for active querying.
- We design PAUH, which combines class-frequency weighting with inter-image similarity to overcome sampling inaccuracies caused by domain shifts and class imbalance.
- We introduce LGMC, which constructs class-wise anchors from limited annotations and continues adapting with unlabeled data to calibrate target representations and mitigate confirmation bias.

## 🚀 Installation & Usage

### 1. Environment

```bash
git clone https://github.com/JiaqiZhang-Sengoku/A2L.git
cd A2L

conda create -n a2l python=3.10 -y
conda activate a2l
```

```bash
pip install numpy pillow opencv-python medpy scipy
```

### 2. Data Preparation

Arrange the target-domain fundus images and masks as follows:

```text
Data/
├── Domain1/
│   ├── train/ROIs/
│   │   ├── image/
│   │   └── mask/
│   └── test/ROIs/
│       ├── image/
│       └── mask/
├── Domain2/
│   ├── train/ROIs/
│   │   ├── image/
│   │   └── mask/
│   └── test/ROIs/
│       ├── image/
│       └── mask/
└── Domain4/
    └── test/ROIs/
        ├── image/
        └── mask/
```

### 3. Source Checkpoint

A<sup>2</sup>L starts from a source-pretrained DeepLabV3 model with a MobileNet backbone. Place the source checkpoint at:

```text
Checkpoints/
└── Source/
    └── source_model.pth.tar
```

### 4. Development Runs

The Domain1 and Domain2 entry points use the fixed split in `Config/validation_split.json`. They hold out samples from the training set for validation and do not open test images or labels.

```bash
python A2L_Domain1.py
python A2L_Domain2.py
```

Inspect the effective configuration without loading the training dependencies or data:

```bash
python A2L_Domain1.py --print-config
python A2L_Domain2.py --print-config
python A2L_Domain4.py --print-config
```

### 5. Compound and Open-Domain Evaluation

```bash
python A2L_Domain4.py
```

Run all retained full-evaluation configurations, including Domain1, Domain2, and Domain4:

```bash
bash A2L_Test.sh
```

## 📝 References

If you find the code useful for your research, please consider citing:


## 📢 LICENSE

The project is under [MIT License](./LICENSE), and is for research purpose ONLY.
