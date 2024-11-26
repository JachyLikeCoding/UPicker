<div align="center">

  <h1><img src="resources/icon.png" width="100"> UPicker: a Semi-Supervised Particle Picking
Transformer Method for Cryo-EM Micrographs</h1>

![GitHub top language](https://img.shields.io/github/languages/top/JachyLikeCoding/UPicker)![GitHub last commit](https://img.shields.io/github/last-commit/JachyLikeCoding/UPicker) ![Static Badge](https://img.shields.io/badge/Platform-Linux-green)


<div align="justify"> 
Automatic single particle picking is a critical step in the data processing pipeline of cryo-electron microscopy (cryo-EM)
structure reconstruction. Here, we propose UPicker, a semi-supervised transformer-based particle-picking method with a two-stage training process: unsupervised pretraining and supervised fine-tuning. During the
unsupervised pretraining, an Adaptive-LoG region proposal generator is proposed to obtain pseudo-labels from unlabeled
data for initial feature learning. For the supervised fine-tuning, UPicker only needs a small amount of labeled data
to achieve high accuracy in particle picking. To further enhance model performance, UPicker employs a contrastive
denoising training strategy to reduce redundant detections and accelerate convergence, along with a hybrid data
augmentation strategy to deal with limited labeled data.

## Install
<table>
<tr>
<th>
1. Download
</th>
<td colspan="2">


```bash
git clone https://github.com/JachyLikeCoding/UPicker.git
```
</td>
</tr>

<tr></tr>

<tr>
<th>
2. Install
</th>
<td>
Create conda environment with <b>conda</b>:

```bash
conda env create \
-f freeze.yml

conda activate upicker
```
</td>

<td>
Alternatively, create the environment with <b>mamba</b>:

```bash
mamba env create \
--file freeze.yml \
--channel-priority flexible -y

mamba activate upicker
```
</td>
</tr>
</table>


## Overview
Figure below demonstrates the particle picking workflow of **UPicker**.

![Alt text](<resources/pipeline.png>)

## Dataset Preparation
You can download datasets from EMPIAR, CryoPPP or use your own dataset and organize them as following: 
```
UPicker_Project/
└── data/
    └── DATASET1/
        ├── micrographs/
            ├── 0001.mrc
            ├── 0002.mrc
            ├── (...).mrc
        	└── xxxx.mrc
        └── annots/
        	├── 0001.star
            ├── 0002.star
            ├── (...).star
        	└── xxxx.star
```

The micrograph name should be consistent with the coordinate file name. We support star, box, txt, csv file formats to read and write coordinate files, details can be seen in the `cryoEM/coord_io.py` file.


Then, you need to preprocess the micrographs and get region proposals by running:


```
python cryoEM/preprocess.py --box_width BOXSIZE --images data/YOUR_DATASET/micrographs/ --output_dir data/YOUR_DATASET
```
```
Optional Arguments:
  --images (str): The folder of micrographs to be preprocessed.
  --bin (int, default: 1): Downsample bin.
  --output_dir (str, default: "output"): Output directory.
  --box_width (int, default:200): The box width. Usually choose 1.5 * particle diameter.
  --device (str, default: "cuda:0" if available, else "cpu"): Device for training (cuda:0 or cpu).
  --mode (str, default: "train", choices=['train','test']): If mode is test, no autopick schedule.
  --noequal (store_true): If need histogram equalization.
  --ifready (store_true): If the micrographs have been preprocessed.
  --denoise (str, default: "gaussian"): The denoise filter.
```



## Make coco-style dataset for training and evaluation.
```
python cryoEM/make_coco_dataset.py --coco_path data/YOUR_DATASET --box_width BOXSIZE --phase pretrain --images_path data/YOUR_DATASET/micrographs/processed/

python cryoEM/make_coco_dataset.py --coco_path data/YOUR_DATASET --box_width BOXSIZE --phase train --images_path data/YOUR_DATASET/micrographs/processed/

python cryoEM/make_coco_dataset.py --coco_path data/YOUR_DATASET --box_width BOXSIZE --phase val --images_path data/YOUR_DATASET/micrographs/processed/

```

### (OPTIONAL) Clean region proposals for pre-training 
Need install the `micrograph_cleaner_em` package first.

```
python cryoEM/box_clean.py \
    --image_path data/YOUR_DATASET/micrographs \
    --boxsize BOXSIZE
```

### 📈 Pretrain with A-LoG region proposals

```
python -u main.py \
    --config_file config/UPICKER/UPICKER_4scale_50epoch.py \
    --output_dir exps/Upicker_exps/YOUR_DATASET/pretrain_YOUR_DATASET \
    --dataset YOUR_DATASET_pretrain \
    --dataset_file YOUR_DATASET \
    --strategy log \
    --box_width BOXSIZE \
    --lr_backbone 0
```

### 🗃️ Fine-tune with pretrained model
```
%%bash

python -u main.py \
    --config_file config/DINO/DINO_4scale_50epoch.py \
    --output_dir exps/Upicker_exps/YOUR_DATASET/finetune_YOUR_DATASET \
    --dataset_file YOUR_DATASET \
    --pretrain exps/Upicker_exps/YOUR_DATASET/pretrain_YOUR_DATASET/checkpoint.pth \
    --box_width BOXSIZE
```

### 🖥️ Inference
```
python -u inference.py \
    --dataset_file YOUR_DATASET \
    --output_dir outputs/finetune_YOUR_DATASET/ \
    --resume exps/Upicker_exps/YOUR_DATASET/finetune_YOUR_DATASET/checkpoint_best_regular.pth \
    -sth 0.25
```



## 🔭 Future Plans

- Tutorial for some datasets.

## Citation

If you find this viewer useful, please consider citing our work:

> (Wait for online)