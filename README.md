# [AAAI 2025] 3DPGS: 3D Probabilistic Graph Search for Archaeological Piece Grouping

Authors: Junfeng Cheng, Yingkai Yang and Tania Stathaki  
Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/32246/34401

<div align="center">
  <img src="./assets/teaser.png" alt="Teaser" width="60%" />
</div>

## Introduction
This repository contains the code and datasets for the paper “3DPGS: 3D Probabilistic Graph Search for Archaeological Piece Grouping.” The paper introduces a new dataset called ArcPie for the archaeological 3D grouping task. In addition, it presents a new algorithm, 3DPGS, which achieves state-of-the-art performance.
<div align="center">
  <img src="./assets/method.png" alt="Teaser" width="95%" />
</div>

## Requirements
```
conda env create -f environment.yml
conda activate 3dpgs
```

## Datasets
We have released the ArcPie dataset, and you can download it from [here](https://modelscope.cn/datasets/JFCheng/ArcPie). Our dataset contains both training data and mesh data. The mesh data are the original fracture pieces data, which can be applied for rendering. You may also use it for other 3D computer vision or graphics tasks. 

We also include the BBArtifact dataset used in our paper. BBArtifact is created from the original meshes in the artifact category of the [Breaking Bad dataset](https://breaking-bad-dataset.github.io/).

## Usage
### Training
```
bash scripts/train.sh
```

### Evaluation
```
bash scripts/eval.sh
```

### Rendering
We have give an example rendering script, and you can modify the arguments in the script according to your needs:
```
bash scripts/render.sh
```

## ToDo
- [x] Release training code
- [x] Release evaluation code
- [x] Upload pretrained models
- [x] Upload dataset
- [x] Add rendering code

## Citation
If you find this code useful for your research, please consider citing:
```
@inproceedings{cheng20253dpgs,
  title={3DPGS: 3D Probabilistic Graph Search for Archaeological Piece Grouping},
  author={Cheng, Junfeng and Yang, Yingkai and Stathaki, Tania},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={2447--2454},
  year={2025}
}
```

## Acknowledgement
This project is built upon [G-FARS](https://github.com/J-F-Cheng/G-FARS-3DPartGrouping).

Besides, we want to express our gratitude to the following great works:
- [PartNet](https://partnet.cs.stanford.edu/)
- [Generative 3D Part Assembly via Dynamic Graph Learning](https://hyperplane-lab.github.io/Generative-3D-Part-Assembly/)
- [Mitsuba](https://www.mitsuba-renderer.org/)
- [PointFlowRenderer](https://github.com/zekunhao1995/PointFlowRenderer)
- [Breaking Bad](https://arxiv.org/pdf/2210.11463)
