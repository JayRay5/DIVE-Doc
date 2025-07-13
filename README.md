# DIVE-Doc: Downscaling foundational Image Visual Encoder into hierarchical architecture for DocVQA

## Table of Contents
- [Abstract](#abstract)
- [Key Features](#key-features)
- [Model Overview](#model-overview)
- [Distillation Strategies](#distillation-strategies)
- [Performance & Efficiency](#performance--efficiency)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Description

In the DocVQA context, current end-to-end models either use lightweight architectures that run efficiently on small devices but have limited performance or rely on LVLMs that achieve high performance at significant computational cost. Thus, we present **DIVE-Doc**, an end-to-end model that bridges this gap by distilling a 400M-parameter SigLIP visual encoder into a small hierarchical Swin transformer, preserving LVLM performance with only one-fifth of the visual encoder's parameters. We investigate two distillation strategies: Fixed-Resolution Distillation (FRD), which matches teacher–student patch counts by forcing student input resolution, and Adaptive-Resolution Distillation (ARD), which aligns mismatched sequences via parameter-free interpolation, enabling various input resolutions. Fine-tuned with QLoRA, DIVE-Doc attains 82.7% ANLS, outperforming lightweight models and sitting within 2 ANLS of its teacher PaliGEMMA on DocVQA, while halving its visual encoder's latency and supporting higher input resolutions. Analysis on RVL-CDIP and DocLayNet shows that the visual encoder captures document-level structure but delegates fine-grained layout reasoning to the language model decoder.

## Demo & Pretrained Models

| Method                    | VE Latency (ms)| ANLS Score ↑ | Download |
|--------------------------|--------------|----------------|-----|
| **DIVE-Doc (FRD)**       | 446     | **82.67**  |   [🤗 Hugging Face](https://huggingface.co)  |
| **DIVE-Doc (ARD/HRes)**  | 520     | 82.63        |  [🤗 Hugging Face](https://huggingface.co)   |
| **DIVE-Doc (ARD/LRes)**  | **270**    | 79.26       |   [🤗 Hugging Face](https://huggingface.co)  |



## Intallation

To set up the development environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/DIVE-Doc.git](https://github.com/yourusername/DIVE-Doc.git)
    cd DIVE-Doc
    ```
2.  **Create a conda environment (recommended):**
    ```bash
    conda create -n dive-doc python=3.9
    conda activate dive-doc
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file detailing all necessary libraries will be provided in the repository.)*
    
## Repositorie Description
```bash
├── data
│   ├── docvqa
    |   ├── build_image_dataset.py
│   |   ├── build_image_embeddings.py
│   |   ├── utils.py
│   |          .
│   |          .
|   ├── doc-cls
|   |          .
|   |          .
|   ├── dla
|      
|            .
|            .
├── models #contains model files
│   ├── config_divedoc.py
│   ├── lightning_modules.py
│   ├── model.py
|   ├── visual_encoders.py
│             .
│             .
└── training
    ├── docvqa #contrains script for training and evaluation of model
    |    ├── config.py # use to set the VE architecture of the student & hyperparameters for the distillation stage
    |    ...
    ├── doc-cls
    ├── dla

```
[model implementation](src/model/model.py)
## Training & Evaluation

### DocVQA
1. **Dataset**
```bash

```
2. Distillation stage
3. Finetuning stage

### Document Classification

### Document Layout Analysis 

## Citation

If you find DIVE-Doc useful for your research, please consider citing our paper:

```bibtex
@inproceedings{your_paper_id,
  author = {Anonymous ICCV Submission}, % Replace with actual authors after blind review
  title = {DIVE-Doc: Downscaling foundational Image Visual Encoder into hierarchical architecture for DocVQA},
  booktitle = {Proceedings of the ICCV Workshop on Vision and Document Intelligence}, % Update with actual workshop name if different
  year = {2025} % Update with actual publication year
}
