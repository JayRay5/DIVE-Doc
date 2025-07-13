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

## Demo


## Performance & Efficiency
| Method                      || #Params (VE) | #Params Total | OCR        | General  (ANLS) ↑   | Figure     | Free-text  | Picture    | Layout     |
|-----------------------------|--------------|----------------|------------|-------------|------------|------------|------------|------------|                               |
| **PaliGEMMA**               | 0.4(B)       | 3(B)           |            | 84.77       | 65.43      | 80.99      | 73.82      | 87.33      |
| **UDOP**                    | -            | 0.8(B)         | ✓          | 84.70       | -          | -          | -          | -          |
| **LayoutLMv3**              | -            | 0.133(B)       | ✓          | 78.76       | -          | -          | -          | -          |
| **Donut**                   | 0.075(B)     | 0.2(B)         |            | 66.26       | 39.60      | 46.43      | 29.69      | 69.87      |
| **Dessurt**                 |              | 0.127(B)       |            | 63.22       | 31.64      | 48.52      | 28.62      | 64.86      |
| **DIVE-Doc (FRD)**          | 0.075(B)     | 2.58(B)        |            | **82.67**   | 59.33      | **78.83**  | 49.96      | 85.00      |
| **DIVE-Doc (ARD/HRes)**     | 0.075(B)     | 2.58(B)        |            | 82.63       | **61.48**  | 77.64      | **58.68**  | **85.34**  |
| **DIVE-Doc (ARD/LRes)**     | 0.075(B)     | 2.58(B)        |            | 79.26       | 54.94      | 74.54      | 58.28      | 83.15      |


## Get Started

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

---

## Usage

This section will guide you on how to use DIVE-Doc for inference and how to replicate our training and evaluation procedures.

1.  **Download Pre-trained Models:**
    *(Instructions on where to download model weights, likely from Hugging Face Hub, will be provided here.)*
2.  **Inference:**
    *(Example Python scripts or command-line commands for running inference on your own document images and questions will be provided here.)*
3.  **Training & Evaluation:**
    *(Detailed instructions on how to set up the training environment, prepare datasets, and run training/evaluation scripts will be provided here.)*

---

## Citation

If you find DIVE-Doc useful for your research, please consider citing our paper:

```bibtex
@inproceedings{your_paper_id,
  author = {Anonymous ICCV Submission}, % Replace with actual authors after blind review
  title = {DIVE-Doc: Downscaling foundational Image Visual Encoder into hierarchical architecture for DocVQA},
  booktitle = {Proceedings of the ICCV Workshop on Vision and Document Intelligence}, % Update with actual workshop name if different
  year = {2025} % Update with actual publication year
}
