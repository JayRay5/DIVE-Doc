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

## Abstract

In the DocVQA context, current end-to-end models either use lightweight architectures that run efficiently on small devices but have limited performance or rely on LVLMs that achieve high performance at significant computational cost. Thus, we present **DIVE-Doc**, an end-to-end model that bridges this gap by distilling a 400M-parameter SigLIP visual encoder into a small hierarchical Swin transformer, preserving LVLM performance with only one-fifth of the visual encoder's parameters. We investigate two distillation strategies: Fixed-Resolution Distillation (FRD), which matches teacher–student patch counts by forcing student input resolution, and Adaptive-Resolution Distillation (ARD), which aligns mismatched sequences via parameter-free interpolation, enabling various input resolutions. Fine-tuned with QLoRA, DIVE-Doc attains 82.7% ANLS, outperforming lightweight models and sitting within 2 ANLS of its teacher PaliGEMMA on DocVQA, while halving its visual encoder's latency and supporting higher input resolutions. Analysis on RVL-CDIP and DocLayNet shows that the visual encoder captures document-level structure but delegates fine-grained layout reasoning to the language model decoder.

---

## Key Features

* **Efficient LVLM for DocVQA:** Introduces DIVE-Doc, a novel end-to-end model designed for efficient DocVQA in resource-constrained environments.
* **Knowledge Distillation:** Effectively distills a large 400M-parameter SigLIP visual encoder into a compact hierarchical Swin transformer.
* **Significant Parameter Reduction:** Achieves competitive LVLM performance with only one-fifth (75M) of the teacher visual encoder's parameters.
* **Novel Distillation Strategies:** Explores and evaluates two distinct approaches:
    * **Fixed-Resolution Distillation (FRD):** Matches teacher-student patch counts by enforcing specific student input resolutions.
    * **Adaptive-Resolution Distillation (ARD):** Aligns mismatched sequences using parameter-free interpolation, supporting diverse input resolutions.
* **Competitive Performance:** Achieves 82.7% ANLS on DocVQA, outperforming lightweight models and remaining within 2 ANLS points of its larger teacher (PaliGEMMA).
* **Enhanced Efficiency:** Halves the visual encoder's latency compared to the teacher model.
* **Flexible Input Handling:** Supports higher and various input resolutions, making it more adaptable.
* **Architectural Insights:** Provides analysis on the role of the visual encoder (capturing document-level structure) vs. the language model decoder (handling fine-grained layout reasoning).

---

## Model Overview

DIVE-Doc is an end-to-end model designed for Document Visual Question Answering. It consists of a distilled visual encoder (a small hierarchical Swin Transformer) paired with a language model decoder. The core idea is to leverage knowledge distillation to transfer the strong representational power of a large foundational vision model (SigLIP) to a much smaller, more efficient architecture suitable for document understanding tasks.

---

## Distillation Strategies

We propose and investigate two distinct knowledge distillation strategies tailored for transferring knowledge from large pre-trained visual encoders to smaller, hierarchical student models:

1.  ### Fixed-Resolution Distillation (FRD)
    * **Approach:** This strategy involves training the student model with a fixed input resolution designed to directly match the patch count of the teacher's output sequence. This "forces" the student to learn representations that align directly with the teacher's at a specific resolution.
    * **Characteristics:** Simpler alignment, potentially less flexible in handling varying input sizes post-training.

2.  ### Adaptive-Resolution Distillation (ARD)
    * **Approach:** ARD addresses the challenge of mismatched sequence lengths between the teacher and student models (e.g., when the student processes different input resolutions). It uses parameter-free interpolation to align the feature sequences, allowing the student to be trained and perform effectively across various input resolutions.
    * **Characteristics:** Offers greater flexibility in deployment across different resolution settings, making it highly suitable for diverse real-world scenarios.

---

## Performance & Efficiency

DIVE-Doc demonstrates a compelling trade-off between performance and efficiency on the DocVQA task:

* **ANLS Score:** Achieves **82.7% ANLS** on DocVQA.
* **Teacher Comparison:** Sits within **2 ANLS points** of its teacher, PaliGEMMA (84.77% ANLS).
* **Lightweight Model Comparison:** Significantly **outperforms lightweight models** (e.g., Donut: 66.26% ANLS, Dessurt: 63.22% ANLS).
* **Parameter Reduction:** The distilled visual encoder is **one-fifth the size** of the teacher's (75M parameters vs. 400M).
* **Latency Improvement:** **Halves the visual encoder's latency** compared to the teacher.
* **VRAM Footprint:** Achieves competitive performance while **halving the VRAM footprint**, making it more suitable for resource-constrained environments.

Further analysis on RVL-CDIP and DocLayNet indicates that DIVE-Doc's visual encoder effectively captures document-level structural information, while the fine-grained layout reasoning is delegated to the language model decoder.

---

## Installation

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
