# DIVE-Doc: Downscaling foundational Image Visual Encoder into hierarchical architecture for DocVQA 
<div align="center">Official implementation of DIVE-Doc|Paper|Poster|Slide </div>

## Description

In the DocVQA context, current end-to-end models either use lightweight architectures that run efficiently on small devices but have limited performance or rely on LVLMs that achieve high performance at significant computational cost. Thus, we present **DIVE-Doc**, an end-to-end model that bridges this gap by distilling a 400M-parameter SigLIP visual encoder into a small hierarchical Swin transformer, preserving LVLM performance with only one-fifth of the visual encoder's parameters. We investigate two distillation strategies: Fixed-Resolution Distillation (FRD), which matches teacher–student patch counts by forcing student input resolution, and Adaptive-Resolution Distillation (ARD), which aligns mismatched sequences via parameter-free interpolation, enabling various input resolutions. Fine-tuned with QLoRA, DIVE-Doc attains 82.7% ANLS, outperforming lightweight models and sitting within 2 ANLS of its teacher PaliGEMMA on DocVQA, while halving its visual encoder's latency and supporting higher input resolutions. Analysis on RVL-CDIP and DocLayNet shows that the visual encoder captures document-level structure but delegates fine-grained layout reasoning to the language model decoder.

## Demo & Trained Models

| Model                    | VE Latency (ms)| ANLS Score ↑ | Download |
|--------------------------|--------------|----------------|-----|
| **DIVE-Doc (FRD)**       | 446     | **82.67**  |   [🤗 Hugging Face](https://huggingface.co)  |
| **DIVE-Doc (ARD/HRes)**  | 520     | 82.63        |  [🤗 Hugging Face](https://huggingface.co)   |
| **DIVE-Doc (ARD/LRes)**  | **270**    | 79.26       |   [🤗 Hugging Face](https://huggingface.co)  |



## Installation

To set up the development environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/JayRay5/DIVE-Doc](https://github.com/JayRay5/DIVE-Doc)
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
    
## Repository Struture
```bash
DIVE-Doc
├── data
│   ├── docvqa
|   |   ├── build_image_dataset.py #generate image from the docvqa dataset without dupplicated samples for the distillation stage.
│   |   ├── build_image_embeddings.py #generate embeddings of the teacher paligemma for the distillation stage.
│   |   └── utils.py       
│   |          
|   ├── doc-cls
|   |        .
|   |        .
|   ├── dla
|      
|            .
|            .
|
|
├── experiments #contains saved models and results of the runned experiments
|            .
|            .
|
├── models 
│   ├── config_divedoc.py #contains config classes for huggingface models.
│   ├── lightning_modules.py #contains lightning torch classes for the distillation stage.
│   ├── model.py #contains huggingface models.
|   ├── visual_encoders.py #contains torch visual encoder models.
│             .
│             .
└── training
|   ├── docvqa #contrains script for training and evaluation of model.
|   |    ├── config.py #use to set the VE architecture of the student & hyperparameters for the distillation stage.
|   |    ├── distillation_stage1.py #pipeline training for the distillation stage.
|   |    ├── evaluation.py #generate answer for the docvqa test set.
|   |    └── finetuning_stage2.py #pipeline training for the end-to-end finetuning stage.
|   |
|   ├── doc-cls
|   |
|   |
|   |
|   ├── dla
|
├── app.py #pipeline to use the model in inference with a web gradio interface
└── token.json #add inside a hugging face token to be able to access to the teacher model from huggingface
```
## Training & Evaluation

### DocVQA
1. **Dataset**
```bash
cd dataset/docvqa
python build_image_dataset.py #generate image png without duplicated samples
python build_image_embeddings.py #generate Paligemma image embeddings 
```
2. **Distillation stage** <br>
You can set the student configuration you want or a new one in
```bash
./trainning/docvqa/config.py
```
Then, start the distillation script: 
```bash
cd training/docvqa
python distillation_stage1.py #the script will create a new folder in ./experiments, which will contain the weights of this training stage
```
3. **Finetuning stage**
Once you have a distilled model, you can finetune with QLORA using the following script:
```bash
python finetuning_stage2.py #You have to put the path of the folder created by the distillation pipeline in this script
```
4. **Evaluation** <br>
You can evaluate the distillation stage model or the final model with the following script by inserting the model path in the experiment folder.
```bash
python evaluation.py
```
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
