# DIVE-Doc: Downscaling foundational Image Visual Encoder into hierarchical architecture for DocVQA 
<div align="center">Official implementation of DIVE-Doc </div>

## Description

**DIVE-Doc** is an end-to-end trade-off between LVLMs and lightweight architectures in the context of DocVQA. It is built by distilling the SigLIP-400m visual encoder of PaliGEMMA into a small hierarchical Swin transformer, while reusing the original GEMMA decoder. This allowed DIVE-Doc to keep competitive performance with its teacher while reducing the visual encoder's parameters to one-fifth.

## Demo & Trained Models

![Alt text for video GIF](./demo_readme.gif)

You can use the model on a gradio web interface by running:
```bash
python app.py
```
The trained models presented in the paper can be downloaded on HuggingFace.

| Model                    | VE Latency (ms)| ANLS Score ↑ | Download |
|--------------------------|--------------|----------------|-----|
| **DIVE-Doc (FRD)**       | 446     | **82.67**  |   [🤗 Hugging Face](https://huggingface.co/JayRay5/DIVE-Doc-FRD)  |
| **DIVE-Doc (ARD/HRes)**  | 520     | 82.63        |  [🤗 Hugging Face](https://huggingface.co/JayRay5/DIVE-Doc-ARD-HRes)   |
| **DIVE-Doc (ARD/LRes)**  | **270**    | 79.26       |   [🤗 Hugging Face](https://huggingface.co/JayRay5/DIVE-Doc-ARD-LRes)  |



## Installation

To set up the development environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JayRay5/DIVE-Doc
    cd DIVE-Doc
    ```
2.  **Create a conda environment:**
    ```bash
    conda create -n dive-doc-env python=3.11.5
    conda activate dive-doc-env
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
## Repository Structure
```bash
DIVE-Doc
├── data
|   ├── dla
|   |    └── utils.py
|   |
|   ├── doc-cls
|   |    └── utils.py
|   |
│   └── docvqa
|       ├── build_image_dataset.py #generate images from the docvqa dataset without duplicated samples for the distillation stage.
│       ├── build_image_embeddings.py #generate embeddings of the teacher PaliGEMMA for the distillation stage.
│       └── utils.py             
|
|
├── experiments #contains saved models and results of the experiments.
|            .
|            .
|
├── models 
│   ├── config_divedoc.py #contains config classes for HuggingFace docvqa models.
|   ├── dla_config.py #contains config classes for Huggingface DLA models.
|   ├── dla_model.py #contains Huggingface DLA models.
│   ├── lightning_modules.py #contains lightning torch classes for the distillation stage.
│   ├── model.py #contains Huggingface DocVQA models.
|   └── visual_encoders.py #contains torch visual encoder models.
│   
│     
└── training
|   ├── dla # contains script for training and evaluation of models.
|   |    ├── config.py #Use to set the architecture and choose the VE.
|   |    ├── test.py #Generate segmentation performance on the test set for the chosen model.
|   |    ├── train.py #Training pipeline for a DIVE-Doc model trained until finetuning_stage2 or for Donut & PaliGEMMA VE.
|   |    └── utils.py 
|   |
|   ├── doc-cls
|   |
|   └── docvqa #contains script for training and evaluation of models.
|        ├── config.py #use to set the VE architecture of the student & hyperparameters for the distillation stage.
|        ├── distillation_stage1.py #pipeline training for the distillation stage.
|        ├── evaluation.py #generate answer for the DocVQA test set.
|        └── finetuning_stage2.py #pipeline training for the end-to-end finetuning stage.
|        
|
├── app.py #pipeline to use the model in inference with a web gradio interface.
└── token.json #add inside a HuggingFace token to be able to access to the teacher model from HuggingFace.
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
./training/docvqa/config.py
```
Then, start the distillation script: 
```bash
cd training/docvqa
python distillation_stage1.py 
```
This script will create a "model_{m}" folder inside the "./experiments" folder with m=0 if this is the first model training.
Then, it will also create a ".experiments/model_{m}/distillation_stage1" folder where weights and config files will be saved.

3. **Finetuning stage**<br>
Once you have a distilled model, you can finetune with QLORA using the following script:
```bash
python finetuning_stage2.py #You have to put the path of the "./experiments/model_m" folder created by the distillation pipeline in this script
```
The weights will be saved in a "finetuning_stage2" folder inside "./experiments/model_{m}/".

4. **Evaluation** <br>
You can evaluate the distillation stage model or the final model with the following script by inserting the model path in the experiment folder.
```bash
python evaluation.py
```
This script generates a results.json file containing the predicted answers for the test set, which is saved in the corresponding model folder in "./experiment". <br>
To assess the performance, please upload the mentioned file on the [Robust Reading Competition website](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=1).
### Document Classification
Once you have finetuned your DIVE-Doc model until stage 2, you can evaluate the visual encoder capacity on the Document Classification (Doc-CLS) task.<br>
For that, you have to put the model directory path in the following script
```bash
./training/doc_cls/config.py
```
Then start the training of the segmentation decoder head:
```bash
cd training/doc_cls
python train.py
```
This will generate a new folder "cls" as "./experiments/model_{m}/cls" for Doc-CLS experiments. <br> 
Then you can evaluate the model on the test with the following script
```bash
python test.py
```
It will save the score as a JSON file in the "./experiments/model_{m}/cls/" folder.
### Document Layout Analysis 
Once you have finetuned your DIVE-Doc model until stage 2, you can evaluate the visual encoder capacity on the Document Layout Analysis (DLA) task.<br>
For that, you have to put the model directory path in the following script
```bash
./training/dla/config.py
```
Then start the training of the segmentation decoder head:
```bash
cd training/dla
python train.py
```
This will generate a new folder "dla" as "./experiments/model_{m}/dla" for DLA experiments. <br> 
Then you can evaluate the model on the test with the following script
```bash
python test.py
```
It will save the score as a JSON file in the "./experiments/model_{m}/dla/" folder.
## Citation
If you find DIVE-Doc useful for your research, please cite:

```bibtex
#TO COMPLET
@inproceedings{
}
