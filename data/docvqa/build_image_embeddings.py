#file and os lib managements
import os
import json

#To prevent the warning about deadlock when initializing the data loader
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

#Change the path of some HF environment variables to store the download data (model and dataset) from the hub to a choosen location
#PUT the datasets in data and transformers in models, other in main branch
os.environ['HF_HOME'] = "../.cache"
os.environ['HF_HUB_CACHE'] = "../.cache"
os.environ['TRANSFORMERS_CACHE'] = "../.cache"
os.environ['HF_DATASETS_CACHE'] = "../.cache"

#ML libraries
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

#utils
from utils import GenerationEmbeddingsPipeline, generate_and_save_embeddings


dataset_path = "./docvqa/images"
saved_path = "./docvqa"
local_path = True
dataset_name = "docvqa"
splits = ['train','validation','test']

with open("../token.json", "r") as f:
    hf_token = json.load(f)["HF_token"] 

paligemma_processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-docvqa-896",token = hf_token)
paligemma_model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-ft-docvqa-896",token = hf_token)

pipeline = GenerationEmbeddingsPipeline(processor = paligemma_processor, model = paligemma_model.vision_tower, device = "cuda:0")

generate_and_save_embeddings(pipeline=pipeline,dataset_name=dataset_name,split_names=splits,dataset_path=dataset_path,saved_path=saved_path,load_from_disk_bool=local_path)