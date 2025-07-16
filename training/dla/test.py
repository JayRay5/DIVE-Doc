import json
import os
import sys
from pathlib import Path
parent_root = Path().resolve().parent.parent.parent 

sys.path.append(str(parent_root))

os.environ['HF_HOME'] = "../../.cache"
os.environ['HF_HUB_CACHE'] = "../../.cache"
os.environ['TRANSFORMERS_CACHE'] = "../../.cache"
os.environ['HF_DATASETS_CACHE'] = "../../.cache"

from transformers import DonutProcessor,AutoProcessor
from accelerate import infer_auto_device_map, dispatch_model
from datasets import load_dataset
import evaluate
import torch 
from torch.utils.data import DataLoader

import numpy as np
import math

from data.dla.utils import DocLayNetSegmentationDataset
from models.dla_model import SegmentationModel
from utils import compute_metrics,id2label

n = 0 #model number in experiments folder for DIVE-Doc model type
path = f"../../experiments/model_{n}/dla"
#path = f"../../experiments/siglip_paligemma/dla"  #for paligemma ve evaluation
#path = f"../../experiments/swin_donut/dla"  #for donut ve evaluation

with open("../../token.json", "r") as f:
        hf_token = json.load(f)["HF_token"]
        
with open(f'{path}/config.json', 'r') as c:
    config = json.load(c)


#load processor and model
if config["encoder_type"] =="swin_qlora":
    with open(f'{config["base_model_path"]}/distillation_stage1/config.json','r') as f:
            swin_qlora_config = json.load(f)
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",token=hf_token)
    processor.image_processor.size["height"],processor.image_processor.size["width"] = swin_qlora_config["student_image_size"][0],swin_qlora_config["student_image_size"][1]

elif config["encoder_type"] =="siglip_paligemma":
    #processor from PaliGEMMA
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-docvqa-896",token=hf_token)

elif config["encoder_type"] =="swin_donut":
    #processor from Donut
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",token=hf_token)
    

model = SegmentationModel.from_pretrained(f"{path}/model")


if config["encoder_type"] in  ["swin_donut","swin_qlora"]:
    no_split_module_classes = ["DonutSwinStage"]

elif config["encoder_type"] == "siglip_paligemma":
    no_split_module_classes = ["SiglipVisionEmbeddings","SiglipEncoderLayer"]

#This is set up for our hardware configuration, feel free to change it in function of yours gpus
device_map = infer_auto_device_map(model,max_memory={d:"0.6GiB" for d in range(torch.cuda.device_count())}, 
                                       no_split_module_classes=no_split_module_classes,dtype=torch.float16)

model = dispatch_model(model,device_map)

#dataset
test_dataset = DocLayNetSegmentationDataset(
                                                raws=load_dataset("ds4sd/DocLayNet",split="test"),
                                                processor=processor,
                                                padding_index=config["ignore_index"],
                                                return_doc_cls=False,
                                                mask_size= None
                                            )
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


#loop eval
metric = evaluate.load("mean_iou")
list_batch_mean_iou = []
list_batch_per_category_iou = []
i=0

for mask,processed_image,doc_cls in test_loader:
    with torch.inference_mode():
        logits = model(processed_image.to(model.device)).logits

    batch_perfs = compute_metrics((logits,mask),metric=metric,num_labels=config["num_class"],ignore_index=config["ignore_index"])
    print(f"Mean IoU: {batch_perfs['mean_iou']}")
    list_batch_mean_iou.append(batch_perfs["mean_iou"])

    list_batch_per_category_iou.append(batch_perfs["per_category_iou"])


mean_iou = sum(list_batch_mean_iou)/len(list_batch_mean_iou)

mean_iou_per_category = np.nanmean(np.stack(list_batch_per_category_iou), axis=0).tolist()
mean_iou_per_category = {id2label[str(k+1)]:v for k,v in enumerate(mean_iou_per_category)}
mean_iou_per_category = {k:None if isinstance(v,float) and math.isnan(v) else v for k,v in mean_iou_per_category.items()}

results = { "mean_IoU":mean_iou,
            "mean_IoU_per_category":mean_iou_per_category}


with open(f"{path}/results.json", "w") as f:
    json.dump(results, f, indent=4)  