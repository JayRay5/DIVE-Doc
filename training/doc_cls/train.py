import json
import os
import sys
from pathlib import Path

parent_root = Path().resolve().parent.parent 
sys.path.append(str(parent_root))

os.environ['HF_HOME'] = "../../.cache"
os.environ['HF_HUB_CACHE'] = "../../.cache"
os.environ['TRANSFORMERS_CACHE'] = "../../.cache"
os.environ['HF_DATASETS_CACHE'] = "../../.cache"

from collections import Counter

from transformers import DonutProcessor,AutoProcessor, VisionEncoderDecoderModel, PaliGemmaForConditionalGeneration
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from peft import PeftModel

from models.model import DIVEdoc
from data.doc_cls.utils import RVLCDIPDataset
from models.cls_lightning_modules import ClsLightModule
from models.cls_model import CLSModel
from config import config

if config["encoder_type"] =="swin_qlora":
    base_model_path = config["base_model_path"]

elif config["encoder_type"] == "siglip_paligemma":
    base_model_path = f"../../experiments/siglip_paligemma"
    if "siglip_paligemma" not in os.listdir("../../experiments"):
        os.mkdir("../../experiments/siglip_paligemma")

elif config["encoder_type"] == "swin_donut":
    base_model_path = f"../../experiments/swin_donut"
    if "swin_donut" not in os.listdir("../../experiments"):
        os.mkdir("../../experiments/swin_donut")

path_curent_model_dir = base_model_path + "/cls"
os.mkdir(path_curent_model_dir)

with open(f"{path_curent_model_dir}/config.json", "w") as json_file:
    json.dump(config, json_file, indent=4)

with open("../../token.json", "r") as f:
        hf_token = json.load(f)["HF_token"]


if config["encoder_type"] =="swin_qlora":
    #SwinPAM ft qlora
    with open(f'{base_model_path}/distillation_stage1/config.json','r') as f:
        swin_qlora_config = json.load(f)
    patch_sequence_length = swin_qlora_config["teacher_features_dim"][0][0]*swin_qlora_config["teacher_features_dim"][0][1]
    embedding_dim = swin_qlora_config["teacher_features_dim"][1]

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",token=hf_token)
    processor.image_processor.size["height"],processor.image_processor.size["width"] = swin_qlora_config["student_image_size"][0],swin_qlora_config["student_image_size"][1]
    swinpamgemma = DIVEdoc.from_pretrained(f"{base_model_path}/distillation_stage1/divedoc_model")
    ve = PeftModel.from_pretrained(swinpamgemma, f"{base_model_path}/finetuning_stage2").base_model.model.vision_tower

elif config["encoder_type"] == "siglip_paligemma":
    #SigLIP from PaliGEMMA
    patch_sequence_length = 4096
    embedding_dim = 1152
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-docvqa-896",token = hf_token)
    ve = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-ft-docvqa-896",token = hf_token).vision_tower

elif config["encoder_type"] == "swin_donut":
    #Swin from Donut
    patch_sequence_length = 4800
    embedding_dim = 1024
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",token=hf_token)
    ve = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",token = hf_token).encoder
else:
     raise ValueError(f"The given model name is unkown, need 'swin_qlora', 'swin_donut' or 'siglip_paligemma', got {config["model_type"]}.")

model = CLSModel(encoder=ve, number_of_class=config["num_classes"], patch_sequence_length=patch_sequence_length, embedding_dim=embedding_dim)

#freeze the visual encoder parameters
model.visual_encoder.eval()
for param in model.visual_encoder.parameters():
    param.requires_grad = False
    

#load the dataset
dataset_name = "aharley/rvl_cdip"

train_dataset = RVLCDIPDataset(dataset_name_or_path=dataset_name, split="train", processor=processor, dataset_streaming=False, subset_split=config["training"]["subset_training"])
train_dataloader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

if config["training"]["subset_training"] != None and config["training"]["subset_training"]!=0:
    labels_list = train_dataset.dataset['label']
    count = Counter(labels_list)
    with open(f'{path_curent_model_dir}/training_samples_per_label.json', 'w') as f:
        json.dump(count, f)

val_dataset = RVLCDIPDataset(dataset_name_or_path=dataset_name, split="validation", processor=processor, dataset_streaming=False, subset_split=config["training"]["subset_validation"])
val_dataloader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

model_module = ClsLightModule(config=config, model=model,train_data_loader=train_dataloader,validation_data_loader=val_dataloader)

trainer = pl.Trainer(
                    accelerator="gpu",
                    devices=config["training"]["devices"],
                    default_root_dir=path_curent_model_dir,
                    accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
                    max_epochs=config["training"]["max_epochs"],
                    gradient_clip_val=config["training"]["gradient_clip_val"],
                    precision=config["training"]["precision"], 
                    num_sanity_val_steps=1, #set how many batches are checked before to start the training
                    logger=None,
                    )
trainer.fit(model_module) 