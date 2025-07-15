import json
import os

os.environ['HF_HOME'] = "../../.cache"
os.environ['HF_HUB_CACHE'] = "../../.cache"
os.environ['TRANSFORMERS_CACHE'] = "../../.cache"
os.environ['HF_DATASETS_CACHE'] = "../../.cache"

from transformers import DonutProcessor,AutoProcessor, PaliGemmaForConditionalGeneration, VisionEncoderDecoderModel, TrainingArguments, Trainer
from accelerate import infer_auto_device_map, dispatch_model
from datasets import load_dataset
import torch 

from data.dla.utils import DocLayNetSegmentationDataset
from models.dla_model import SegmentationModel
from models.dla_config import SegmentationConfig,DecoderSegmentationConfig, SwinQLoRAConfig
from models.config_divedoc import get_siglip_vision_config, get_swin_vision_config
from utils import collate_fn
from config import base_model_path, model,config

with open("../../token.json", "r") as f:
        hf_token = json.load(f)["HF_token"]

"""
        Set Files
"""

#to train a model with donut or paligemma ve
if model[0]!="swin_qlora":
    experiments_path = "../../experiments"
    base_model_path = f"{experiments_path}/{model[0]}"
    if model[0] not in os.listdir(experiments_path):
        os.mkdir(base_model_path)

    experiment_path = f"{base_model_path}/dla"
    if "dla" not in base_model_path:
        os.mkdir(experiment_path)
    
    model_path = f"{experiment_path}/model"
    os.mkdir(model_path)

#to train a model DIVE-Doc ve
else:
    experiment_path = f"{base_model_path}/dla"
    if "dla" not in os.listdir(base_model_path):
        os.mkdir(experiment_path)

    model_path = f"{experiment_path}/model"
    os.mkdir(model_path)

with open(f"{experiment_path}/config.json", "w") as f:
    json.dump(config, f, indent=4)  


"""
        Load Model and data
"""

if config["encoder_type"] =="swin_qlora":
    #SwinPAM ft qlora
    with open(f"{base_model_path}/distillation_stage1/config.json","r") as f:
        swin_qlora_config = json.load(f)
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",token=hf_token)
    processor.image_processor.size["height"],processor.image_processor.size["width"] = swin_qlora_config["student_image_size"][0],swin_qlora_config["student_image_size"][1]
    vision_config = SwinQLoRAConfig(base_vision_encoder_decoder_name_or_path=f"{base_model_path}/distillation_stage1/divedoc_model",peft_adapter_path=f"{base_model_path}/finetuning_stage2")
    decoder_config = DecoderSegmentationConfig(number_of_class=config["num_class"])

elif config["encoder_type"] =="siglip_paligemma":
    #SigLIP from PaliGEMMA
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-docvqa-896",token = hf_token)
    vision_config = get_siglip_vision_config()
    decoder_config = DecoderSegmentationConfig(number_of_class=config["num_class"])

elif config["encoder_type"] =="swin_donut":
    #Swin from Donut
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",token=hf_token)
    vision_config = get_swin_vision_config()
    decoder_config = DecoderSegmentationConfig(number_of_class=config["num_class"],
                                               embedding_dim = 1024,fmap_size=[80,60])


model_config  = SegmentationConfig(vision_config=vision_config,head_config=decoder_config,semantic_loss_ignore_index=config["ignore_index"])
model = SegmentationModel(config=model_config)


#Load weights
#Note: for finetuning qlora models, weights are loaded during the segmentation model class instantiation
if model.config.vision_config.model_type == "siglip_vision_model":
   paligemma_ve = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-ft-docvqa-896",token = hf_token).vision_tower
   checkpoint = paligemma_ve.state_dict()
   model.vision_encoder.load_state_dict(checkpoint)

elif model.config.vision_config.model_type == "donut-swin":
   donut_ve = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",token = hf_token).encoder
   checkpoint = donut_ve.state_dict()
   model.vision_encoder.load_state_dict(checkpoint)
    
model.vision_encoder.eval()
for param in model.vision_encoder.parameters():
    param.requires_grad = False

train_dataset = DocLayNetSegmentationDataset(
                                                raws=load_dataset("ds4sd/DocLayNet",split="train"),
                                                processor=processor,
                                                padding_index=config["ignore_index"],
                                                mask_size=None
                                            )

#This is set up for our hardware configuration, feel free to change it in function of yours gpus
model_gib_per_device = "1GiB"
if config["encoder_type"] in  ["swin_donut","swin_qlora"]:
    no_split_module_classes = ["DonutSwinStage"]
    model_gib_per_device = "0.2GiB"

elif config["encoder_type"] == "siglip_paligemma":
    no_split_module_classes = ["SiglipVisionEmbeddings","SiglipEncoderLayer"]
    model_gib_per_device = "0.5GiB"

device_map = infer_auto_device_map(model,max_memory={d:model_gib_per_device for d in range(torch.cuda.device_count())}, 
                                       no_split_module_classes=no_split_module_classes,dtype=torch.float16)

    
model = dispatch_model(model,device_map)



"""
        Training
"""
training_args = TrainingArguments(
    num_train_epochs=config["epochs"],
    output_dir=model_path,
    learning_rate=config["learning_rate"],
    #lr_scheduler_type="linear",
    #lr_scheduler_kwargs ="", args for schedurler (dict)
    per_device_train_batch_size=config["batch_size"],
    gradient_accumulation_steps=config["grad_acc"],
    fp16 =True,
    save_total_limit=2,
    #per_device_eval_batch_size=1,
    #eval_strategy="steps",
    #eval_steps=1,
    save_strategy="steps",
    remove_unused_columns=False,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    #eval_dataset=val_dataset,
    #compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(model_path)