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

from transformers import AutoProcessor, DonutProcessor, VisionEncoderDecoderModel, PaliGemmaForConditionalGeneration
import torch 
from torch.utils.data import DataLoader
from peft import PeftModel

from data.doc_cls.utils import RVLCDIPDataset
from models.model import DIVEdoc
from models.cls_model import CLSModel


DEVICE = 1
BATCH_SIZE = 2
n = 0 #model number in experiments folder for DIVE-Doc model type
path = f"../../experiments/model_{n}/cls"
#path = f"../../experiments/siglip_paligemma/cls"  #for paligemma ve evaluation
#path = f"../../experiments/swin_donut/cls"  #for donut ve evaluation

with open("../../token.json", "r") as f:
        hf_token = json.load(f)["HF_token"]
        
with open(f'{path}/config.json', 'r') as c:
    config = json.load(c)

if config["encoder_type"] =="swin_qlora":
    #SwinPAM ft qlora
    base_model_path = config["base_model_path"]
    with open(f'{base_model_path}/distillation_stage1/config.json','r') as f:
        swin_qlora_config = json.load(f)

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",token=hf_token)
    processor.image_processor.size["height"],processor.image_processor.size["width"] = swin_qlora_config["student_image_size"][0],swin_qlora_config["student_image_size"][1]

    model = DIVEdoc.from_pretrained(f"{base_model_path}/distillation_stage1/divedoc_model")
    ve = PeftModel.from_pretrained(model, f"{base_model_path}/finetuning_stage2").base_model.model.vision_tower
    patch_sequence_length = swin_qlora_config["teacher_features_dim"][0][0]*swin_qlora_config["teacher_features_dim"][0][1]
    embedding_dim = swin_qlora_config["teacher_features_dim"][1]
  

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

#load weights
saved_path_weights_file = os.listdir(f"{path}/lightning_logs/version_0/checkpoints/")[-1]
checkpoint = torch.load(f"{path}/lightning_logs/version_0/checkpoints/{saved_path_weights_file}")
model.load_state_dict(checkpoint['state_dict'])

model.to(f"cuda:{DEVICE}")
model.eval() 



print("[INFO] Load model & dataset [INFO]")
dataset = RVLCDIPDataset(dataset_name_or_path = "aharley/rvl_cdip", split="test",processor=processor,dataset_streaming =False)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


print("[INFO] Evaluation Start [INFO]")
with torch.inference_mode():
    predicted_class = []
    labels_list = []

    for b,batch in enumerate(dataloader):
        inputs, labels = batch
        inputs = inputs.to(f"cuda:{DEVICE}")
        logits = model(inputs)

        # model predicts one of the 16 RVL-CDIP classes
        for l,logit in enumerate(logits):
            pred_idx = logit.argmax(-1).item()
            predicted_class.append(pred_idx)
            label_idx = labels[l].item()

            labels_list.append(label_idx)
            print(f"Prediction : {pred_idx}, Label : {label_idx}")
        
        if len(predicted_class) != len(labels_list):
            raise ValueError(f"Prediction list and labels list should have the same length, got {len(predicted_class)} and {len(labels_list)}")
        print(f"[INFO] End of Batch {b} [INFO]")
    
    correct_predictions = sum(p == l for p, l in zip(predicted_class, labels_list))
    accuracy = correct_predictions / len(labels_list)
    print(f"Acc : {accuracy}")


if isinstance(accuracy, torch.Tensor):
    accuracy = accuracy.tolist()

results = {
            "accuracy": round(accuracy,2)
          }

with open(f"{path}/results.json","w") as f:
    json.dump(results,f)