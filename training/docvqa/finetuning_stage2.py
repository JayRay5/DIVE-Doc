import os 
import random
import sys
import json
from pathlib import Path
parent_root = Path().resolve().parent.parent

sys.path.append(str(parent_root))

#To prevent the warning about deadlock when initializing the data loader
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

#Change the path of some HF environment variables to store the download data (model and dataset) from the hub to a choosen location
os.environ['HF_HOME'] = "../../.cache"
os.environ['HF_HUB_CACHE'] = "../../.cache"
os.environ['TRANSFORMERS_CACHE'] = "../../.cache"
os.environ['HF_DATASETS_CACHE'] = "../../.cache"

from transformers import DonutProcessor,AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import get_peft_model,LoraConfig
from datasets import load_dataset
import torch

from models.model import DIVEdoc


def train(path):
    base_model_path = f"{path}/distillation_stage1/divedoc_model"

    if f"finetuning_stage2" not in os.listdir(path):
        os.makedirs(f"{path}/finetuning_stage2")
    saved_path = f"{path}/finetuning_stage2"

    with open("../../token.json", "r") as f:
        hf_token = json.load(f)["HF_token"]

    train_dataset = load_dataset("pixparse/docvqa-single-page-questions", split="train",streaming=False)

    with open(f'{path}/config.json','r') as f:
        config = json.load(f)

    #processor
    if config["student_name"] == "swinpam":
        processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-docvqa-896",token = hf_token)
        processor.image_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",token=hf_token).image_processor
        processor.image_processor.size = {'height': config["student_image_size"][0], 'width': config["student_image_size"][1]}

    elif config["student_name"] == "siglip80m":
        processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-docvqa-896",token = hf_token)
        processor.image_processor.size = {'height': config["student_image_size"][0], 'width': config["student_image_size"][1]}

    #device_map for a split on 2 gpu    
    device2_map = {
                    "vision_tower": "cuda:0",
                    "multi_modal_projector": "cuda:0",

                    "language_model.model.embed_tokens": "cuda:0",

                    "language_model.model.layers.0": "cuda:0",
                    "language_model.model.layers.1": "cuda:0",
                    "language_model.model.layers.2": "cuda:0",

                    "language_model.model.layers.3": "cuda:0",
                    "language_model.model.layers.4": "cuda:0",
                    "language_model.model.layers.5": "cuda:1",
                    "language_model.model.layers.6": "cuda:1",

                    "language_model.model.layers.7": "cuda:1",
                    "language_model.model.layers.8": "cuda:1",
                    "language_model.model.layers.9": "cuda:1",
                    "language_model.model.layers.10": "cuda:1",
                    "language_model.model.layers.11": "cuda:1",
                    "language_model.model.layers.12": "cuda:1",
                    "language_model.model.layers.13": "cuda:1",
                    "language_model.model.layers.14": "cuda:1",
                    "language_model.model.layers.15": "cuda:1",
                    "language_model.model.layers.16": "cuda:1",
                    "language_model.model.layers.17": "cuda:1",

                    "language_model.model.norm": "cuda:0",
                    "language_model.model.rotary_emb": "cuda:0",
                    "language_model.lm_head": "cuda:0"
                }
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, # stored the weight in 4bit 
        bnb_4bit_quant_type="nf4",  # Normalized Float 4
        bnb_4bit_compute_dtype=torch.float16, #dtype to switch for operation
        bnb_4bit_use_double_quant=True #add a second quantization for the constant c use to quantize parameters
    )

    learning_rate = 3e-5
    divedoc = DIVEdoc.from_pretrained(base_model_path,quantization_config=bnb_config,device_map=device2_map)

    lora_config = LoraConfig(r=16, target_modules="all-linear")
    divedoc = get_peft_model(divedoc,lora_config)
    
    training_args = TrainingArguments(
        output_dir=saved_path,
        overwrite_output_dir = True,
        save_strategy="steps",
        num_train_epochs = 3,
        learning_rate = learning_rate, 
        #optim="",default="adamw_torch"
        weight_decay=0.01, #default 0
        adam_beta1=0.9,
        adam_beta2=0.999, #default 0.999
        adam_epsilon =1e-8, #default 1e-8
        lr_scheduler_type="linear",
        warmup_ratio =0.15,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 16,
        fp16 =True,
        save_total_limit =2,
        dataloader_pin_memory =False,
        push_to_hub=False,
        remove_unused_columns=False
    )


    def collate_fn(batch):
        images = []
        answers = []
        questions = []
        
        for example in batch:
            images.append(example["image"].convert("RGB"))
            answers.append(random.choice(example["answers"]))
            questions.append(example["question"])

        tokens = processor(text=questions, images=images, suffix=answers, return_tensors="pt", padding=True)
        tokens = tokens.to(dtype=torch.float16)
        return tokens

    trainer = Trainer(
        divedoc,
        training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn, #preprocessing function
    )

    try:
        trainer.train()
        trainer.save_model(saved_path)
    except Exception as e:
        print(f"Échec à la sauvegarde du checkpoint : {e}", file=sys.stderr)
        sys.exit(1)
    print("[INFO]END TRAINING[INFO]")

if __name__ == "__main__":
    path = ""
    train(path)
