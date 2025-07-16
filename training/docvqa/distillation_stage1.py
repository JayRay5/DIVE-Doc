#file and os lib managements
import json
import os
import sys
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

#ML libraries
from transformers import DonutProcessor, VisionEncoderDecoderModel, PaliGemmaForConditionalGeneration, AutoProcessor

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

#utils
from config import config
from data.docvqa.utils import ImageDataset
from models.lightning_modules import Distillation
from models.config_divedoc import get_model_config,get_vision_config
from models.model import DIVEdoc, SiglipPAMVisionEncoder, SwinPamVisionEncoder

#os.environ["CUDA_VISIBLE_DEVICES"] = 2
print(f"Available device : {torch.cuda.device_count()}")  # Should print 1 if only GPU 2 is visible

def main():
    with open("../token.json", "r") as f:
        hf_token = json.load(f)["HF_token"]
    #create a new folder for the new trained model
    path_saved_models_dir = "../../experiments"

    model_version_number = len(os.listdir(path_saved_models_dir))
    path_curent_model_dir = path_saved_models_dir + "/model_{}".format(model_version_number)
    os.mkdir(path_curent_model_dir)

    path_stage1 = f"{path_curent_model_dir}/distillation_stage1"
    os.mkdir(path_stage1)

    #save the config file
    with open('{}/config.json'.format(path_stage1), 'w') as f:
        json.dump(config, f,indent=4)

    #processor
    if config["student_name"] == "swinpam":
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",token=hf_token)
        processor.image_processor.size = {'height': config["student_image_size"][0], 'width': config["student_image_size"][1]}

    elif config["student_name"] == "siglip80m":
        processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-docvqa-896",token = hf_token)
        processor.image_processor.size = {'height': config["student_image_size"][0], 'width': config["student_image_size"][1]}
        processor.image_processor.image_seq_length  = config["student_features_dim"][0][0]*config["student_features_dim"][0][1]

    if config["connected_teacher"]:
        teacher_processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-docvqa-896",token = hf_token)
    else:
        teacher_processor = None

    '''

    DATA LOADING

    '''

    #dataset loading
    if config["connected_teacher"]:
        table_index_train_paligemma_feature = None
        table_index_validation_paligemma_feature = None
        
    else:
        with open(config["teacher_features_indexing_table_train_path"], "r") as f:
            table_index_train_paligemma_feature = json.load(f)

        with open(config["teacher_features_indexing_table_validation_path"], "r") as f:
            table_index_validation_paligemma_feature = json.load(f)

    train_dataset = ImageDataset(
                                config["dataset"], processor=processor, teacher_processor= teacher_processor,
                                split="train", path_teacher_features = config["teacher_features_path"],
                                table_index_teacher_feature = table_index_train_paligemma_feature
                                )

    validation_dataset = ImageDataset(
                                config["dataset"], processor=processor,  teacher_processor= teacher_processor,
                                split="validation", path_teacher_features = config["teacher_features_path"],
                                table_index_teacher_feature = table_index_validation_paligemma_feature,
                                )

    train_dataloader = DataLoader(train_dataset, batch_size=config["train_batch_sizes"][0], shuffle=True, num_workers=4)
    validation_data_loader = DataLoader(validation_dataset, batch_size=config["val_batch_sizes"][0], shuffle=False, num_workers=4)

    '''

    MODEL LOADING

    '''

    teacher_model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-ft-docvqa-896",token = hf_token)

    ve_config = get_vision_config( visual_encoder_type = config["student_name"],
                image_size=config["student_image_size"] if config["student_name"] == "swinpam" else config["student_image_size"][0],
                sequence_mapping_layer_type=config["patch_alignement_type"],
                student_fmap_dim=config["student_features_dim"][0],
                student_embedding_dim= config["student_features_dim"][1],
                teacher_fmap_dim= config["teacher_features_dim"][0],
                teacher_embedding_dim= config["teacher_features_dim"][1])
    
    if config["student_name"] == "swinpam":
        vision_encoder = SwinPamVisionEncoder(ve_config).model
        if config["donut_weight"]:
            donut = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",token= hf_token)
            vision_encoder.encoder_model.load_state_dict(donut.encoder.state_dict())

    elif config["student_name"] == "siglip80m":
        vision_encoder = SiglipPAMVisionEncoder(ve_config).model
        
    if config["connected_teacher"]:
        model_module = Distillation(config, vision_encoder,teacher = [teacher_model.vision_tower],train_data_loader=train_dataloader,validation_data_loader=validation_data_loader)

    else:
         model_module = Distillation(config, vision_encoder,teacher = [teacher_model.vision_tower],train_data_loader=train_dataloader,validation_data_loader=validation_data_loader)

    for param in model_module.student.parameters():
        param.requires_grad = True 


    '''

    Training 

    '''
    
    if type(config.get("devices")) == list:
        device = 1
    else:
        device = config.get("devices")
    model_module = model_module.train()

    trainer = pl.Trainer(
                        accelerator="gpu",
                        devices=device,
                        default_root_dir=path_stage1,
                        accumulate_grad_batches=config.get("accumulate_grad_batches"),
                        max_epochs=config.get("max_epochs"),
                        gradient_clip_val=config.get("gradient_clip_val"),
                        precision=config.get("precision"), # we'll use mixed precision
                        num_sanity_val_steps=1, #set how many batches are checked before to start the training
                        logger=None
                        )
    trainer.fit(model_module)
    

    os.mkdir(f"{path_stage1}/divedoc_model")
    config_model = get_model_config(visual_encoder_type = config["student_name"],
                                    image_size = config["student_image_size"] if config["student_name"]== "swinpam" else config["student_image_size"][0],
                                    sequence_mapping_layer_type = config["patch_alignement_type"],
                                    student_fmap_dim = config["student_features_dim"][0],
                                    student_embedding_dim = config["student_features_dim"][1],
                                    teacher_fmap_dim = config["teacher_features_dim"][0],
                                    teacher_embedding_dim = config["teacher_features_dim"][1])
    model = DIVEdoc(config = config_model)
    model.vision_tower.model = model_module.student
    model.multi_modal_projector = teacher_model.multi_modal_projector
    model.language_model = teacher_model.language_model

    model.save_pretrained(f"{path_stage1}/divedoc_model")
 

if __name__ == "__main__":
    main() 