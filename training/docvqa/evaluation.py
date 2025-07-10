#file and os lib managements
import json
import os
import tqdm
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
from transformers import DonutProcessor, AutoProcessor
from datasets import load_dataset
from accelerate import infer_auto_device_map, dispatch_model
from peft import PeftModel

import torch

#utils
import time 
from models.model import DIVEdoc


def test(path,eval_stage2=False):
    with open("../token.json", "r") as f:
        hf_token = json.load(f)["HF_token"]

    #create a new folder to save the results
    results_path = "{}/results".format(path)
    if "results" not in os.listdir(path):
        os.mkdir(results_path)


    with open(f'{path}/distillation_stage1/config.json','r') as f:
        config = json.load(f)

    '''

    STUDENT LOADING

    '''
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-docvqa-896",token = hf_token)
    if config["student_name"] == "swinpam":
        processor.image_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",token=hf_token).image_processor
        processor.image_processor.size = {'height': config["student_image_size"][0], 'width': config["student_image_size"][1]}

    elif config["student_name"] == "siglip80m":
        processor.image_processor.size = {'height': config["student_image_size"][0], 'width': config["student_image_size"][1]}

    model = DIVEdoc.from_pretrained(f"{path}/distillation_stage1/divedoc_model")

    if eval_stage2:
        if "finetuning_stage2" in os.listdir(path):
                model = PeftModel.from_pretrained(model, f"{path}/finetuning_stage2")
        else:
            raise ValueError("Cannot find a model for the stage 2 in '{path}/finetuning_stage2'")

    """

    Evaluation 


    """

    test_dataset = load_dataset("pixparse/docvqa-single-page-questions", split="test",streaming=True)
    batch_dataset = test_dataset.batch(2) 

    device_map = infer_auto_device_map(model,max_memory={d:"5GiB" for d in range(torch.cuda.device_count())}, 
                                       no_split_module_classes=["DonutSwinStage","GemmaDecoderLayer"])
    model = dispatch_model(model,device_map)
    pred_list = []




    print("[INFO] Generate answers on the test set [INFO]")
    with torch.inference_mode():
        start = time.time()
        for batch in tqdm.tqdm(batch_dataset):
            #preprocessing 
            imgs = []
            for i in batch["image"]:
                imgs.append(i.convert('RGB'))
            txt = batch["question"]
            model_inputs = processor(text=txt, images=imgs, return_tensors="pt",padding=True)
            input_len = model_inputs["input_ids"].shape[-1]

            #processing
            model_inputs = model_inputs.to(model.device)
            generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)

            #postprocessing 
            for i in range(len(generation)):
                pred = generation[i][input_len:]
                decoded = processor.decode(pred, skip_special_tokens=True)
                pred_list.append({'answer': decoded, 'questionId': batch['question_id'][i]})
                print("QuestionId : {}, Question : {}, Reponse : {}".format(batch["question_id"][i],batch["question"][i],decoded))

        end = time.time()
        inference_time = end - start
        test_docvqa_dict = {"inference_time":inference_time}
        if eval_stage2:
            time_file = "model_stage2_time_docvqa_test.json"
            results_file = "model_stage2_docvqa_test_results.json"
        else:
            time_file = "model_stage1_time_docvqa_test.json"
            results_file = "model_stage1_docvqa_test_results.json"
        with open('{}/{}'.format(results_path,time_file), 'w') as f:
            json.dump(test_docvqa_dict, f)

        with open('{}/{}'.format(results_path,results_file), 'w') as f:
            json.dump(pred_list, f)



if __name__== "__main__":
    default_path = "../../experiments/model_0"
    test(default_path,qualitative_results=False,qlora=False)
