import json
import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

#Change the path of some HF environment variables to store the download data (model and dataset) from the hub to a choosen location
os.environ['HF_HOME'] = "./.cache"
os.environ['HF_HUB_CACHE'] = "./.cache"
os.environ['TRANSFORMERS_CACHE'] = "./.cache"
os.environ['HF_DATASETS_CACHE'] = "./.cache"

import gradio as gr
import torch
from transformers import AutoProcessor, DonutProcessor
from accelerate import infer_auto_device_map, dispatch_model
from peft import PeftModel

from models.model import DIVEdoc

def app(path,stage_2_model=True):
    with open("./token.json", "r") as f:
            hf_token = json.load(f)["HF_token"]

    with open(f'{path}/distillation_stage1/config.json','r') as f:
        config = json.load(f)

    processor = AutoProcessor.from_pretrained("google/paligemma-3b-ft-docvqa-896",token = hf_token)
    if config["student_name"] == "swinpam":
        processor.image_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa",token=hf_token).image_processor
        processor.image_processor.size = {'height': config["student_image_size"][0], 'width': config["student_image_size"][1]}

    elif config["student_name"] == "siglip80m":
        processor.image_processor.size = {'height': config["student_image_size"][0], 'width': config["student_image_size"][1]}

    model = DIVEdoc.from_pretrained(f"{path}/distillation_stage1/divedoc_model",torch_dtype=torch.float16)
    device_map = infer_auto_device_map(model,max_memory={1: "5GiB"}, 
                                        no_split_module_classes=["DonutSwinStage","GemmaDecoderLayer"])
        
    model = dispatch_model(model,device_map)

    if stage_2_model:
        if "finetuning_stage2" in os.listdir(path):
                model = PeftModel.from_pretrained(model, f"{path}/finetuning_stage2")
        else:
            raise ValueError("Cannot find a model for the stage 2 in '{path}/finetuning_stage2'")
    model.eval()


    def answer_question(image, question):
        # Process the image and question
        model_inputs = processor(text=question, images=image, return_tensors="pt",padding=True)
        model_inputs = model_inputs.to(model.device,dtype=torch.float16)
        input_len = model_inputs["input_ids"].shape[-1]

        print("[INFO] INFERENCE STARTING [INFO]")

        # Answer generation
        with torch.inference_mode():
            pred = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)[0][input_len:]
            print("[INFO] INFERENCE ENDING [INFO]")
        answer = processor.decode(pred, skip_special_tokens=True)

        print(f"Question:{question}\nAnswer:{answer}")
        return answer

    interface = gr.Interface(
        fn=answer_question,
        inputs=[gr.Image(type="pil"), gr.Textbox(label="Question")],
        outputs=gr.Textbox(label="Answer"),
        title="Visual Question Answering",
        description="Upload an image of document and ask a question related to the image. The model will try to answer it. \nNote: Processing time depends on whether you’re running the model on a CPU or a GPU."
    ).queue()

    interface.launch( server_name="0.0.0.0",
            server_port=7860)