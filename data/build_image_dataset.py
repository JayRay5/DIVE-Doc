#file and os lib managements
import os

#To prevent the warning about deadlock when initializing the data loader
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

#Change the path of some HF environment variables to store the download data (model and dataset) from the hub to a choosen location
#PUT the datasets in data and transformers in models, other in main branch
os.environ['HF_HOME'] = "../.cache"
os.environ['HF_HUB_CACHE'] = "../.cache"
os.environ['TRANSFORMERS_CACHE'] = "../.cache"
os.environ['HF_DATASETS_CACHE'] = "../.cache"

#ML libraries
from  datasets import load_dataset


#utils
import numpy as np

dataset_name = "docvqa"
splits = ['train','validation','test']

def image_dataset_generation(dataset_name,split):
    if dataset_name == "docvqa":
        dataset = load_dataset("JayRay5/DocVQA", split=split, streaming = False)

    if dataset_name not in os.listdir():
        os.mkdir(f"./{dataset_name}")
    if "images" not in os.listdir(f"./{dataset_name}"):
        os.mkdir(f"./{dataset_name}/images")
    if split not in os.listdir(f"./{dataset_name}/images"):
        os.mkdir(f"./{dataset_name}/images/{split}")

    cols_to_remove = dataset.column_names
    cols_to_remove.remove("image")
    cols_to_remove.remove("other_metadata")

    reduce_dataset = dataset.remove_columns(cols_to_remove)

    reduce_dataset = reduce_dataset.map(lambda row: {'image_id':row['other_metadata']['image']})

    reduce_dataset = reduce_dataset.remove_columns(["other_metadata"])

    _, unique_indexes = np.unique(reduce_dataset['image_id'],return_index=True)

    unique_indexes = unique_indexes.tolist()

    images_docvqa = reduce_dataset.select(unique_indexes)


    images_docvqa.save_to_disk(f"./{dataset_name}/images/{split}")

for split in splits:
    image_dataset_generation(dataset_name,split)




