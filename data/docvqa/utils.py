import h5py
import json
from tqdm import tqdm
import os

from  datasets import load_dataset, load_from_disk
from transformers import Pipeline
import torch
from torch.utils.data import Dataset as TorchDataset

import numpy as np
from typing import List, Tuple


class ImageDataset(TorchDataset):

    def __init__(
        self,
        dataset_name_or_path: str,
        processor: None,
        teacher_processor : None,
        split: str ,
        path_teacher_features: str = '',
        table_index_teacher_feature: dict = {},
    ):
        super().__init__()

        self.split = split
        self.path_teacher_features = path_teacher_features
        self.table_index_teacher_feature = table_index_teacher_feature
 
        
        self.dataset = load_dataset(dataset_name_or_path)
        self.dataset_length = self.dataset.num_rows
        self.processor = processor
        self.teacher_processor = teacher_processor
    
    def __len__(self) -> int:
        return self.dataset_length[self.split] - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[self.split][idx]

        # input_tensor
        input_tensor = self.processor(sample["image"].convert("RGB"), return_tensors="pt").pixel_values.squeeze() 

        if self.teacher_processor == None:
            if len(self.path_teacher_features) > 0 and len(self.table_index_teacher_feature)>0:
                #get teacher outputfeatures
                with h5py.File(self.path_teacher_features, 'r') as f: 
                    # Access the dataset
                    if sample['image_id'] in self.table_index_teacher_feature.keys():
                        index_teacher_output_features = self.table_index_teacher_feature[sample['image_id']]
                        teacher_output_features = f[self.split][index_teacher_output_features][1]
                    else:
                        raise KeyError(f'The image id is not in the index table of the teacher output dataset! Looking for : {sample["image_id"]}')

                    return input_tensor,teacher_output_features
            else:
                raise ValueError("teacher processor not specified and there is an issue with the teacher's features path.")
        else:
            teacher_input_tensor = self.teacher_processor(sample["image"].convert("RGB"), return_tensors="pt").pixel_values
            teacher_input_tensor = teacher_input_tensor.squeeze()
            return input_tensor,teacher_input_tensor
        


"To generate and save teacher's embeddings"
class GenerationEmbeddingsPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}
        if "processor" in kwargs:
            preprocess_kwargs["processor"] = kwargs["processor"]
        if "model" in kwargs:
            forward_kwargs["model"] = kwargs["model"]           
        

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs):
        images = None
        if type(inputs['image']) == list:
            images = []
            for i in inputs['image']:
                images.append(i.convert('RGB'))
        else: 
            images = inputs['image'].convert('RGB')
        images = self.processor(images, return_tensors="pt").pixel_values
        
        
        return {'images':images,'image_name':inputs['image_id']}

    def _forward(self, data):
        outputs =  self.model(data['images'])
        return {'features_embedding':outputs,'image_name':data['image_name']}

    def postprocess(self, data):
        postprocess_features = data['features_embedding'].last_hidden_state
        return {'features_embedding':postprocess_features,'image_name':data['image_name']}
    

def append_to_hdf5(file_path, new_data, dataset_name, chunk_size=100):
    image_name = np.array(new_data['image_name'])
    teacher_features_output = np.array(new_data['teacher_features_output'])
    # Convert new_data to a structured NumPy array
    # Open HDF5 file in append mode
    with h5py.File(file_path, 'a') as f:
        if dataset_name in f:
            # If dataset exists, get the current shape and extend
            dataset = f[dataset_name]
            current_shape = dataset.shape[0]
            new_shape = current_shape + image_name.shape[0]
            
            # Resize the dataset to accommodate new rows
            dataset.resize(new_shape, axis=0)
            new_data = np.array(list(zip(image_name, teacher_features_output)), dtype=dataset.dtype)
            dataset[current_shape:] = new_data
        else:
            # If dataset does not exist, create a new dataset
            dtype = np.dtype([('image_name', h5py.string_dtype(encoding='utf-8')), ('teacher_features_output', 'f4', teacher_features_output.shape[-2:])])
            new_data = np.array(list(zip(image_name, teacher_features_output)), dtype=dtype)
            f.create_dataset(dataset_name, data=new_data, maxshape=(None,))


def generate_and_save_embeddings(pipeline:Pipeline,dataset_name:str,split_names:List[str],dataset_path:str,saved_path:str,load_from_disk_bool=True):
    with open("../token.json", "r") as f:
        hf_token = json.load(f)["HF_token"]
    for split_name in split_names:
        if load_from_disk_bool:
            dataset = load_from_disk(f"{dataset_path}/{split_name}")
        else:
            dataset = load_dataset(dataset_name, split=split_name, streaming=False,token = hf_token)
        processed_data = {'image_name':[], 'teacher_features_output':[]}
        image_name_list = []
        with torch.inference_mode():
            for i,sample in tqdm(enumerate(dataset),desc="Processing dataset"):
                print(i)
                if sample['image_id'] not in image_name_list:
                    image_name_list.append(sample['image_id'])
                    output = pipeline(sample)
                    processed_data['image_name'].append(output['image_name'])
                    processed_data['teacher_features_output'].append(output['features_embedding'][0])
                
                    append_to_hdf5(f"{saved_path}/{dataset_name}_paligemma_embeddings.h5",processed_data,split_name)
                    processed_data = {'image_name':[], 'teacher_features_output':[]}

        with h5py.File(f"{saved_path}/{dataset_name}_paligemma_embeddings.h5", "r") as f:
            # Assuming 'document_id' is in a dataset named 'data'
            print('charging')
            document_ids = f[split_name]
            print('debut boucle')
            # Create a dictionary comprehension to store document_id as keys and indices as values
            doc_id_to_index = {line[0]: idx for idx, line in enumerate(document_ids)}

        doc_id_to_index = {key.decode('utf-8'): value for key, value in doc_id_to_index.items()}

        with open(f"{saved_path}/teacher_features_table_{split_name}.json", "w") as json_file:
            json.dump(doc_id_to_index, json_file, indent=4)

def check_subfolder_in_folder(subfolder_name,path="./"):
    if subfolder_name not in os.listdir(path):
        return False
    return True