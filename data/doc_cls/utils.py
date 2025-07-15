import os
os.environ['HF_HOME'] = "../../.cache"
os.environ['HF_HUB_CACHE'] = "../../.cache"
os.environ['TRANSFORMERS_CACHE'] = "../../.cache"
os.environ['HF_DATASETS_CACHE'] = "../../.cache"

from datasets import load_dataset
import torch 
from torch.utils.data import Dataset as TorchDataset
from typing import Tuple


class RVLCDIPDataset(TorchDataset):

    def __init__(
        self,
        dataset_name_or_path: str,
        processor: None,
        split: str ,
        dataset_streaming: bool = False,
        subset_split : int = None
    ):
        super().__init__()


        self.dataset = load_dataset(dataset_name_or_path, split=split, streaming = dataset_streaming)
        if dataset_name_or_path == "aharley/rvl_cdip" and split =="test" and dataset_streaming == False :
            print("[INFO] Delete the unusable data [INFO]")
            self.dataset = self.dataset.select([i for i in range(len(self.dataset)) if i != 33669])

        self.dataset_length = self.dataset.info.splits[split].num_examples

        if subset_split!=None and subset_split >= 0:
            self.dataset = self.dataset.shuffle(seed=42)
            self.dataset = self.dataset.select(range(subset_split))
            self.dataset_length = subset_split
        self.split = split
        self.processor = processor
    
    def __len__(self) -> int:
        return self.dataset_length -1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]

        # input_tensor
        inputs = self.processor(images=sample["image"].convert("RGB"), return_tensors="pt").pixel_values[0]
        return inputs, sample["label"]