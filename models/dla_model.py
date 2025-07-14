import sys
from pathlib import Path
parent_root = Path().resolve().parent.parent.parent 

sys.path.append(str(parent_root))

import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, SiglipVisionModel, DonutSwinModel
from peft import PeftModel
from transformers.modeling_outputs import SemanticSegmenterOutput
from typing import Optional, Tuple, Union

from dla_config import SegmentationConfig
from .model import SwinPamVisionEncoder, DIVEdoc


#Si mauvaise performance, ajouter un MLP comme dans segformer
class SegmentationHead(PreTrainedModel):
    def __init__(self,config) -> None:
        super().__init__(config)
        self.mlp_head = nn.Sequential(
                                        nn.Linear(config.embedding_dim, config.embedding_dim // 2),
                                        nn.ReLU(),
                                        nn.Linear(config.embedding_dim // 2, config.embedding_dim // 4)
                                    )
        
        self.layernorm = (nn.LayerNorm(config.embedding_dim//4, eps=1e-06))
        self.activation = nn.ReLU()
        self.head = nn.Linear(config.embedding_dim//4, config.number_of_class)
        
        self.fmap_size = config.fmap_size
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp_head(x)
        x = self.layernorm(x)
        x = self.activation(x)  
        logits = self.head(x)
        batch_size,_,cls_number = logits.size()
        logits = logits.view(batch_size,self.fmap_size[0],self.fmap_size[1],cls_number).permute(0,3, 1, 2) # Shape: (1, num_labels, target_height, target_width)
        return logits
    


class SegmentationModel(PreTrainedModel):
    config_class = SegmentationConfig
    def __init__(self, config):
        super().__init__(config)
        if config.vision_config.model_type == "swinpam":
            self.vision_encoder = SwinPamVisionEncoder(config.vision_config)

        elif config.vision_config.model_type == "donut-swin":
            self.vision_encoder = DonutSwinModel(config.vision_config)
            
        elif config.vision_config.model_type == "siglip_vision_model":
            self.vision_encoder = SiglipVisionModel(config.vision_config)

        elif config.vision_config.model_type == "SwinQLoRA":
            self.vision_encoder = DIVEdoc.from_pretrained(config.vision_config.base_vision_encoder_decoder_name_or_path)
            self.vision_encoder = PeftModel.from_pretrained(self.vision_encoder,config.vision_config.peft_adapter_path,is_trainable=False).base_model.model.vision_tower
        
        self.decode_head = SegmentationHead(config.head_config)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_last_hidden_state: Optional[bool] = None,
        interpolation_mode: Optional[str] = "bilinear"
    ) -> Union[Tuple, SemanticSegmenterOutput]:
   

        if labels is not None and self.config.num_labels < 1:
            raise ValueError(f"Number of labels should be >=0: {self.config.num_labels}")

        outputs = self.vision_encoder(
            pixel_values
        )

        encoder_last_hidden_state = outputs.last_hidden_state 

        logits = self.decode_head(encoder_last_hidden_state)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            # upsample logits to the mask's size if different
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode=interpolation_mode
                )
            loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
            loss = loss_fct(logits, labels)
     
        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_last_hidden_state if output_last_hidden_state else None,
        )