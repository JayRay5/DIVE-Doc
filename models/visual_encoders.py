from typing import Literal, Tuple
from torch import Tensor

import torch
import torch.nn as nn
from transformers import AutoConfig, SiglipVisionModel
from transformers.models.donut.modeling_donut_swin import DonutSwinModel

import torch.nn.functional as F

class PAM(nn.Module):
    def __init__(
        self,
        sequence_mapping_layer_type: Literal["linear_projection","bilinear","bicubic","nearest-exact"] = "bilinear",
        student_fmap_dim: Tuple[int,int]=(80,60),
        student_embedding_dim: int = 1024,
        teacher_fmap_dim: Tuple[int,int] = (64,64),
        teacher_embedding_dim: int = 1152
                ):
        super().__init__()
        self.sequence_mapping_layer_type = sequence_mapping_layer_type
        self.sequence_mapping_layer = nn.Linear(student_fmap_dim[0]*student_fmap_dim[1],teacher_fmap_dim[0]*teacher_fmap_dim[1]) if sequence_mapping_layer_type == "linear_projection" else None
        self.embedding_projection_layer = nn.Sequential(
            nn.Linear(student_embedding_dim,teacher_embedding_dim),
            nn.LayerNorm((teacher_embedding_dim,),eps=1e-06))
        
        self.student_fmap_dim = student_fmap_dim
        self.student_embedding_dim = student_embedding_dim
        self.teacher_fmap_dim = teacher_fmap_dim
        self.teacher_embedding_dim = teacher_embedding_dim
        
        print(self.student_fmap_dim)
    #take input x of shape (Batch, Nb_token, Dim_embedding)
    def forward(self,x:Tensor) -> Tensor:
        #
        '''
        if x.shape[1] != self.student_fmap_dim[0] * self.student_fmap_dim[1] or x.shape[2] != self.student_embedding_dim:
            raise ValueError(f"Expected input shape (*, {self.student_fmap_dim[0] * self.student_fmap_dim[1],self.student_embedding_dim}), "
                             f"but got {x.shape}")
        '''
        
        if x.shape[1]!=(self.teacher_fmap_dim[0]*self.teacher_fmap_dim[1]):
            print(x.shape[1])
            print(self.teacher_fmap_dim[0]*self.teacher_fmap_dim[1])
            print("Resizing")
            if self.sequence_mapping_layer_type == "linear_projection":
                x = torch.permute(x,(0,2,1))
                x = self.sequence_mapping_layer(x)
                x = torch.permute(x,(0,2,1))

            elif self.sequence_mapping_layer_type in ["bilinear","bicubic","nearest-exact"]:
                batch_size,_,embedding_size = x.size()
                x = x.view(batch_size,self.student_fmap_dim[0],self.student_fmap_dim[1],embedding_size).permute(0,3, 1, 2)
                x = F.interpolate(x,size=self.teacher_fmap_dim,mode=self.sequence_mapping_layer_type)  # Shape: (1, D, target_height, target_width)
                x = x.permute(0,2, 3, 1).reshape(batch_size,-1, embedding_size)
            
        x = self.embedding_projection_layer(x)
        return x
 
class SwinPam(nn.Module):
    def __init__(
        self,
        encoder_config: AutoConfig,
        pam_sequence_mapping_layer_type: Literal["linear_projection","bilinear","bicubic","nearest-exact"] = "bilinear",
        pam_student_fmap_dim: Tuple[int,int] = (80,60),
        pam_student_embedding_dim: int = 1024,
        pam_teacher_fmap_dim: Tuple[int,int] = (64,64),
        pam_teacher_embedding_dim: int = 1152
        ):
        super().__init__()
        self.encoder_model = DonutSwinModel(encoder_config)
        print(pam_student_fmap_dim)
        self.pam = PAM( 
            sequence_mapping_layer_type = pam_sequence_mapping_layer_type,
            student_fmap_dim = pam_student_fmap_dim,
            student_embedding_dim = pam_student_embedding_dim,
            teacher_fmap_dim = pam_teacher_fmap_dim,
            teacher_embedding_dim = pam_teacher_embedding_dim)

    def forward(self,x):
        x = self.encoder_model(x).last_hidden_state
        x = self.pam(x)
        return x 

class SiglipPAM(nn.Module):
    def __init__(
    self,
    encoder_config: AutoConfig,
    pam_sequence_mapping_layer_type: Literal["linear_projection","bilinear","bicubic","nearest-exact"] = "bilinear",
    pam_student_fmap_dim: Tuple[int,int] = (64,64),
    pam_student_embedding_dim: int = 1024,
    pam_teacher_fmap_dim: Tuple[int,int] = (64,64),
    pam_teacher_embedding_dim: int = 1152
    ):
        super().__init__()
        self.encoder_model = SiglipVisionModel(encoder_config)
        print(pam_student_fmap_dim)
        self.pam = PAM( 
            sequence_mapping_layer_type = pam_sequence_mapping_layer_type,
            student_fmap_dim = pam_student_fmap_dim,
            student_embedding_dim = pam_student_embedding_dim,
            teacher_fmap_dim = pam_teacher_fmap_dim,
            teacher_embedding_dim = pam_teacher_embedding_dim)

    def forward(self,x):
        x = self.encoder_model(x).last_hidden_state
        x = self.pam(x)
        return x 