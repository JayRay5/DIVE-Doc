import torch
import torch.nn as nn

    
class ClsHead(nn.Module):
    def __init__(self,number_patches:int,number_of_class:int, embedding_dim: int = 1152,dropout_prob=0.2) -> None:
        super().__init__()

        self.reductor = nn.Sequential(
                                        nn.Linear(number_patches, number_patches // 2),
                                        nn.ReLU(),
                                        nn.Linear(number_patches // 2, 1)  # learned weights to reduce
)
        self.dropout = nn.Dropout(dropout_prob)
        self.layernorm = (
            nn.LayerNorm(embedding_dim, eps=1e-06) 
        )
        
        self.mlp_head = nn.Sequential(
                                        nn.Linear(embedding_dim, embedding_dim // 2),
                                        nn.ReLU(),
                                        nn.Linear(embedding_dim // 2, number_of_class)
                                    )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.reductor(x.last_hidden_state.permute(0,2,1))
        x = x.permute(0,2,1).squeeze(1)
        x = self.layernorm(x)  
        x = self.dropout(x)
        logits = self.mlp_head(x)
        return logits
    
    
#Wrapper class that add a cls head to a visual encoder (assuming the ve output a sequence of shape (B,N,D), with B:batch_size, N: number_of_patch and D: embedding_size)
class CLSModel(nn.Module):
    def __init__(
        self,
        visual_encoder,
        number_of_class:int,
        patch_sequence_length: int = 4096,
        embedding_dim: int = 1152
        ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.classication_head = ClsHead(number_patches=patch_sequence_length,number_of_class=number_of_class,embedding_dim=embedding_dim)

    def forward(self,x):
        x = self.visual_encoder(x)
        logits = self.classication_head(x)
        return logits
