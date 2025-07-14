import sys
from pathlib import Path
parent_root = Path().resolve().parent.parent 
sys.path.append(str(parent_root))


from transformers import PretrainedConfig, DonutSwinConfig, CONFIG_MAPPING, SiglipVisionConfig

from config_divedoc import SwinPamVisionEncoderConfig, PamConfig



class DecoderSegmentationConfig(PretrainedConfig): 
    model_type = "segmentation_decoder"
    def __init__(
        self,
        number_of_class:int=11, 
        embedding_dim: int = 1152,
        fmap_size:tuple=[64,64],
        positional_encoding:bool=False,
        **kwargs,
    ):
        self.number_of_class = number_of_class
        self.embedding_dim = embedding_dim
        self.fmap_size = fmap_size
        self.positional_encoding = positional_encoding
        super().__init__(**kwargs)

    def to_dict(self):
        # Start with the base dictionary
        output = super().to_dict()
        # Define the keys specific to your custom configuration
        custom_keys = ['number_of_class', 'embedding_dim', 'fmap_size',"positional_encoding",'model_type','transformers_version']
        # Filter the dictionary to include only custom keys
        return {k: output[k] for k in custom_keys if k in output}



class SwinQLoRAConfig(PretrainedConfig):
    model_type = "SwinQLoRA"

    def __init__(
        self,
        base_vision_encoder_decoder_name_or_path: str = "",
        peft_adapter_path: str = "",
        **kwargs,
    ):
        """
        Args:
            base_vision_encoder_decoder_name_or_path (str):
                Path or hub ID of the original end‐to‐end VisionEncoderDecoderModel
                (e.g. "google/vit-gpt2-coco", or your local folder).
            peft_adapter_name (str):
                Path (or hub ID) where the PEFT adapter/config is saved,
                e.g. "./my_peft_adapter".
            encoder_hidden_size (int):
                Hidden size of the vision encoder output (for any parent layers you add).
            **kwargs: any extra HF config args you want to accept.
        """
        super().__init__(**kwargs)
        self.base_vision_encoder_decoder_name_or_path = base_vision_encoder_decoder_name_or_path
        self.peft_adapter_path = peft_adapter_path

class SegmentationConfig(PretrainedConfig):
    sub_configs = {"vision_config": SwinPamVisionEncoderConfig, "head_config": DecoderSegmentationConfig}
    model_type = "swinpamsegmentation"
    def __init__(
        self,
        vision_config=None,
        head_config=None,
        semantic_loss_ignore_index=12,
        #_attn_implementation_autoset = True,
        **kwargs,
    ):
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
        self.vision_config = vision_config
        self.head_config = head_config
        #self._attn_implementation_autoset = _attn_implementation_autoset
    

        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "swinpam"
            )
            if vision_config["model_type"] == "swinpam":
                self.vision_config = SwinPamVisionEncoderConfig(encoder_config=vision_config["encoder_config"],pam_config=vision_config["pam_config"])
            elif vision_config["model_type"] == "donut-swin":
                self.vision_config = DonutSwinConfig(**vision_config)
            elif vision_config["model_type"] == "SwinQLoRA":
                self.vision_config = SwinQLoRAConfig(**vision_config)
            else:
                self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
    

        if isinstance(self.head_config, dict):
            if head_config["model_type"] !="segmentation_decoder":
                raise ValueError(f"mode_type should be segmentation_decoder, got {head_config['model_type']}")
                        
            self.head_config = DecoderSegmentationConfig(**head_config)
        
        super().__init__(**kwargs)


def get_original_vision_config( image_size=[2560,1920],
                                sequence_mapping_layer_type= "bilinear",
                                student_fmap_dim=(80,60),
                                student_embedding_dim= 1024,
                                teacher_fmap_dim= (64,64),
                                teacher_embedding_dim= 1152):
    encoder_config = DonutSwinConfig(
        attention_probs_dropout_prob= 0.0,
        depths =[
            2,
            2,
            14,
            2
        ],
        drop_path_rate= 0.1,
        embed_dim =128,
        hidden_act ="gelu",
        hidden_dropout_prob = 0.0,
        hidden_size = student_embedding_dim,
        image_size = image_size,
        initializer_range = 0.02,
        layer_norm_eps = 1e-05,
        mlp_ratio = 4.0,
        model_type = "donut-swin",
        num_channels = 3,
        num_heads =[
            4,
            8,
            16,
            32
        ],
        num_layers =4,
        patch_size = 4,
        path_norm = True,
        qkv_bias = True,
        use_absolute_embeddings = False,
        window_size = 10
        )
    pam_config = PamConfig(
                            sequence_mapping_layer_type= sequence_mapping_layer_type,
                            student_fmap_dim=student_fmap_dim,
                            student_embedding_dim= student_embedding_dim,
                            teacher_fmap_dim= teacher_fmap_dim,
                            teacher_embedding_dim= teacher_embedding_dim)
    swinpam_config = SwinPamVisionEncoderConfig(encoder_config=encoder_config,pam_config=pam_config)
    return swinpam_config

def get_paligemma_ve_config():
    dict_config = { "attention_dropout": 0.0,
                    "hidden_act": "gelu_pytorch_tanh",
                    "hidden_size": 1152,
                    "image_size": 896,
                    "intermediate_size": 4304,
                    "layer_norm_eps": 1e-06,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_channels": 3,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 4096,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    "vision_use_head": False}
    return SiglipVisionConfig(**dict_config)

def get_donut_ve_config():
    config = DonutSwinConfig(
        attention_probs_dropout_prob= 0.0,
        depths =[
            2,
            2,
            14,
            2
        ],
        drop_path_rate = 0.1,
        embed_dim = 128,
        hidden_act = "gelu",
        hidden_dropout_prob = 0.0,
        hidden_size =  1024,
        image_size = [
            2560,
            1920
        ],
        initializer_range = 0.02,
        layer_norm_eps = 1e-05,
        mlp_ratio = 4.0,
        model_type = "donut-swin",
        num_channels = 3,
        num_heads =[
            4,
            8,
            16,
            32
        ],
        num_layers =4,
        patch_size = 4,
        path_norm = True,
        qkv_bias = True,
        use_absolute_embeddings = False,
        window_size = 10
        )
    return config
