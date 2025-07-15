n = 0 #experiment model you want to train
base_model_path = f"./experiments/model_{n}"

model_types = ["swin_qlora","siglip_paligemma","swin_donut"]
model_type = model_types[0] #set the indice of the visual encoder you want to chose


config = {
    "num_classes":16,
    "base_model_path":base_model_path if model_type == "swin_qlora" else None,
    "model_type":model_type,
    "training":{
                "max_epochs":4,
                "learning_rate":3e-4,
                "scheduler":"ReduceLROnPlateau",
                "devices":[2,3], #id of the used gpu
                "batch_size":3,
                "accumulate_grad_batches":1,
                "gradient_clip_val":1.0,
                "precision":'16-mixed',
                "subset_training" : None,
                "subset_validation" : None
                }
}