#put here the path of the DIVE-Doc model folder finetuned until stage 2, which you want ton evaluate 
n=0
base_model_path = f"../../experiments/model_{n}"

#list of models you can evaluate: 
# 0: VE of paligemma, 
# 1: VE of donut, 
# 2: your own DIVE-Doc VE (or donwload from HuggingFace)
#format: [
#           [
#            name,
#            path_for_dive-doc_models, 
#            input_img_size(only for Paligemma & Donut VE), 
#            batch_size, 
#            grad_accumlation_size
#           ],
#          ...]
models = [
    ["siglip_paligemma", None,(896,896),16,1],
    ["swin_donut", None,(2560,1920),1,16],
    ["swin_qlora",base_model_path,None,1,16]]


model = models[2]

config = {
    "epochs":5,
    "learning_rate":9e-4,
    "batch_size":model[3],
    "grad_acc":model[4],
    "encoder_type":model[0],
    "image_size":model[2] if model[0]!="swin_qlora" else None,
    "fmap_dim_bf_pam":model[1] if model[0]!="swin_qlora" else None,
    "base_model_path": base_model_path if model[0]=="swin_qlora" else None,
    "num_class":11,
    "ignore_index":255,
}