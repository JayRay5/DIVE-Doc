#[name,(output_resolution,embedding_dim),image_resolution]
STUDENT = [
            ["swinpam", ((80,60),1024),(2560,1920)],
            ["swinpam", ((64,64),1024),(2048,2048)],
            ["swinpam", ((90,90),1024),(2880,2880)], #((100,100),1024),(3200,3200)
            ["swinpam", ((120,120),1024),(3840,3840)],
            ["swinpam", ((48,48),1024),(1536,1536)],#((32,32),1024),(1024,1024)
            ["swinpam", ((80,80),1024),(2560,2560)],
            ["siglip80m", ((64,64),768),(896,896)],
            ["siglip80m", ((80,80),768),(1120,1120)],
            ["siglip80m", ((48,48),768),(672,672)]
        ]

student_index = 6

patch_alignement_type = ["linear_projection","bilinear","bicubic","nearest-exact"]

patch_alignement_type_index = 1

window_sizes = [6,10,14]
window_size = window_sizes[2]

devices = [0,1]

pretrained_donut_weight = True
PaliGEMMA_FEATURE_DIM =((64,64),1152)
connected_teacher = False


datasets = {"docvqa":"JayRay5/Images_docvqa","infographvqa":"JayRay5/Image_Infographvqa"}
dataset_names = ["docvqa","infographvqa"]
selected_dataset = dataset_names[0]


config = {"max_epochs":20,
          "just_results_generation":False,
          "gradient_clip_val":1.0,
          "lr":3e-4, #8e-4
          "accumulate_grad_batches":16,
          "patch_alignement_type":patch_alignement_type[patch_alignement_type_index] if STUDENT[student_index][1][0]!=(64,64) else None,
          "window_size": window_size if STUDENT[student_index][0] == "swinpam" else None,
          "train_batch_sizes": [1],
          "val_batch_sizes": [1],
          "verbose": True,
          "precision": '16-mixed',
          "devices": devices, #number of GPU for the training
          "dataset": datasets[selected_dataset],
          "student_name":STUDENT[student_index][0],
          "teacher_features_dim":PaliGEMMA_FEATURE_DIM,
          "student_features_dim":STUDENT[student_index][1],
          "student_image_size":STUDENT[student_index][2],
          "donut_weight":pretrained_donut_weight if STUDENT[student_index][0] == "swinpam" else False,
          "connected_teacher":connected_teacher,
          "teacher_features_path":f'../data/{selected_dataset}/{selected_dataset}_paligemma_embeddings.h5' if connected_teacher == False else None,
          "teacher_features_indexing_table_train_path":f'../data/{selected_dataset}/teacher_features_table_train.json' if connected_teacher == False else None,
          "teacher_features_indexing_table_validation_path":f'../data/{selected_dataset}/teacher_features_table_validation.json' if connected_teacher == False else None,
          }

