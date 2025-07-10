import numpy as np

import torch 
import torch.nn as nn
import pytorch_lightning as pl

class Distillation(pl.LightningModule):
    def __init__(self, config, model,train_data_loader,validation_data_loader,teacher=None):
        super().__init__()
        self.config = config
        self.student = model
        self.teacher = teacher
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.validation_step_outputs = {"loss":[]}
    
    #function use to compute loss during training (train and validation)
    def compute_loss(self,batch,distill_loss=nn.MSELoss(reduction='mean')):
        #step init/ location set up 1 gpu for the first half of the model, and one other for all the other part
        if self.teacher != None:
            pixel_values, teacher_pixel_values = batch
        else:
            pixel_values, teacher_features_output = batch
        
        #take embedding of student encoder and align it to the same space and patch number to the teacher
        hidden_state = self.student(pixel_values)

        if self.teacher!=None:
            with torch.no_grad():
                teacher_features_output = self.teacher[0](teacher_pixel_values).last_hidden_state

        loss = distill_loss(hidden_state,teacher_features_output)
        return loss

    def training_step(self, batch):
        loss = self.compute_loss(batch)
        self.log_dict({"loss":loss.item()}, sync_dist=True)
        print(loss)
        return loss
    
    def validation_step(self, batch, batch_idx, dataset_idx=0):
        loss = self.compute_loss(batch)
        self.validation_step_outputs['loss'].append(loss.item())
        return {"loss":loss.item()}

    def on_validation_epoch_end(self):
        # I set this to 1 manually
        mean_val_loss = np.mean(self.validation_step_outputs["loss"])
        self.log_dict({"val_loss":mean_val_loss}, sync_dist=True)
        self.validation_step_outputs["loss"].clear()
        return {"val_loss":mean_val_loss}

    def on_epoch_end(self): 
        # Retrieve the current learning rate
        for i, optimizer in enumerate(self.trainer.optimizers):
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                print(f"Epoch {self.current_epoch}: Learning Rate (Optimizer {i}) = {current_lr}")

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.student.parameters())

        # Create the optimizer only for trainable parameters #reessayer factor 0.95 base at 0.80
        optimizer = torch.optim.Adam(trainable_params, lr=self.config.get("lr"))
        scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode='min',factor=0.80, patience=2, cooldown =0, threshold=0.01, threshold_mode='abs')
        return {"optimizer":optimizer, "lr_scheduler":{ "scheduler":scheduler,"interval": "epoch","monitor":"val_loss","frequency":1}}
        
    def train_dataloader(self):
        return self.train_data_loader
  
    def val_dataloader(self):
        return self.validation_data_loader