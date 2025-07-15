import torch 
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

class ClsLightModule(pl.LightningModule):
    def __init__(self, config, model,train_data_loader,validation_data_loader,output_hidden_states=False):
        super().__init__()
        self.config = config
        self.model = model.model.eval()
        self.classication_head = model.classication_head
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.criterion = nn.CrossEntropyLoss()
        self.epoch_training_loss = []
        self.epoch_validation_loss = []
    
    def loss_computation(self,batch):
        pixel_values, label = batch

        #forward
        with torch.no_grad():
            output = self.model(pixel_values)
           
        output = self.classication_head(output)        
        loss = self.criterion(output,label)
        print(loss)
        return loss

    def training_step(self, batch):
        loss = self.loss_computation(batch)
        self.epoch_training_loss.append(loss.item())
        self.log("train_loss",loss.item(), prog_bar=True, on_epoch=True,sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        loss_epoch = np.mean(self.epoch_training_loss)
        self.log("train_loss",loss_epoch, prog_bar=True, on_epoch=True,sync_dist=True)
        self.epoch_training_loss.clear()
    
    def validation_step(self, batch):
        val_loss = self.loss_computation(batch)
        self.epoch_validation_loss.append(val_loss.item())
        return val_loss
    
    def on_validation_epoch_end(self):
        val_loss_epoch = np.mean(self.epoch_validation_loss)
        self.log("val_loss", val_loss_epoch, prog_bar=True, on_epoch=True,sync_dist=True)
        self.epoch_validation_loss.clear()


    def configure_optimizers(self):
        trainable_params = list(self.classication_head.parameters()) 
        # Create the optimizer only for trainable parameters
        optimizer = torch.optim.Adam(trainable_params, lr=self.config["training"]["learning_rate"])

        if self.config["training"]["scheduler"] == "stepLR":
            scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config["scheduler_epoch"], gamma=0.1)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        elif self.config["training"]["scheduler"] == "ReduceLROnPlateau": 
            
            scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  mode='min',factor=0.97, patience=1000, cooldown =0, threshold=0.01, threshold_mode='abs')
            return {"optimizer":optimizer, "lr_scheduler":{ "scheduler":scheduler,"interval": "step","monitor":"train_loss","frequency":1}}
            
        return optimizer
 
    def train_dataloader(self):
        return self.train_data_loader
  
    def val_dataloader(self):
        return self.validation_data_loader