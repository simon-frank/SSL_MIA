from torch import nn
import pytorch_lightning as pl
from lightly.models.modules import MoCoProjectionHead
from lightly.loss import  NTXentLoss
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

import copy

class MoCo(pl.LightningModule):
    def __init__(self, backbone, config:dict):
        super().__init__()
        self.backbone = backbone
        projectionhead_config = config["pretraining"]["method"]["projection_head"]
        self.projection_head = MoCoProjectionHead(projectionhead_config['input_size'], projectionhead_config['hidden_size'], projectionhead_config['output_size'])
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        # self.projection_head = nn.Identity()
        # self.projection_head_momentum = nn.Identity()
        

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
        
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.optim = config['optimizer']
        self.criterion = NTXentLoss(memory_bank_size=4096)
        self.config = config
        
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim = 1)
        return self.projection_head(x)
    
    def forward_momentum(self, x):
        x = self.backbone_momentum(x).flatten(start_dim=1)
        return self.projection_head_momentum(x).detach()
    
    def training_step(self, batch, batch_index):
         momentum = cosine_schedule(self.current_epoch, self.config['epochs'], 0.996, 1)
         update_momentum(self.backbone, self.backbone_momentum, m=momentum)
         update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

         x_query, x_key = batch[0]
         
         query = self.forward(x_query)
         key = self.forward_momentum(x_key)

         loss = self.criterion(query, key)

         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
         return loss
    
    def configure_optimizers(self):
          #optimizer = self.optim([
         #       {'params': self.parameters(), 'lr': 0.2},      # Set learning rate for weights
         #       {'params': self.bias, 'lr': 0.02},              # Set learning rate for biases
         #       ], lr=0.1)
         optimizer = self.optim(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
         return optimizer
