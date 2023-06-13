from torch import nn
import pytorch_lightning as pl
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules.heads import VicRegLLocalProjectionHead
from lightly.loss import  VICRegLLoss

class VicRegL(pl.LightningModule):
    def __init__(self, backbone, config:dict):
        super().__init__()
        self.backbone = backbone
        projectionhead_config = config["pretraining"]["method"]["projection_head"]
        self.projection_head = BarlowTwinsProjectionHead(projectionhead_config['input_size'], projectionhead_config['hidden_size'], projectionhead_config['output_size'])
        self.local_projection_head = VicRegLLocalProjectionHead(projectionhead_config['input_size'], int(projectionhead_config['input_size']/4), int(projectionhead_config['input_size']/4))
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.optim = config['optimizer']
        self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.criterion = VICRegLLoss()
        self.config = config
        
    def forward(self, x):
        # print('x:', x.shape)
        x = self.backbone(x)
        # print(x.shape)
        y = self.average_pool(x)
        y = y.flatten(start_dim = 1)
        z = self.projection_head(y)
        y_local = x.permute(0,2,3,1) # (B, D, W, H) to (B, W, H, D)
        z_local = self.local_projection_head(y_local)
        return z,z_local
    
    def training_step(self, batch, batch_index):
        #  (x1, x2),_,_ = batch
        #  print(self.config['transform'])

         views_and_grids = batch[0]
         views = views_and_grids[: len(views_and_grids) // 2]
         grids = views_and_grids[len(views_and_grids) // 2 :]
        #  print(views[0].shape)

         features = [self.forward(view) for view in views]

         loss = self.criterion(
             global_view_features = features[:2],
             global_view_grids = grids[:2],
             local_view_features= features[2:],
             local_view_grids = grids[2:]
         )
         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
         return loss
    
    def configure_optimizers(self):
          #optimizer = self.optim([
         #       {'params': self.parameters(), 'lr': 0.2},      # Set learning rate for weights
         #       {'params': self.bias, 'lr': 0.02},              # Set learning rate for biases
         #       ], lr=0.1)
         optimizer = self.optim(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
         return optimizer
