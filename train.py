import torch
from timm import create_model
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import lightning as L
from torchmetrics import Metric
import torchmetrics

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class LitClassification(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = create_model('resnet34', num_classes=5)

        total_samples = 25810 + 5292 + 2443 + 873 + 708
        class_weights = total_samples / (5 * torch.tensor([25810, 5292, 2443, 873, 708], dtype=torch.float))
        # Normalized weights
        class_weights = class_weights / class_weights.min()

        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=5)
        self.f1_score = torchmetrics.F1Score(task="multiclass",num_classes=5)

    def training_step(self, batch):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)

        accuracy = self.accuracy(outputs,targets)
        f1_score = self.f1_score(outputs,targets)

        self.log_dict({"train_loss":loss,"train_accuracy":accuracy,"train_f1_score":f1_score},on_step=False,on_epoch=True,prog_bar=True,logger=True)

        # self.log("train_loss", loss)
        return loss
    
    def validation_step(self,batch):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)

        accuracy = self.accuracy(outputs,targets)
        f1_score = self.f1_score(outputs,targets)

        self.log_dict({"val_loss":loss,"val_accuracy":accuracy,"val_f1_score":f1_score},on_step=False,on_epoch=True,prog_bar=True,logger=True)


        # self.log("val_loss", loss)
        return loss
    
    def test_step(self,batch):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)

        accuracy = self.accuracy(outputs,targets)
        f1_score = self.f1_score(outputs,targets)

        self.log_dict({"test_loss":loss,"test_accuracy":accuracy,"test_f1_score":f1_score},on_step=False,on_epoch=True,prog_bar=True,logger=True)


        # self.log("test_loss", loss)
        return loss
    
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=0.005)
    
    
class ClassificationData(L.LightningDataModule):

    def train_dataset(self,train_path):
        return ImageFolder(root=train_path,transform=DEFAULT_TRANSFORM)

    def val_dataset(self,val_path):
        return ImageFolder(root=val_path,transform=DEFAULT_TRANSFORM)

if __name__ == "__main__":
    model = LitClassification()

    data = ClassificationData()
    train_dataset = data.train_dataset("/home/mahmud/Downloads/retinopathy_dataset/data_split/train")
    train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=8)
    val_dataset = data.train_dataset("/home/mahmud/Downloads/retinopathy_dataset/data_split/val")
    val_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=False,num_workers=8)
    
    trainer = L.Trainer(max_epochs=20)
    trainer.fit(model, train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
