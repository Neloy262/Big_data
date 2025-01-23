import torch
from timm import create_model
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import lightning as L

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class LitClassification(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = create_model('resnet34', num_classes=5)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss)
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
    train_dataset = data.train_dataset("/media/mahmud/Work/retinopathy_dataset/data_split/train")
    train_dataloader = DataLoader(train_dataset,batch_size=4,shuffle=True)
    val_dataset = data.train_dataset("/media/mahmud/Work/retinopathy_dataset/data_split/val")
    val_dataloader = DataLoader(train_dataset,batch_size=4,shuffle=True)
    
    trainer = L.Trainer(max_epochs=3)
    trainer.fit(model, train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)