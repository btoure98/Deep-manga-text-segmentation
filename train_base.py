import os
import segmentation_models_pytorch as smp
import torch

from data import MangaDataset
import config
from torch.utils.data import DataLoader
from models import PretrainedUNet34
from torch.optim import Adam
import torchvision.transforms as T
from utils import split_train_val

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# folder paths
images_path = config.images
masks_path = config.masks

# lists of path
images = [os.path.join(images_path, img)
          for img in sorted(os.listdir(images_path))]
masks = [os.path.join(masks_path, mask)
         for mask in sorted(os.listdir(masks_path))]


train_img, val_img, train_masks, val_masks = split_train_val((images,
                                                             masks))
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
    T.ToTensor()])
transform_mask = T.Compose([
    T.ToPILImage(),
    T.Grayscale(),
    T.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
    T.ToTensor()])

train_ds = MangaDataset(train_img, train_masks, None,
                        transform, transform_mask)
val_ds = MangaDataset(val_img, val_masks, None, transform, transform_mask)

train_loader = DataLoader(train_ds, shuffle=True,
                          batch_size=config.BATCH_SIZE,
                          num_workers=os.cpu_count())
val_loader = DataLoader(val_ds, shuffle=True,
                        batch_size=1,
                        num_workers=os.cpu_count())

# initialize our UNet model
net = PretrainedUNet34()
net.to(DEVICE)

# initialize loss function and optimizer
criterion = smp.utils.losses.DiceLoss()
# optimizer
opt = Adam(net.parameters(), lr=config.INIT_LR)

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Precision(),
    smp.utils.metrics.Recall(),
    smp.utils.metrics.Fscore()
]

train_logs_list, valid_logs_list = [], []

for epoch in range(config.EPOCHS):
    net.train()
    train_loss = 0.0
    for ii, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        opt.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        opt.step()
        train_loss += loss.item()

    valid_loss = 0.0
    net.eval()     # Optional when not using Model Specific layer
    for data, target in val_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)

        # Forward Pass
        target = net(data)
        # Find the Loss
        loss = criterion(target, target)
        # Calculate Loss
        valid_loss += loss.item()
    valid_score = valid_loss/len(val_loader)
    train_logs_list.append(train_loss/len(train_loader))
    valid_logs_list.append(valid_score)
    # Save model if a better val IoU score is obtained
    if valid_score <= min(valid_logs_list):
        best_score = valid_score
        torch.save(net, os.path.join(config.models, config.model_name))
        print('Model saved!')
    print('Epoch: {} - Loss: {:.6f} - Validation: {:.6f}'.format(epoch + 1,
                                                                 train_loss /
                                                                 len(train_loader),
                                                                 valid_loss/len(val_loader)))
