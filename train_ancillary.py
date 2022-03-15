import os
import segmentation_models_pytorch as smp
import torch

from data import MangaDataset
import config
from torch.utils.data import DataLoader
from models import AncillaryModel
from torch.optim import Adam
import torchvision.transforms as T
from utils import split_train_val, plot_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# folder paths
images_path = config.images
masks_path = config.masks
bbox_path = config.bboxes

# lists of path
images = sorted([os.path.join(images_path, img)
                 for img in sorted(os.listdir(images_path))])
masks = sorted([os.path.join(masks_path, mask)
                for mask in sorted(os.listdir(masks_path))])
bboxes = sorted([os.path.join(bbox_path, bbox)
                 for bbox in sorted(os.listdir(bbox_path))])


train_img, val_img, train_masks, val_masks, train_boxes, val_boxes = split_train_val((images,
                                                                                     masks, bboxes))
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
    T.ToTensor()])
transform_mask = T.Compose([
    T.ToPILImage(),
    T.Grayscale(),
    T.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
    T.ToTensor()])

train_ds = MangaDataset(train_img, train_masks,
                        train_boxes, transform, transform_mask)
val_ds = MangaDataset(val_img, val_masks, val_boxes, transform, transform_mask)

train_loader = DataLoader(train_ds, shuffle=True,
                          batch_size=config.BATCH_SIZE,
                          num_workers=os.cpu_count())
val_loader = DataLoader(val_ds, shuffle=True,
                        batch_size=1,
                        num_workers=os.cpu_count())

# initialize our UNet model
net = AncillaryModel()
net.to(DEVICE)
#net.encoder.requires_grad = False

# initialize loss function and optimizer
criterion = smp.utils.losses.DiceLoss()
# optimizer
opt = Adam(net.parameters(), lr=config.INIT_LR, weight_decay=config.WEIGHT_DECAY)


train_logs_list, valid_logs_list = [], []
for epoch in range(config.EPOCHS):
    net.train()
    train_loss = 0.0
    for (data, target, bbox) in train_loader:
        data, target, bbox = data.to(
            DEVICE), target.to(DEVICE), bbox.to(DEVICE)
        opt.zero_grad()
        output = net(data, bbox)
        loss = criterion(output, target)
        loss.backward()
        opt.step()
        train_loss += loss.item()

    valid_loss = 0.0
    net.eval()
    for (data, target, bbox) in val_loader:
        data, target, bbox = data.to(
            DEVICE), target.to(DEVICE), bbox.to(DEVICE)
        # Forward Pass
        prediction = net(data, bbox)
        # Find the Loss
        loss = criterion(target, prediction)
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
plot_loss(train_logs_list, valid_logs_list)
