import os
import segmentation_models_pytorch as smp
import torch

from sklearn.model_selection import train_test_split
from src.data import MangaDataset
from src import config
from torch.utils.data import DataLoader
from src.losses import dice_loss
from src.network import model
from torch.optim import Adam
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images_path = config.images
masks_path =  config.masks

images = [os.path.join(images_path, img) for img in sorted(os.listdir(images_path))]
masks = [os.path.join(masks_path, mask) for mask in sorted(os.listdir(masks_path))]

split = train_test_split(images, masks, test_size=0.2, random_state=42)
test_img, test_masks = split[1], split[3]
train_img, val_img, train_masks, val_masks = train_test_split(split[0],
                                            split[2],
                                            test_size=0.1,
                                            random_state = 42)

transform = T.Compose([
        T.ToPILImage(),
        T.Resize((1001, 1001)),
        T.ToTensor()])
train_ds = MangaDataset(train_img, train_masks, transform)
val_ds = MangaDataset(val_img, val_masks, transform)
test_ds = MangaDataset(test_img, test_masks, transform)

train_loader = DataLoader(train_ds, shuffle=True,
    batch_size=config.BATCH_SIZE,
    num_workers=os.cpu_count())
val_loader = DataLoader(val_ds, shuffle=True,
    batch_size=1,
    num_workers=os.cpu_count())
test_loader = DataLoader(test_ds, shuffle=True,
	batch_size=config.BATCH_SIZE,
	num_workers=os.cpu_count())

# initialize our UNet model
unet = model
unet.encoder.requires_grad = False
# initialize loss function and optimizer
loss = smp.utils.losses.DiceLoss()


opt = Adam(unet.parameters(), lr=config.INIT_LR)

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=opt,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)



best_iou_score = 0.0
train_logs_list, valid_logs_list = [], []

for i in range(0, config.EPOCHS):

    # Perform training & validation
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(val_loader)
    train_logs_list.append(train_logs)
    valid_logs_list.append(valid_logs)

    # Save model if a better val IoU score is obtained
    if best_iou_score < valid_logs['iou_score']:
        best_iou_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')