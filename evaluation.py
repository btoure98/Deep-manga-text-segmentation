import argparse
import config
import os
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms as T

from data import MangaDataset
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metrics = [
    smp.utils.metrics.Accuracy(activation="sigmoid"),
    smp.utils.metrics.IoU(threshold=0.5, activation="sigmoid"),
    smp.utils.metrics.Precision(activation="sigmoid"),
    smp.utils.metrics.Recall(activation="sigmoid"),
    smp.utils.metrics.Fscore(activation="sigmoid")
]


def test_base(test_path, model_path):
    print("================== Loading model ==================")
    model = torch.load(model_path, map_location=DEVICE)
    model.to(DEVICE)
    print("Ancillary model loaded successfully")
    print("================== Instantiating data loader ==================")
    images_path = os.path.join(test_path, "images")
    images = [os.path.join(images_path, img)
              for img in sorted(os.listdir(images_path))]

    masks_path = os.path.join(test_path, "masks")
    masks = [os.path.join(masks_path, mask)
             for mask in sorted(os.listdir(masks_path))]

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        T.ToTensor()])
    transform_mask = T.Compose([
        T.ToPILImage(),
        T.Grayscale(),
        T.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        T.ToTensor()])

    test_ds = MangaDataset(images, masks, None,
                           transform, transform_mask)
    print("Test data contains ", len(images), "images")
    print("================== Inference ==================")
    loader = DataLoader(test_ds, shuffle=True,
                        batch_size=1,
                        num_workers=os.cpu_count())
    model.eval()
    scores = [0]*len(metrics)
    for (data, target) in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        # Forward Pass
        with torch.no_grad():
            prediction = model(data)
        # Get metrics
        for ind, metric in enumerate(metrics):
            score = metric(target, prediction > 0.7)
            scores[ind] += score.item()
    print("done!")
    print("================== Getting metrics ==================")
    len_loader = len(loader)
    print("Accuracy : ", scores[0]/len_loader)
    print("IoU : ", scores[1]/len_loader)
    print("Precision : ", scores[2]/len_loader)
    print("Recall : ", scores[3]/len_loader)
    print("Fscore : ", scores[4]/len_loader)


def test_ancillary(test_path, model_path):
    print("================== Loading model ==================")
    model = torch.load(model_path, map_location=DEVICE)
    model.to(DEVICE)
    print("Ancillary model loaded successfully")
    print("================== Instantiating data loader ==================")
    images_path = os.path.join(test_path, "images")
    images = [os.path.join(images_path, img)
              for img in sorted(os.listdir(images_path))]

    masks_path = os.path.join(test_path, "masks")
    masks = [os.path.join(masks_path, mask)
             for mask in sorted(os.listdir(masks_path))]

    bbox_path = os.path.join(test_path, "bboxes")
    bboxes = sorted([os.path.join(bbox_path, bbox)
                     for bbox in sorted(os.listdir(bbox_path))])

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        T.ToTensor()])
    transform_mask = T.Compose([
        T.ToPILImage(),
        T.Grayscale(),
        T.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        T.ToTensor()])

    test_ds = MangaDataset(images, masks, bboxes,
                           transform, transform_mask)
    print("Test data contains ", len(images), "images")
    print("================== Inference ==================")
    loader = DataLoader(test_ds, shuffle=True,
                        batch_size=1,
                        num_workers=os.cpu_count())

    model.eval()
    scores = [0]*len(metrics)
    for (data, target, bbox) in loader:
        data, target, bbox = data.to(
            DEVICE), target.to(DEVICE), bbox.to(DEVICE)
        # Forward Pass
        with torch.no_grad():
            prediction = model(data, bbox)
        # Get metrics
        for ind, metric in enumerate(metrics):
            score = metric(target, prediction > 0.7) #binarize
            scores[ind] += score.item()
    print("done!")
    print("================== Getting metrics ==================")
    len_loader = len(loader)
    print("Accuracy : ", scores[0]/len_loader)
    print("IoU : ", scores[1]/len_loader)
    print("Precision : ", scores[2]/len_loader)
    print("Recall : ", scores[3]/len_loader)
    print("Fscore : ", scores[4]/len_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluation of models')
    parser.add_argument(
        'test_folder', help='should contain image, masks and bboxes if necessary')
    parser.add_argument('model_path', help='model pth')
    args = parser.parse_args()
    if "ancillary" in args.model_path:
        test_ancillary(args.test_folder, args.model_path)
    else:
        test_base(args.test_folder, args.model_path)
