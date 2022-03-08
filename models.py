import segmentation_models_pytorch as smp
import torch

from utils import reshape_bbox
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PretrainedUNet34(nn.Module):
    def __init__(self):
        super(PretrainedUNet34, self).__init__()
        encoder = "resnet34"
        encoder_wts = "imagenet"
        activation = "sigmoid"

        self.pretrained_unet34 = smp.Unet(
            encoder_name=encoder, activation=activation, encoder_weights=encoder_wts)
        pretrained_unet34 = self.pretrained_unet34
        self.conv1 = nn.Sequential(pretrained_unet34.encoder.conv1,
                                   pretrained_unet34.encoder.bn1,
                                   pretrained_unet34.encoder.relu
                                   )
        self.conv2 = nn.Sequential(pretrained_unet34.encoder.maxpool,
                                   pretrained_unet34.encoder.layer1
                                   )
        self.conv3 = pretrained_unet34.encoder.layer2
        self.conv4 = pretrained_unet34.encoder.layer3
        self.conv5 = pretrained_unet34.encoder.layer4
        # decoder
        self.center = pretrained_unet34.decoder.center
        decoder_blocks = list(pretrained_unet34.decoder.blocks)
        self.down1 = decoder_blocks[0]
        self.down2 = decoder_blocks[1]
        self.down3 = decoder_blocks[2]
        self.down4 = decoder_blocks[3]
        self.down5 = decoder_blocks[4]
        self.final = pretrained_unet34.segmentation_head

    def forward(self, x):
        x = x
        with torch.no_grad():
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
        center = self.center(conv5)
        down1 = self.down1(center, conv4)
        down2 = self.down2(down1, conv3)
        down3 = self.down3(down2, conv2)
        down4 = self.down4(down3, conv1)
        down5 = self.down5(down4)

        x_out = self.final(down5)

        return x_out


class BboxEncoder(nn.Module):
    def __init__(self):
        super(BboxEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding="same"),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.encoder(x)


class AncillaryModel(nn.Module):
    def __init__(self):
        super(AncillaryModel, self).__init__()
        self.bbox_encoder = BboxEncoder()
        self.bbox_encoder = self.bbox_encoder.to(DEVICE)

        encoder = "resnet34"
        encoder_wts = "imagenet"
        activation = "sigmoid"
        pretrained_unet34 = smp.Unet(
            encoder_name=encoder, activation=activation, encoder_weights=encoder_wts)

        self.conv1 = nn.Sequential(pretrained_unet34.encoder.conv1,
                                   pretrained_unet34.encoder.bn1,
                                   pretrained_unet34.encoder.relu
                                   )
        self.conv2 = nn.Sequential(pretrained_unet34.encoder.maxpool,
                                   pretrained_unet34.encoder.layer1
                                   )
        self.conv3 = pretrained_unet34.encoder.layer2
        self.conv4 = pretrained_unet34.encoder.layer3
        self.conv5 = pretrained_unet34.encoder.layer4
        # decoder
        self.center = pretrained_unet34.decoder.center
        decoder_blocks = list(pretrained_unet34.decoder.blocks)
        self.down1 = decoder_blocks[0]
        self.down2 = decoder_blocks[1]
        self.down3 = decoder_blocks[2]
        self.down4 = decoder_blocks[3]
        self.down5 = decoder_blocks[4]
        self.final = pretrained_unet34.segmentation_head

    def forward(self, x, bbox):
        bbox = torch.unsqueeze(bbox, 1).float()
        bbox = self.bbox_encoder(bbox)
        with torch.no_grad():
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)

        center = self.center(conv5)

        down1 = self.down1(center, torch.mul(
            conv4, reshape_bbox(bbox, conv4.shape)))
        down2 = self.down2(down1, torch.mul(
            conv3, reshape_bbox(bbox, conv3.shape)))
        down3 = self.down3(down2, torch.mul(
            conv2, reshape_bbox(bbox, conv2.shape)))
        down4 = self.down4(down3,  torch.mul(
            conv1, reshape_bbox(bbox, conv1.shape)))
        down5 = self.down5(down4)

        x_out = self.final(down5)

        return x_out
