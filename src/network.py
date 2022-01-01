import segmentation_models_pytorch as smp

encoder = "resnet34"
encoder_wts = "imagenet"
activation = "sigmoid"

model = smp.Unet(encoder_name=encoder,activation=activation,encoder_weights=encoder_wts)

