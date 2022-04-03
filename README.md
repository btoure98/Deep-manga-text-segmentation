# Manga text segmentation
## :mag_right: Deep learning


This project is about japanese text segmentation in manga. 
It contains implementation of a of Unet + Resnet34 to perform this task. It also has a semi-supervised approach to enhance state of the art results.

## :file_folder: Project layout

```text
.
├── models                  # Contains weights of the trained neural nets
│   ├── ancillary_model.pth # The ancillary model weights
│   ├── simple_Unet34.pth   # The base unet + resnet34 auto-encorder
├── tools                   # Contains a set of helper scripts
│   ├── get_bbox.py         # Draws bounding boxes on image from xml for manga109 dataset
│   ├── img_cropper.py      # Crops and save image for dataset selection
│   ├── viz_dataset.py      # Side by side image viz to validate segmentation
├── data                    # Contains the financial data for the environment
├── notebooks               # Training and prediction notebooks
├── exemples                # Contains a manga image and the ouput
├── config.py               # Contains training specs for models
├── data.py                 # Pytorch dataset class
├── evaluation.py           # For model testing. Prints out some metrics
├── train_base.py           # A train script for Unet34 model
├── train_ancillary.py      # A train script for ancillary model
├── models.py               # Contains neural nets
├── utils.py                # Some helper functions
├── report.pdf              # Article to get details about the project
├── requirement.txt
├── .gitignore
└── README.md

```

---

## :wrench: How to run the code

1. Install `requirements.txt` in your `Python` environment
2. Run the predict notebook after setting right paths to model etc


