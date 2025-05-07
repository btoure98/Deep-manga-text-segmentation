# Manga text segmentation
## :mag_right: Deep learning


This project is about japanese text segmentation in manga. 
It contains implementation of a Unet + Resnet34 with semi-supervised approach to enhance state of the art performance.

## :file_folder: Project layout

```text
.
├── models                  # Contains weights of the trained nets
│   ├── ancillary_model.pth 
│   ├── simple_Unet34.pth   
├── tools                   # Contains a set of helper scripts
│   ├── get_bbox.py        
│   ├── img_cropper.py      
│   ├── viz_dataset.py      
├── data                    
├── notebooks               # Training and prediction notebooks
├── examples                # Contains a manga image and the ouput
├── config.py               # Contains training specs for models
├── data.py                 # Pytorch dataset class
├── evaluation.py           # For model testing.
├── train_base.py           # A train script for Unet34 model
├── train_ancillary.py      # A train script for ancillary model
├── models.py              
├── utils.py                # Some helper functions
├── report.pdf              
├── requirement.txt
├── .gitignore
└── README.md

```

---

## Dataset
Manga109 dataset  [Manga 109 website](http://www.manga109.org/en/).
Annotated version for segmentation [zenodo](https://zenodo.org/record/4511796)
https://github.com/juvian. 

## :wrench: How to run the code

1. Install `requirements.txt` in your `Python` environment
2. Run the predict notebook after setting right paths to model etc


