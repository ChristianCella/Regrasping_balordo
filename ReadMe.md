# Los Balordos Bros. Presents: Regrasping

This is the best repo ever. NO DOUBTS ABOUT IT.

## Branches
At the moemnt, aside from the master, there are two branches. This readme is meant for the one called ```dev_detection```.

## Explanation
This repo is part of a more complex structure that aims at picking objects (bottles and beckers) from a box and inserting them in specified supports.
The second step is the regrasping, that relies on a former detection and a latter segmentation of the objects extracted at the first step of the procedure. 
The final aim for the Balordos Bros. will be to produce a single package that takes a label as input (i.e. 'bottle' or 'cup') and uses the GoFa CRB15000 to grasp the object and place it bottom-up in intermedite fixtures, before the final skill-based insertion.

## Detection pipeline
The detection happens before the segmentation. In order to be as flexible as possible, the deep learning-based model YOLO is employed. The iamges of bottles and cups present in the Coco dataset are not enough to obtain a satisfactory detection; therefore, a fine-tuning stage is devised, and it is based on the following steps:
- Create a set of images for the two objects; they don't have to be a huge number, since you can 'artificially' augment them in the following steps (folder called ```Raw_images```); 
- Create the augmented dataset (```Augmented_bottles```): this was manually executed by selecting some 'representative images inside ```Raw_images``` and by running the script ```Augmentation.py``` inside ```Single_codes```.
- At this point, by leveraging the weights ```yolov8l-seg.pt``` available in the ```ultralytics``` package, the dataset should be augmented even more to account for more backgrounds (work in progress).
- The labelling of the images should be done in Roboflow (link: https://app.roboflow.com/christiancella22). At this point two options: by running the code ```Alternative_version.py``` (folder ```From_roboflow```) you can make inference on images by leveraging the weights obtained after a training online (you will not have the folders ```Train-bottle-detection-1``` and ```runs``` in the working directory); the second, is to run in sequence ```Import_labelledf_data.py```, ```Train_model.py``` and ```Inference.py```. 

For the final step, some improvements in terms of relative path is still necessary.

## Useful links
The link to the ```ultralytics``` repository for the detection stage is the following:

- https://github.com/ultralytics/ultralytics?tab=readme-ov-file

Moreover, look at the following link for some useful tutorials to investigate the potential of YOLO:
-  https://www.youtube.com/watch?v=5ku7npMrW40

In general, all the useful links and possible ideas are detailed in the shared Notion page.

## Requirements
The suggestion is to create a virtual environment (```conda``` should be avoided):
```
python3 -m venv my_venv
``` 
The version of the python interpreter that surely works is 3.8.10, while more recent versions are not guaranteed to work with the packages in the file ```requirements.txt```, that can easily be installed with

```
pip install -r requirements.txt
``` 
