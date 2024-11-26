# Los Balordos Bros. Presents: Regrasping

This is the best repo ever. NO DOUBTS ABOUT IT.

## Branches
At the moemnt, aside from the master, there are two branches. This readme is meant for the one called ```dev_detection```. I added a new temporary branch to make modifications also for the becker.

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

## Training
The training has been performed following the pipeline described above. At the moment, 3 different training campaigns have been executed:
- the first one (folder called ```train``` inside ```runs```) has been carried out with 200 epochs;
- the second one (results stored in ```train2```) with 1000 epochs;
the third one (```train3```) with just 100 epochs.

Contrary to what expected at the beginning, the best training is the third, probably because the algorithm does not learn specific features of the images used in the training dataset, but learns the most generic ones. 

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


## Verify to have CUDA Installed
First of all, install CUDA by following the instructions at https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/.

To verify if CUDA is correctly installed, run the following line directly in the terminal:

```
python -c "import torch; print(f'Is CUDA available: {torch.cuda.is_available()}')"
``` 

If the output of this line is False, try to install the correct torch version by visiting the site at the following link https://pytorch.org/get-started/locally/. As an example, in my case, the command to run is the following:
``` 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
``` 
If an error occurs, usually it is due to the fact that the path that Windows can handle by default should not have more than 260 characters; if more characters are needed, access the ```register editor``` at the following path: 
``` 
Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
```
At this point, access the variable called ```LongPathsEnabled```, doublt click on it and change it value from 0 to 1. 