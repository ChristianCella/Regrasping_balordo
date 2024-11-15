# Los Balordos Bros. Presents: Regrasping

This is the best repo ever. NO DOUBTS ABOUT IT.

## Branches
There are 2 branches as of now.

## Explanation
This repo is part of a more complex structure that aims at picking objects (bottles and cups) from a box and inserting them in specified supports.
The secind step is the regrasping, that relies on a former detection and a latter segmentation of the objects extracted at the first step of the procedure.

## Useful links
The link to the ```ultralytics``` repository for the detection stage is the following:

- https://github.com/ultralytics/ultralytics?tab=readme-ov-file

Moreover, look at the following link for some useful tutorials to investigate the potential of YOLO:
-  https://www.youtube.com/watch?v=5ku7npMrW40

## Requirements
After cloning the repository, the creation of a virtual environment (```conda``` should be avoided) is suggested:
```
python3 -m venv my_venv
``` 
The version of the python interpreter that surely works is 3.8.10, while more recent versions are not guaranteed to work with the packages in the file ```requirements.txt```, that can easily be installed with

```
pip install -r requirements.txt
``` 

## Detection
The script ```main.py``` allows to perform a very simple detection of (all) the objects appearing in an image. Some sample pictures are collected inside the folder ```Images```, and the instance of the YOLO model must be created after specifying the desired weights, that will be downloaded from the original repo and saved in the current directory as ```.pt``` files .